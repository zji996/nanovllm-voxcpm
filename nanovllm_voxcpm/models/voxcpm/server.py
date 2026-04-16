import io

import numpy as np
import torchaudio
from numpy.typing import NDArray
from typing import AsyncGenerator

from nanovllm_voxcpm.models.base_server import BaseModelInfoResponse, BaseModelServerImpl
from nanovllm_voxcpm.models.server_runtime import (
    AsyncServerPool,
    AsyncServerProcess,
    SyncServerPool,
    normalize_devices,
)
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMConfig, VoxCPMEngine
from nanovllm_voxcpm.utils.torch_numpy import float32_array_from_buffer

Waveform = NDArray[np.float32]


class ModelInfoResponse(BaseModelInfoResponse):
    pass


class VoxCPMServerImpl(BaseModelServerImpl[VoxCPMConfig]):
    config_cls = VoxCPMConfig
    engine_cls = VoxCPMEngine

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_tensor, sr = torchaudio.load(io.BytesIO(wav), format=wav_format)
        wav_tensor = wav_tensor.cuda()
        if sr != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, self.sample_rate)

        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

        return self._encode_latents_from_tensor(wav_tensor)

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
    ) -> None:
        if prompt_latents is None:
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")
            self.llm.add_request(
                seq_id=seq_id,
                target_text=target_text,
                prompt_text="",
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
            )
            return

        if len(prompt_text) == 0:
            raise ValueError("Prompt text is required when prompt latents are provided")

        # Help static type checkers: prompt_latents is non-None here.
        assert prompt_latents is not None
        prompt_latents_buf: bytes = prompt_latents
        prompt_latents_arr: np.ndarray = float32_array_from_buffer(prompt_latents_buf, self.llm.feat_dim)
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents_arr,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
        )


class AsyncVoxCPMServer(AsyncServerProcess):
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: LoRAConfig | None = None,
        **kwargs,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unknown kwargs: {kwargs}")
        super().__init__(
            VoxCPMServerImpl,
            (
                model_path,
                inference_timesteps,
                max_num_batched_tokens,
                max_num_seqs,
                max_model_len,
                gpu_memory_utilization,
                enforce_eager,
                normalize_devices(devices),
                lora_config,
            ),
        )

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
    ) -> AsyncGenerator[Waveform, None]:
        async for data in self.stream_request(
            target_text,
            prompt_latents,
            prompt_text,
            max_generate_length,
            temperature,
            cfg_value,
        ):
            yield data


class AsyncVoxCPMServerPool(AsyncServerPool[AsyncVoxCPMServer]):
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: LoRAConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            AsyncVoxCPMServer,
            model_path=model_path,
            inference_timesteps=inference_timesteps,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            devices=devices,
            lora_config=lora_config,
            **kwargs,
        )

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
    ):
        """Generate audio conditioned on text and optional prompt.

        This is an async generator that yields waveform chunks (one chunk per
        model step) as NumPy arrays.

        Exactly one of the following prompt sources may be used:

        - Provide ``prompt_latents`` + matching ``prompt_text``.
        - Provide ``prompt_id`` (a previously-added prompt via ``add_prompt``).
        - Provide no prompt (zero-shot).

        Args:
            target_text: Text to synthesize.
            prompt_latents: Serialized prompt latents (float32 bytes). If set,
                ``prompt_text`` must be non-empty.
            prompt_text: Text corresponding to ``prompt_latents``.
            prompt_id: ID of a stored prompt from ``add_prompt``. Mutually
                exclusive with ``prompt_latents`` and ``prompt_text``.
            max_generate_length: Maximum number of generated steps.
            temperature: Sampling temperature.
            cfg_value: Classifier-free guidance scale.

        Yields:
            Waveform chunks as ``np.ndarray`` of dtype ``float32``.

        Raises:
            ValueError: If prompt arguments are inconsistent (e.g. unknown
                ``prompt_id``, or both ``prompt_id`` and ``prompt_latents`` are
                provided).
        """
        prompt_latents, prompt_text = self.resolve_prompt_inputs(prompt_latents, prompt_text, prompt_id)

        async with self.borrow_server() as server:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
            ):
                yield data


class SyncVoxCPMServerPool(SyncServerPool[AsyncVoxCPMServerPool]):
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: LoRAConfig | None = None,
        **kwargs,
    ):
        async def init_async_server_pool():
            return AsyncVoxCPMServerPool(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=normalize_devices(devices),
                lora_config=lora_config,
                **kwargs,
            )

        super().__init__(init_async_server_pool)

    def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
    ):
        """Generate audio conditioned on text and optional prompt.

        This is a synchronous generator wrapper around
        ``AsyncVoxCPMServerPool.generate``.

        Args:
            target_text: Text to synthesize.
            prompt_latents: Serialized prompt latents (float32 bytes). If set,
                ``prompt_text`` must be non-empty.
            prompt_text: Text corresponding to ``prompt_latents``.
            prompt_id: ID of a stored prompt from ``add_prompt``. Mutually
                exclusive with ``prompt_latents`` and ``prompt_text``.
            max_generate_length: Maximum number of generated steps.
            temperature: Sampling temperature.
            cfg_value: Classifier-free guidance scale.

        Yields:
            Waveform chunks as ``np.ndarray`` of dtype ``float32``.

        Raises:
            ValueError: If prompt arguments are inconsistent.
        """
        assert self.loop is not None
        async_gen = self.server_pool.generate(
            target_text,
            prompt_latents,
            prompt_text,
            prompt_id,
            max_generate_length,
            temperature,
            cfg_value,
        )
        yield from self.iterate_async_generator(async_gen)
