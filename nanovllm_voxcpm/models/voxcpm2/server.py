import io
from typing import Any, AsyncGenerator

import librosa
import numpy as np
import torch

from nanovllm_voxcpm.models.base_server import BaseModelServerImpl, ExtendedModelInfoResponse
from nanovllm_voxcpm.models.server_runtime import (
    AsyncServerPool,
    AsyncServerProcess,
    SyncServerPool,
    Waveform,
    normalize_devices,
)
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig, VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine
from nanovllm_voxcpm.utils.torch_numpy import float32_array_from_buffer, torch_from_numpy_writable


class ModelInfoResponse(ExtendedModelInfoResponse):
    pass


class VoxCPM2ServerImpl(BaseModelServerImpl[VoxCPM2Config]):
    config_cls = VoxCPM2Config
    engine_cls = VoxCPM2Engine

    def _init_model_info_from_runner(self, model_runner: Any) -> None:
        self.encoder_sample_rate = int(model_runner.vae.sample_rate)
        self.output_sample_rate = int(model_runner.vae.out_sample_rate)
        self.sample_rate = self.output_sample_rate

    def _get_model_info_extra_fields(self) -> dict[str, int]:
        return {
            "encoder_sample_rate": int(self.encoder_sample_rate),
            "output_sample_rate": int(self.output_sample_rate),
        }

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_np, _ = librosa.load(io.BytesIO(wav), sr=self.encoder_sample_rate, mono=False)
        wav_tensor = torch_from_numpy_writable(wav_np)
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        wav_tensor = wav_tensor.cuda()
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
        ref_audio_latents: bytes | None = None,
    ) -> None:
        ref_audio_latents_arr = (
            float32_array_from_buffer(ref_audio_latents, self.llm.feat_dim) if ref_audio_latents is not None else None
        )

        if prompt_latents is None:
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")
            self.llm.add_request(
                seq_id=seq_id,
                target_text=target_text,
                prompt_text="",
                ref_audio_latents=ref_audio_latents_arr,
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
            )
            return

        if len(prompt_text) == 0:
            raise ValueError("Prompt text is required when prompt latents are provided")

        prompt_latents_arr = float32_array_from_buffer(prompt_latents, self.llm.feat_dim)
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents_arr,
            ref_audio_latents=ref_audio_latents_arr,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
        )


class AsyncVoxCPM2Server(AsyncServerProcess):
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
            VoxCPM2ServerImpl,
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
        ref_audio_latents: bytes | None = None,
    ) -> AsyncGenerator[Waveform, None]:
        async for data in self.stream_request(
            target_text,
            prompt_latents,
            prompt_text,
            max_generate_length,
            temperature,
            cfg_value,
            ref_audio_latents,
        ):
            yield data


class AsyncVoxCPM2ServerPool(AsyncServerPool[AsyncVoxCPM2Server]):
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
            AsyncVoxCPM2Server,
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
        ref_audio_latents: bytes | None = None,
    ):
        prompt_latents, prompt_text = self.resolve_prompt_inputs(prompt_latents, prompt_text, prompt_id)

        async with self.borrow_server() as server:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                ref_audio_latents,
            ):
                yield data


class SyncVoxCPM2ServerPool(SyncServerPool[AsyncVoxCPM2ServerPool]):
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
            return AsyncVoxCPM2ServerPool(
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
        ref_audio_latents: bytes | None = None,
    ):
        assert self.loop is not None
        async_gen = self.server_pool.generate(
            target_text,
            prompt_latents,
            prompt_text,
            prompt_id,
            max_generate_length,
            temperature,
            cfg_value,
            ref_audio_latents,
        )
        yield from self.iterate_async_generator(async_gen)
