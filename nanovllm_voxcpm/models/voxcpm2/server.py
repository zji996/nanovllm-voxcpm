import io
import os
from typing import AsyncGenerator, cast

import librosa
import numpy as np
import torch
from typing_extensions import Literal, TypedDict

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.models.server_runtime import (
    AsyncServerPool,
    AsyncServerProcess,
    SyncServerPool,
    Waveform,
    normalize_devices,
)
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig, VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine
from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
from nanovllm_voxcpm.utils.loader import load_lora_weights
from nanovllm_voxcpm.utils.torch_numpy import float32_array_from_buffer, torch_from_numpy_writable


class HealthResponse(TypedDict):
    status: Literal["ok"]


class SetLoraEnabledResponse(TypedDict):
    status: Literal["ok"]
    lora_enabled: bool


class LoadLoraResponse(TypedDict):
    status: Literal["ok"]
    loaded_keys: int
    skipped_keys: int


class ResetLoraResponse(TypedDict):
    status: Literal["ok"]


class ModelInfoResponse(TypedDict):
    sample_rate: int
    encoder_sample_rate: int
    output_sample_rate: int
    channels: int
    feat_dim: int
    patch_size: int
    model_path: str


class VoxCPM2ServerImpl:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: list[int] | None = None,
        lora_config: LoRAConfig | None = None,
    ):
        devices = normalize_devices(devices)
        model_config = VoxCPM2Config.model_validate_json(open(os.path.join(model_path, "config.json")).read())
        model_config.inference_timesteps = inference_timesteps
        self.lora_config = lora_config
        self.model_path = model_path

        engine_config = Config(
            model=model_path,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_config=model_config,
            devices=devices,
            lora_config=lora_config,
        )
        self.llm = VoxCPM2Engine(engine_config)
        model_runner = cast(VoxCPM2Runner, self.llm.model_runner)
        self.encoder_sample_rate = model_runner.vae.sample_rate
        self.output_sample_rate = model_runner.vae.out_sample_rate

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def get_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            sample_rate=int(self.output_sample_rate),
            encoder_sample_rate=int(self.encoder_sample_rate),
            output_sample_rate=int(self.output_sample_rate),
            channels=1,
            feat_dim=int(self.llm.feat_dim),
            patch_size=int(self.llm.patch_size),
            model_path=str(self.model_path),
        )

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_np, _ = librosa.load(io.BytesIO(wav), sr=self.encoder_sample_rate, mono=False)
        wav_tensor = torch_from_numpy_writable(wav_np)
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        wav_tensor = wav_tensor.cuda()
        latents = self.llm.encode_latents(wav_tensor)
        assert latents.shape[0] % self.llm.patch_size == 0
        return latents.tobytes()

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

    def cancel(self, seq_id: str):
        self.llm.cancel_sequence(seq_id)

    def step(self):
        return self.llm.step()

    def is_finished(self):
        return self.llm.is_finished()

    def set_lora_enabled(self, enabled: bool) -> SetLoraEnabledResponse:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        model = cast(VoxCPM2Runner, self.llm.model_runner).model
        model.set_lora_enabled(enabled)
        return SetLoraEnabledResponse(status="ok", lora_enabled=enabled)

    def load_lora(self, lora_path: str) -> LoadLoraResponse:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model. Initialize with lora_config.")
        model = cast(VoxCPM2Runner, self.llm.model_runner).model
        loaded, skipped = load_lora_weights(model, lora_path, device="cuda")
        return LoadLoraResponse(status="ok", loaded_keys=len(loaded), skipped_keys=len(skipped))

    def reset_lora(self) -> ResetLoraResponse:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        model = cast(VoxCPM2Runner, self.llm.model_runner).model
        model.reset_lora_parameters()
        return ResetLoraResponse(status="ok")


class AsyncVoxCPM2Server(AsyncServerProcess):
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
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
        gpu_memory_utilization: float = 0.9,
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
        gpu_memory_utilization: float = 0.9,
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
