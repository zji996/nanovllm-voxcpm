from __future__ import annotations

import math
import os
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from typing_extensions import Literal, TypedDict

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.models.server_runtime import normalize_devices
from nanovllm_voxcpm.utils.loader import load_lora_weights


class HealthResponse(TypedDict):
    status: Literal["ok"]


class BaseModelInfoResponse(TypedDict):
    sample_rate: int
    channels: int
    feat_dim: int
    patch_size: int
    model_path: str


class ExtendedModelInfoResponse(BaseModelInfoResponse, total=False):
    encoder_sample_rate: int
    output_sample_rate: int
    configured_max_model_len: int
    model_max_length: int
    max_position_embeddings: int
    default_max_generate_length: int
    approx_step_audio_seconds: float
    approx_max_audio_seconds_no_prompt: float


TModelConfig = TypeVar("TModelConfig", bound=BaseModel)


class BaseModelServerImpl(Generic[TModelConfig]):
    config_cls: type[TModelConfig]
    engine_cls: type[Any]

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
        lora_config: Any = None,
    ) -> None:
        devices = normalize_devices(devices)
        model_config = self._load_model_config(model_path, inference_timesteps)
        self.model_config = model_config
        self.lora_config = lora_config
        self.model_path = model_path
        self.configured_max_model_len = int(max_model_len)

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

        self.llm = self.engine_cls(engine_config)
        self._init_model_info_from_runner(self.llm.model_runner)

    def _load_model_config(self, model_path: str, inference_timesteps: int) -> TModelConfig:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, encoding="utf-8") as f:
            model_config = self.config_cls.model_validate_json(f.read())
        model_config.inference_timesteps = inference_timesteps
        return model_config

    def _init_model_info_from_runner(self, model_runner: Any) -> None:
        self.sample_rate = int(model_runner.vae.sample_rate)

    def _get_model_info_extra_fields(self) -> dict[str, int]:
        response: dict[str, int | float] = {
            "configured_max_model_len": int(self.configured_max_model_len),
            "default_max_generate_length": 2000,
        }

        model_max_length = getattr(self.model_config, "max_length", None)
        if model_max_length is not None:
            response["model_max_length"] = int(model_max_length)

        lm_config = getattr(self.model_config, "lm_config", None)
        max_position_embeddings = getattr(lm_config, "max_position_embeddings", None)
        if max_position_embeddings is not None:
            response["max_position_embeddings"] = int(max_position_embeddings)

        audio_vae_config = getattr(self.model_config, "audio_vae_config", None)
        patch_size = getattr(self.model_config, "patch_size", None)
        decoder_rates = getattr(audio_vae_config, "decoder_rates", None)
        output_sample_rate = getattr(audio_vae_config, "out_sample_rate", None)
        if (
            isinstance(patch_size, int)
            and patch_size > 0
            and isinstance(output_sample_rate, int)
            and output_sample_rate > 0
            and decoder_rates is not None
        ):
            decoder_chunk_size = math.prod(int(rate) for rate in decoder_rates)
            step_audio_seconds = float(patch_size * decoder_chunk_size) / float(output_sample_rate)
            response["approx_step_audio_seconds"] = step_audio_seconds
            response["approx_max_audio_seconds_no_prompt"] = step_audio_seconds * float(self.configured_max_model_len)

        return response

    def _get_primary_sample_rate(self) -> int:
        sample_rate = getattr(self, "sample_rate", None)
        if sample_rate is not None:
            return int(sample_rate)

        output_sample_rate = getattr(self, "output_sample_rate", None)
        if output_sample_rate is not None:
            return int(output_sample_rate)

        raise AttributeError("sample_rate")

    def _encode_latents_from_tensor(self, wav_tensor: Any) -> bytes:
        latents = self.llm.encode_latents(wav_tensor)
        assert latents.shape[0] % self.llm.patch_size == 0
        return latents.tobytes()

    def _get_lora_model(self) -> Any:
        return self.llm.model_runner.model

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def get_model_info(self) -> ExtendedModelInfoResponse:
        response = ExtendedModelInfoResponse(
            sample_rate=self._get_primary_sample_rate(),
            channels=1,
            feat_dim=int(self.llm.feat_dim),
            patch_size=int(self.llm.patch_size),
            model_path=str(self.model_path),
        )
        response.update(self._get_model_info_extra_fields())
        return response

    def cancel(self, seq_id: str) -> None:
        self.llm.cancel_sequence(seq_id)

    def step(self) -> Any:
        return self.llm.step()

    def is_finished(self) -> bool:
        return self.llm.is_finished()

    def set_lora_enabled(self, enabled: bool) -> dict[str, Any]:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        self._get_lora_model().set_lora_enabled(enabled)
        return {"status": "ok", "lora_enabled": enabled}

    def load_lora(self, lora_path: str) -> dict[str, Any]:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model. Initialize with lora_config.")
        loaded, skipped = load_lora_weights(self._get_lora_model(), lora_path, device="cuda")
        return {"status": "ok", "loaded_keys": len(loaded), "skipped_keys": len(skipped)}

    def reset_lora(self) -> dict[str, Any]:
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        self._get_lora_model().reset_lora_parameters()
        return {"status": "ok"}
