from __future__ import annotations

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
        return {}

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
