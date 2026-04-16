from __future__ import annotations

import json
from typing import Any

import numpy as np

from nanovllm_voxcpm.models.base_server import BaseModelServerImpl


class FakeModelConfig:
    def __init__(self, architecture: str, inference_timesteps: int = 0) -> None:
        self.architecture = architecture
        self.inference_timesteps = inference_timesteps

    @classmethod
    def model_validate_json(cls, payload: str) -> "FakeModelConfig":
        data = json.loads(payload)
        return cls(architecture=data["architecture"])


class FakeLoRAModel:
    def __init__(self) -> None:
        self.enabled: bool | None = None
        self.reset_called = False

    def set_lora_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def reset_lora_parameters(self) -> None:
        self.reset_called = True


class FakeEngine:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.feat_dim = 64
        self.patch_size = 4
        self.finished = False
        self.cancelled_seq_id: str | None = None
        self.model_runner = type(
            "FakeRunner",
            (),
            {
                "vae": type("FakeVAE", (), {"sample_rate": 16000, "out_sample_rate": 48000})(),
                "model": FakeLoRAModel(),
            },
        )()

    def encode_latents(self, wav_tensor: Any) -> np.ndarray:
        return np.zeros((8, 64), dtype=np.float32)

    def cancel_sequence(self, seq_id: str) -> None:
        self.cancelled_seq_id = seq_id

    def step(self) -> list[str]:
        return ["step-ok"]

    def is_finished(self) -> bool:
        return self.finished


class FakeServerImpl(BaseModelServerImpl[FakeModelConfig]):
    config_cls = FakeModelConfig
    engine_cls = FakeEngine


class FakeExtendedServerImpl(FakeServerImpl):
    def _init_model_info_from_runner(self, model_runner: Any) -> None:
        self.encoder_sample_rate = int(model_runner.vae.sample_rate)
        self.output_sample_rate = int(model_runner.vae.out_sample_rate)
        self.sample_rate = self.output_sample_rate

    def _get_model_info_extra_fields(self) -> dict[str, int]:
        return {
            "encoder_sample_rate": int(self.encoder_sample_rate),
            "output_sample_rate": int(self.output_sample_rate),
        }


def _write_model_dir(tmp_path) -> str:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architecture": "fake"}', encoding="utf-8")
    return str(model_dir)


def test_base_model_server_impl_exposes_common_runtime_metadata(tmp_path):
    server = FakeServerImpl(
        _write_model_dir(tmp_path),
        inference_timesteps=12,
        max_num_batched_tokens=4096,
        max_model_len=2048,
        devices=[2],
    )

    info = server.get_model_info()

    assert server.health() == {"status": "ok"}
    assert info == {
        "sample_rate": 16000,
        "channels": 1,
        "feat_dim": 64,
        "patch_size": 4,
        "model_path": str(tmp_path / "model"),
    }
    assert server.llm.config.devices == [2]
    assert server.llm.config.model == str(tmp_path / "model")
    assert server.llm.config.model_config.inference_timesteps == 12

    server.cancel("seq-1")
    assert server.llm.cancelled_seq_id == "seq-1"
    assert server.step() == ["step-ok"]
    assert server.is_finished() is False


def test_base_model_server_impl_lora_methods_delegate_to_runner_model(tmp_path, monkeypatch):
    server = FakeServerImpl(_write_model_dir(tmp_path), lora_config=object())

    def fake_load_lora_weights(model: Any, lora_path: str, device: str) -> tuple[list[str], list[str]]:
        assert model is server.llm.model_runner.model
        assert lora_path == "/tmp/fake-lora"
        assert device == "cuda"
        return ["loaded"], ["skipped"]

    monkeypatch.setattr("nanovllm_voxcpm.models.base_server.load_lora_weights", fake_load_lora_weights)

    assert server.set_lora_enabled(True) == {"status": "ok", "lora_enabled": True}
    assert server.llm.model_runner.model.enabled is True
    assert server.load_lora("/tmp/fake-lora") == {"status": "ok", "loaded_keys": 1, "skipped_keys": 1}

    assert server.reset_lora() == {"status": "ok"}
    assert server.llm.model_runner.model.reset_called is True


def test_base_model_server_impl_requires_lora_config(tmp_path):
    server = FakeServerImpl(_write_model_dir(tmp_path))

    for action in (lambda: server.set_lora_enabled(True), lambda: server.load_lora("/tmp/x"), server.reset_lora):
        try:
            action()
        except RuntimeError as exc:
            assert "LoRA is not configured" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError when LoRA is not configured")


def test_base_model_server_impl_supports_extended_model_info(tmp_path):
    server = FakeExtendedServerImpl(_write_model_dir(tmp_path))

    assert server.get_model_info() == {
        "sample_rate": 48000,
        "encoder_sample_rate": 16000,
        "output_sample_rate": 48000,
        "channels": 1,
        "feat_dim": 64,
        "patch_size": 4,
        "model_path": str(tmp_path / "model"),
    }
