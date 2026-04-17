import io
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_encode_latents_uses_librosa_resample(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    captured = {}

    class _FakeLLM:
        patch_size = 1

        def encode_latents(self, wav_tensor):
            captured["wav_tensor"] = wav_tensor
            return np.zeros((2, 4), dtype=np.float32)

    server = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    server.encoder_sample_rate = 16000
    server.output_sample_rate = 48000
    server.llm = _FakeLLM()

    def _fake_librosa_load(file_obj, sr, mono):
        assert isinstance(file_obj, io.BytesIO)
        captured["target_sr"] = sr
        captured["mono"] = mono
        wav = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        wav.setflags(write=False)
        return wav, sr

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm2.server.librosa.load", _fake_librosa_load)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        out = server.encode_latents(b"fake-wav-bytes", wav_format="wav")

    assert isinstance(out, bytes)
    assert captured["target_sr"] == 16000
    assert captured["mono"] is False
    assert tuple(captured["wav_tensor"].shape) == (1, 3)
    assert not [warning for warning in caught if "not writable" in str(warning.message)]


def test_get_model_info_uses_output_sample_rate():
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    server = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    server.configured_max_model_len = 8192
    server.encoder_sample_rate = 16000
    server.output_sample_rate = 48000
    server.model_path = "/fake/model"
    server.model_config = type(
        "_ModelConfig",
        (),
        {
            "max_length": 8192,
            "lm_config": type("_LMConfig", (), {"max_position_embeddings": 32768})(),
            "patch_size": 4,
            "audio_vae_config": type(
                "_AudioVAEConfig",
                (),
                {
                    "decoder_rates": [8, 6, 5, 2, 2, 2],
                    "out_sample_rate": 48000,
                },
            )(),
        },
    )()
    server.llm = type("_LLM", (), {"feat_dim": 64, "patch_size": 4})()

    info = server.get_model_info()

    assert info["sample_rate"] == 48000
    assert info["encoder_sample_rate"] == 16000
    assert info["output_sample_rate"] == 48000
    assert info["configured_max_model_len"] == 8192
    assert info["model_max_length"] == 8192
    assert info["max_position_embeddings"] == 32768
    assert info["default_max_generate_length"] == 2000
    assert info["approx_step_audio_seconds"] == pytest.approx(0.16)
    assert info["approx_max_audio_seconds_no_prompt"] == pytest.approx(1310.72)
