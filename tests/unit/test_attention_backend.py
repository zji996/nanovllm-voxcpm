import importlib
import sys

import pytest


def _reload_attention_module():
    sys.modules.pop("nanovllm_voxcpm.layers.attention", None)
    return importlib.import_module("nanovllm_voxcpm.layers.attention")


def test_attention_backend_accepts_sdpa(monkeypatch):
    monkeypatch.setenv("NANOVLLM_ATTENTION_BACKEND", "sdpa")

    attention = _reload_attention_module()

    assert attention._resolve_attention_backend() == "sdpa"


def test_attention_backend_rejects_legacy_torch_name(monkeypatch):
    monkeypatch.setenv("NANOVLLM_ATTENTION_BACKEND", "torch")

    attention = _reload_attention_module()

    with pytest.raises(RuntimeError, match="expected auto, flash, or sdpa"):
        attention._resolve_attention_backend()
