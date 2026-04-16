from pathlib import Path


def test_dockerfile_defaults_to_sdpa_runtime_image():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("# syntax=docker/dockerfile:1.7")
    assert "ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04" in dockerfile
    assert "NANOVLLM_ATTENTION_BACKEND=sdpa" in dockerfile
    assert "UV_CACHE_DIR=/home/appuser/.cache/uv" in dockerfile
    assert "--mount=type=cache,target=/home/appuser/.cache/uv" in dockerfile
    assert "--no-install-workspace" in dockerfile
    assert "MAX_JOBS=1" in dockerfile
    assert "NVCC_THREADS=1" in dockerfile
    assert "COPY --chown=appuser:appuser pyproject.toml uv.lock README.md LICENSE ./" in dockerfile
    assert "EXPOSE 8020" in dockerfile
    assert 'CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app"' in dockerfile
    assert '"8020"]' in dockerfile
