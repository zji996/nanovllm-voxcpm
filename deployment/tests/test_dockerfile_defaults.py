from pathlib import Path


def test_dockerfile_defaults_to_sdpa_runtime_image():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04" in dockerfile
    assert "NANOVLLM_ATTENTION_BACKEND=sdpa" in dockerfile
    assert "MAX_JOBS=1" in dockerfile
    assert "NVCC_THREADS=1" in dockerfile
    assert "uv sync --all-packages --frozen --no-dev" in dockerfile
    assert "EXPOSE 8020" in dockerfile
    assert 'CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app"' in dockerfile
    assert '"8020"]' in dockerfile
