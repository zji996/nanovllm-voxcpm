from pathlib import Path


def test_deployment_readme_documents_docker_flow_on_8020():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest ." in readme
    assert "docker run --rm --gpus all -p 8020:8020" in readme
    assert "http://127.0.0.1:8020/ready" in readme
    assert "docs/reference/http-api.md" in readme


def test_env_example_matches_compose_defaults():
    env_example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=0" in env_example
    assert "NANOVLLM_HTTP_PORT=8020" in env_example
    assert "NANOVLLM_MODEL_PATH=/models/VoxCPM2" in env_example
    assert "NANOVLLM_SERVICE_PORT" not in env_example
    assert "TORCH_CUDA_ARCH_LIST" not in env_example
