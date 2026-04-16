from pathlib import Path


def test_deployment_readme_documents_docker_flow_on_8020():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest ." in readme
    assert "docker run --rm --gpus all -p 8020:8020" in readme
    assert "http://127.0.0.1:8020/ready" in readme
    assert "docs/reference/http-api.md" in readme
