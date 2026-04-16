# Docker Deployment

## 目标

- 使用 `deployment/Dockerfile` 构建 SDPA-first 的部署镜像
- 默认避开宿主机常见占用的 `8000`，容器端口固定为 `8020`
- 保持运行命令与镜像默认入口一致

## 预拉取基础镜像

```bash
docker pull nvidia/cuda:12.6.3-runtime-ubuntu22.04
```

说明:

- 这个基础镜像层较大，单独预拉取通常比反复在 `docker build` 里等待更省时间

## 构建

在仓库根目录执行:

```bash
docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
```

## 运行

```bash
docker run --rm --gpus all -p 8020:8020 \
  -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 \
  -e NANOVLLM_CACHE_DIR=/var/cache/nanovllm \
  -v /path/to/models:/models \
  nano-vllm-voxcpm-deployment:latest
```

## 健康检查

```bash
curl -f http://127.0.0.1:8020/ready
```

OpenAPI:

- `http://127.0.0.1:8020/docs`

## 当前默认值

- 基础镜像: `nvidia/cuda:12.6.3-runtime-ubuntu22.04`
- attention backend: `sdpa`
- 容器入口: `uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8020`
- 构建约束: `MAX_JOBS=1`、`NVCC_THREADS=1`
