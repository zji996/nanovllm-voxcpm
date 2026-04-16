# Runtime Baseline

## 当前运行面

这个仓库当前只有一个主要运行面:

- Python 包 `nanovllm_voxcpm/`: 核心推理引擎与模型集成
- `deployment/`: FastAPI 服务封装，负责 HTTP、MP3 流式输出和启动期 LoRA 解析

## 当前模型加载方式

- `VoxCPM.from_pretrained(...)` 同时支持本地目录和 Hugging Face repo id
- 真实架构由模型目录中的 `config.json` 决定
- `deployment` 生命周期里调用 `VoxCPM.from_pretrained(...)`，因此对 `voxcpm` 和 `voxcpm2` 都可工作

## 当前本地推荐基线

- 模型目录: `models/VoxCPM2`
- `models/` 作为仓库内本地调试模型目录，已被 `.gitignore` 忽略
- GPU: `0`
- 服务端口: `8010`
- Docker 默认端口: `8020`
- 环境安装: `uv sync --all-packages --frozen`
- CUDA 架构: `TORCH_CUDA_ARCH_LIST=8.6`
- 注意力后端: `NANOVLLM_ATTENTION_BACKEND=sdpa`
- 模型下载源: 默认 `modelscope` (`OpenBMB/VoxCPM2`)，也支持 `huggingface` (`openbmb/VoxCPM2`)

## 现有已知约束

- 推理路径依赖 CUDA，不支持 CPU-only 运行
- `flash-attn` 现在是可选加速依赖，未安装时会回退到 PyTorch SDPA
- `8000` 端口当前已被其他进程占用，不适合作为默认本地端口
- Docker 构建与运行命令已单独整理到 `docs/reference/docker-deployment.md`
