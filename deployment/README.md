# VoxCPM FastAPI Service

This folder contains a production-oriented FastAPI wrapper around
`nanovllm_voxcpm.models.voxcpm.server.AsyncVoxCPMServerPool`.

Key properties:

- Stateless API (no `prompt_id`, no prompt pool endpoints)
- No runtime LoRA management endpoints
- `/generate` streams MP3 (`audio/mpeg`) encoded server-side via `lameenc`

## Install (uv)

This repo uses `uv` and `deployment/` is a uv workspace member.

Install workspace dependencies at the repo root:

```bash
uv sync --all-packages --frozen
```

Alternatively, to sync only the deployment service dependencies:

```bash
uv sync --package nano-vllm-voxcpm-deployment --frozen
```

Note: `uv sync --frozen` (without `--all-packages/--package`) only syncs the root package by default.

## Configure

Environment variables:

- `NANOVLLM_MODEL_PATH` (recommended local value `./models/VoxCPM2`; deployment default `~/VoxCPM1.5`)
- MP3 encoding (read at startup):
  - `NANOVLLM_MP3_BITRATE_KBPS` (int, default `192`)
  - `NANOVLLM_MP3_QUALITY` (int, default `2`, allowed `0..2`)
- LoRA startup (optional; instance-level fixed, no runtime switching):
  - `NANOVLLM_LORA_URI` (examples: `file:///...`, `https://...`, `s3://bucket/key`, `hf://repo@rev?path=...`)
  - `NANOVLLM_LORA_ID` (required if `NANOVLLM_LORA_URI` is set)
  - `NANOVLLM_LORA_SHA256` (optional; full-file checksum)
  - `NANOVLLM_CACHE_DIR` (default `~/.cache/nanovllm`)

- Server pool startup (read at startup):
  - `NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS` (int, default `8192`)
  - `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS` (int, default `16`)
  - `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN`
    - if unset, defaults to `max_length` from the model `config.json`
    - for the repo-local `VoxCPM2` model this resolves to `8192`
    - fallback default is `4096` only when the model config cannot be read
  - `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` (float, default `0.92`, allowed `(0, 1]`)
  - `NANOVLLM_SERVERPOOL_ENFORCE_EAGER` (bool, default `false`; accepts `1/0,true/false,yes/no,on/off`)
  - `NANOVLLM_SERVERPOOL_DEVICES` (comma-separated ints, default `0`; e.g. `0,1`)

LoRA checkpoint layout (recommended):

```
step_0002000/
  lora_weights.safetensors
  lora_config.json
```

If `lora_config.json` exists, the service will read `lora_config` from it to initialize LoRA structure.

## Run

Preferred local entrypoint from the repo root:

```bash
./manage.sh dev api
```

This uses:

- model path `./models/VoxCPM2`
- devices `0`
- port `8010`
- attention backend `sdpa`

Manual equivalent:

From the repo root:

```bash
NANOVLLM_MODEL_PATH=./models/VoxCPM2 \
NANOVLLM_SERVERPOOL_DEVICES=0 \
NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION=0.92 \
NANOVLLM_SERVERPOOL_ENFORCE_EAGER=false \
NANOVLLM_ATTENTION_BACKEND=sdpa \
uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8010
```

Alternatively (matches the container entrypoint):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8020
```

OpenAPI:

- local manage.sh path: http://localhost:8010/docs
- container/default uvicorn path: http://localhost:8020/docs
- detailed API reference: `docs/reference/http-api.md`

Runtime tuning notes:

- 推荐默认组合是 `sdpa + enforce_eager=false + gpu_memory_utilization=0.92`
- 当前 `VoxCPM2` deployment 默认会把 `max_model_len` 跟到模型配置里的 `max_length=8192`
- 如果这台机器后续主要用于托管，优先从这组默认开始，不建议先把显存占用直接顶到 `0.95+`
- 更详细的 `CUDA graph` 行为说明和 smoke 命令见 `docs/reference/cudagraph-runtime.md`

## Docker Compose

仓库根目录现在提供了一个开箱即用的 [`compose.yaml`](/home/zji/docker/nanovllm-voxcpm/compose.yaml)。

默认行为:

- 构建 `deployment/Dockerfile`
- 把宿主机 `./models` 挂载到容器 `/models`
- 默认使用 `GPU 0`：`CUDA_VISIBLE_DEVICES=0`
- 服务监听宿主机 `8020`
- 默认运行参数是 `sdpa + enforce_eager=false + gpu_memory_utilization=0.92`
- `max_model_len` 默认跟随模型 `config.json`，对 `VoxCPM2` 会自动取到 `8192`

最小启动方式:

```bash
docker compose up -d --build
```

默认情况下它会读取:

- 模型目录: `./models/VoxCPM2`
- 缓存卷: `nanovllm-cache`

常见自定义:

```bash
CUDA_VISIBLE_DEVICES=0 \
NANOVLLM_MODEL_PATH=/models/VoxCPM2 \
NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION=0.92 \
NANOVLLM_SERVERPOOL_MAX_MODEL_LEN=8192 \
docker compose up -d --build
```

查看状态:

```bash
docker compose ps
docker compose logs -f voxcpm
curl -f http://127.0.0.1:8020/ready
```

停止:

```bash
docker compose down
```

前提:

- 已安装 Docker Compose v2
- 已安装 NVIDIA Container Toolkit
- 宿主机上 `./models/VoxCPM2` 已准备好

## Tests

```bash
uv run pytest deployment/tests -q
```

## Docker (k8s-ready)

This repo ships a multi-stage CUDA image at `deployment/Dockerfile`.

Build from the repo root (important: build context is `.`):

```bash
docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
```

Run:

```bash
docker run --rm --gpus all -p 8020:8020 \
  -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 \
  -e NANOVLLM_CACHE_DIR=/var/cache/nanovllm \
  -v /path/to/models:/models \
  nano-vllm-voxcpm-deployment:latest
```

Health check:

```bash
curl -f http://127.0.0.1:8020/ready
```

Notes:

- GPU: on a GPU node you typically need `--gpus all` (Docker) or the NVIDIA device plugin (k8s).
- The container runs as a non-root user (uid `10001`) and uses `NANOVLLM_CACHE_DIR` for writable cache.
- Probes: use `GET /health` (liveness) and `GET /ready` (readiness).
- The image now exposes `8020` by default to avoid colliding with hosts that already use `8000`.

## Client example

`deployment/client.py` demonstrates calling `/encode_latents` and `/generate` and writes MP3 files:

It expects a prompt audio file at `deployment/prompt_audio.wav`.

```bash
uv run python deployment/client.py
```

Outputs:

- `out_zero_shot.mp3`
- `out_prompted.mp3`

## API

### Health

- `GET /health` (liveness): returns `{"status":"ok"}`
- `GET /ready` (readiness): returns 200 only after the model is loaded

### Info

`GET /info`

Returns model metadata from core (`sample_rate/channels/feat_dim/...`) plus MP3 encoder config.

流式和上限相关的关键字段:

- `model.configured_max_model_len`: 当前服务实例实际采用的 runtime 上下文上限
- `model.model_max_length`: 模型 `config.json` 里的 `max_length`
- `model.max_position_embeddings`: 底层 LM 的位置编码上限
- `model.default_max_generate_length`: API 默认生成步数
- `model.approx_step_audio_seconds`: 单个 generation step 大约产出多少秒音频
- `model.approx_max_audio_seconds_no_prompt`: 零样本且不带额外 prompt 时的大致最长音频时长

### Metrics

`GET /metrics`

Prometheus metrics.

### Encode prompt wav to latents

`POST /encode_latents`

Request body (JSON):

- `wav_base64`: base64-encoded bytes of the *entire audio file* (not a data URI)
- `wav_format`: container format for decoding (e.g. `wav`, `flac`, `mp3`; passed to torchaudio)

Response body (JSON):

- `prompt_latents_base64`: base64-encoded float32 bytes
- `feat_dim`: reshape with `np.frombuffer(bytes, np.float32).reshape(-1, feat_dim)`
- `latents_dtype`: `"float32"`
- `sample_rate`: output sample rate (from the model)
- `channels`: `1`

### Generate (streaming MP3)

`POST /generate`

Request body (JSON):

- `target_text`: required
- Prompt (optional, mutually exclusive):
  - wav prompt: `prompt_wav_base64` + `prompt_wav_format` + `prompt_text`
  - latents prompt: `prompt_latents_base64` + `prompt_text`
  - zero-shot: omit all prompt fields
- Reference audio (optional, mutually exclusive):
  - wav reference: `ref_audio_wav_base64` + `ref_audio_wav_format`
  - latents reference: `ref_audio_latents_base64`

`ref_audio_*` is independent from the prompt fields, so you can combine reference audio with either zero-shot or prompted generation.

Response:

- `Content-Type: audio/mpeg`
- body is a streamed MP3 byte stream, not JSON
- clients should read it incrementally as bytes; do not wait for a final structured payload
- headers:
  - `X-Audio-Sample-Rate`
  - `X-Audio-Channels`

一个最小 `curl` 示例:

```bash
curl -X POST http://127.0.0.1:8020/generate \
  -H 'Content-Type: application/json' \
  -o out.mp3 \
  -d '{
    "target_text": "Please speak clearly.",
    "max_generate_length": 200
  }'
```

一个最小 Python 流式消费示例:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8020/generate",
    json={"target_text": "Please speak clearly.", "max_generate_length": 200},
    stream=True,
)
resp.raise_for_status()

with open("out.mp3", "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```
