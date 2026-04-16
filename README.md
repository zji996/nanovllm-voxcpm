# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API (can be wrapped by an HTTP server; see `deployment/README.md`)

This repository contains a Python package (`nanovllm_voxcpm/`) plus an optional FastAPI demo.

## Installation

### Install from PyPI

Core package:

```bash
pip install nano-vllm-voxcpm
```

Or with `uv`:

```bash
uv pip install nano-vllm-voxcpm
```

Note: the optional FastAPI demo service (`deployment/`) is not published on PyPI.

### Prerequisites

- Linux + NVIDIA GPU (CUDA)
- Python >= 3.10
- `flash-attn` is optional; if unavailable the project falls back to PyTorch SDPA

The runtime is GPU-centric (Triton + FlashAttention). CPU-only execution is not supported.

### Install from source (dev)

This repo uses `uv` and includes a lockfile (`uv.lock`).

```bash
uv sync --frozen
```

Dev deps (tests):

```bash
uv sync --frozen --dev
```

Preferred local workflow for this repo:

```bash
./manage.sh setup env
./manage.sh setup model
./manage.sh dev api
```

This downloads `OpenBMB/VoxCPM2` from ModelScope into the repo-local, gitignored `./models/VoxCPM2`
directory, starts the FastAPI service on port `8010`, and defaults to `GPU 0`.
The local service entrypoint also defaults `NANOVLLM_ATTENTION_BACKEND=sdpa`
to avoid requiring `flash-attn` during setup.

Note: `flash-attn` may require additional system CUDA tooling depending on your environment.

`./manage.sh setup model` now defaults to ModelScope. If you want to force the explicit ModelScope entrypoint, use:

```bash
./manage.sh setup modelscope
```

If you want to force Hugging Face instead, use:

```bash
./manage.sh setup huggingface
```

You can still override the source-specific repo or pin a revision:

```bash
MODEL_REPO=OpenBMB/VoxCPM2 MODEL_REVISION=<revision> ./manage.sh setup modelscope
```

## Basic Usage

See `example.py` for an end-to-end async example.

Quickstart:

```bash
uv run python example.py
```

### Load a model

`VoxCPM.from_pretrained(...)` accepts either:

- a local model directory path, or
- a HuggingFace repo id (it will download via `huggingface_hub.snapshot_download`).

The model directory is expected to contain:

- `config.json`
- one or more `*.safetensors` weight files
- `audiovae.pth` (VAE weights)

### Generate (async)

If you call `from_pretrained()` inside an async event loop, it returns an `AsyncVoxCPMServerPool`.

```python
import asyncio
import numpy as np

from nanovllm_voxcpm import VoxCPM


async def main() -> None:
    server = VoxCPM.from_pretrained(
        model="/path/to/VoxCPM",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.92,
    )
    await server.wait_for_ready()

    chunks = []
    async for chunk in server.generate(target_text="Hello world"):
        chunks.append(chunk)  # each chunk is a float32 numpy array

    wav = np.concatenate(chunks, axis=0)
    # Write with the model's sample rate (see your model's AudioVAE config; often 16000)
    # import soundfile as sf; sf.write("out.wav", wav, sample_rate)

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

### Generate (sync)

If you call `from_pretrained()` outside an event loop, it returns a `SyncVoxCPMServerPool`.

```python
import numpy as np

from nanovllm_voxcpm import VoxCPM


server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = []
for chunk in server.generate(target_text="Hello world"):
    chunks.append(chunk)
wav = np.concatenate(chunks, axis=0)
server.stop()
```

### Prompting and reference audio (optional)

The VoxCPM2 server supports these conditioning inputs:

- zero-shot: no prompt or reference audio
- prompt continuation: provide `prompt_latents` + `prompt_text`
- stored prompt: provide a `prompt_id` (via `add_prompt`) and then generate with that id
- reference audio: provide `ref_audio_latents` to add a separate reference-audio condition

`ref_audio_latents` is independent from `prompt_latents`:

- use `prompt_latents` when you want to continue from an existing audio prefix
- use `ref_audio_latents` when you want to provide extra reference audio without treating it as the decode prefix

See the public API in `nanovllm_voxcpm/models/voxcpm2/server.py` for details.

## FastAPI demo

The HTTP server demo is documented separately to keep this README focused:

- `deployment/README.md`
- `docs/reference/docker-deployment.md`
- `docs/reference/cudagraph-runtime.md`

If you want the deployment server dependencies too, use:

```bash
uv sync --all-packages --frozen
```

## Benchmark

The `benchmark/` directory contains an end-to-end inference benchmark that drives
the public server API and reports throughput/latency metrics.

Quick run:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5
```

Use a longer English prompt (~100 words) for more stable results:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt
```

See `benchmark/README.md` for more flags.

### Recommended Runtime Defaults

Current recommended defaults for a dedicated `GPU 0` runtime are:

- `gpu_memory_utilization=0.92`
- `enforce_eager=False`
- `NANOVLLM_ATTENTION_BACKEND=sdpa`

The detailed rationale, smoke commands, and deployment advice live in
`docs/reference/cudagraph-runtime.md`.

### Reference Results (RTX 4090)

All reference numbers in this section are measured on NVIDIA GeForce RTX 4090.

The benchmark reports `RTF_per_req_mean`, defined as the mean over requests of
`(request_wall_time / request_audio_duration)` under the given concurrency.

Test setup:

- GPU: NVIDIA GeForce RTX 4090
- Model: `~/VoxCPM1.5`
- Benchmark: `benchmark/bench_inference.py`
- Runs: `--warmup 1 --iters 5`

Short prompt (`"Hello world."`):

Note: with a very short prompt, the model's stopping behavior can be noisy, so output audio duration (and thus RTF) may have high variance at higher concurrency.

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1741 ± 0.0012 | 0.1741 ± 0.0012 | 0.1918 ± 0.0127 |
| 8 | 0.1804 ± 0.0041 | 0.1807 ± 0.0040 | 0.2353 ± 0.0162 |
| 16 | 0.1870 ± 0.0055 | 0.1878 ± 0.0054 | 0.3009 ± 0.0094 |
| 32 | 0.1924 ± 0.0052 | 0.1932 ± 0.0051 | 0.4055 ± 0.0099 |
| 64 | 0.2531 ± 0.0823 | 0.2918 ± 0.0938 | 0.6755 ± 0.0668 |

Long prompt (`benchmark/target_text_100w_en.txt`):

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1909 ± 0.0102 | 0.1909 ± 0.0102 | 0.0805 ± 0.0007 |
| 8 | 0.1902 ± 0.0021 | 0.1905 ± 0.0021 | 0.1159 ± 0.0004 |
| 16 | 0.2044 ± 0.0050 | 0.2050 ± 0.0051 | 0.1825 ± 0.0007 |
| 32 | 0.2168 ± 0.0034 | 0.2185 ± 0.0032 | 0.3207 ± 0.0022 |
| 64 | 0.3235 ± 0.0063 | 0.3250 ± 0.0064 | 0.5556 ± 0.0033 |

Closed-loop users benchmark (`benchmark/bench_closed_loop_users.py`):

- Model: `~/VoxCPM1.5`
- Command:

```bash
uv run python benchmark/bench_closed_loop_users.py \
  --model ~/VoxCPM1.5 \
  --num-users 60 --warmup-s 5 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt \
  --max-generate-length 2000
```

Results (measured window):

| item | value |
|---|---:|
| sample_rate (Hz) | 44100 |
| users | 60 |
| started | 119 |
| achieved rps | 1.98 |
| ok | 119 |
| err | 0 |

TTFB (seconds, ok requests):

| p50 | p90 | p95 | p99 | mean | stdev |
|---:|---:|---:|---:|---:|---:|
| 0.2634 | 0.3477 | 0.3531 | 0.3631 | 0.2884 | 0.0451 |

RTF (wall/audio, ok requests):

| p50 | p90 | p95 | p99 | mean | stdev |
|---:|---:|---:|---:|---:|---:|
| 0.7285 | 0.7946 | 0.8028 | 0.8255 | 0.6929 | 0.1062 |

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the errors below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', 'base_lm.layers.0.self_attn.qkv_proj.weight', ... , 'stop_proj.weight', 'stop_proj.bias', 'stop_head.weight']
[rank0]:[W1106 07:26:04.469150505 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

It's because nanovllm loads model parameters from `*.safetensors`, but some VoxCPM releases ship weights as `.pt`.

Fix:

- use a safetensors-converted checkpoint (or convert the checkpoint yourself)
- ensure the `*.safetensors` files live next to `config.json` in the model directory
