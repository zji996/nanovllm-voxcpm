# Runtime Benchmarking

这份文档记录当前可复现的 runtime benchmark 面板和推荐命令。

## 目标

- 用统一命令量化 `TTFB`、总生成时长、音频秒吞吐和 GPU 利用率
- 对比 `NANOVLLM_QUEUE_COALESCE_MS` 在 `0 / 1 / 2 / 5` 下的影响
- 对比 `NANOVLLM_RECV_QUEUE_MODE=bridge|to_thread` 的 IPC 接收开销
- 覆盖 zero-shot、prompt latents、reference latents 三类场景

## 单次基准

Zero-shot:

```bash
uv run python benchmark/bench_inference.py \
  --model ~/VoxCPM1.5 \
  --devices 0 \
  --concurrency 1 \
  --warmup 1 \
  --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt
```

Prompt latents 复用:

```bash
uv run python benchmark/bench_inference.py \
  --model ~/VoxCPM2 \
  --devices 0 \
  --concurrency 1 \
  --warmup 1 \
  --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt \
  --prompt-wav-file ./assets/prompt.wav \
  --prompt-wav-format wav \
  --prompt-text "Hello from the prompt audio."
```

Reference latents 复用:

```bash
uv run python benchmark/bench_inference.py \
  --model ~/VoxCPM2 \
  --devices 0 \
  --concurrency 1 \
  --warmup 1 \
  --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt \
  --ref-audio-wav-file ./assets/reference.wav \
  --ref-audio-wav-format wav
```

说明:

- `bench_inference.py` 会把 prompt/reference wav 只编码一次，然后复用 latents 进入正式测量
- `--gpu-sample-interval-ms` 默认为 `500`，会用 `nvidia-smi` 采样 GPU utilization；设为 `0` 可关闭
- `--json-out` 可写出单次实验的结构化结果

## 矩阵面板

推荐直接跑 matrix 脚本，把模型、并发、队列收敛参数和 IPC 接收模式一起扫出来:

```bash
uv run python benchmark/bench_runtime_matrix.py \
  --model ~/VoxCPM1.5 \
  --model ~/VoxCPM2 \
  --devices 0 \
  --concurrency-values 1,2,4 \
  --queue-coalesce-values 0,1,2,5 \
  --recv-queue-modes bridge,to_thread \
  --scenarios zero-shot,prompt-latents,reference-latents \
  --target-text-file benchmark/target_text_100w_en.txt \
  --prompt-wav-file ./assets/prompt.wav \
  --prompt-wav-format wav \
  --prompt-text "Hello from the prompt audio." \
  --ref-audio-wav-file ./assets/reference.wav \
  --ref-audio-wav-format wav
```

输出:

- 每个组合会生成一个单独的 `*.json`
- 汇总结果写到 `benchmark/results/.../matrix.json`
- 终端会打印一张简表，列出 `wall_s`、`audio_s/s`、`ttfb_p95`、`gpu_util` 和失败数

## Runtime 开关

- `NANOVLLM_QUEUE_COALESCE_MS`
  - server 子进程在主 loop 中做 queue coalescing 的窗口
  - 推荐纳入基线矩阵: `0 / 1 / 2 / 5`

- `NANOVLLM_RECV_QUEUE_MODE`
  - `bridge`: 常驻 bridge thread 从 `queue_out.get()` 拉数据，再投递回 asyncio loop
  - `to_thread`: 旧路径，每轮 `recv_queue()` 通过 `asyncio.to_thread(queue.get)` 取消息

如果不设置，当前默认值是 `bridge`。

## 指标定义

- `wall_s`: 一轮并发请求全部结束的总墙钟时长
- `audio_s_total`: 本轮生成出来的总音频秒数
- `audio_seconds_per_second`: `audio_s_total / wall_s`，可直接看作音频秒吞吐
- `TTFB p95`: 每轮请求首次收到音频 chunk 的 p95
- `gpu_util`: 基于 `nvidia-smi` 轮询的平均 / p95 / max GPU 利用率
- `failed_requests_total`: 测量轮次中失败请求总数

## 结果记录建议

建议至少保留下面这组矩阵:

- 模型: `VoxCPM1.5`, `VoxCPM2`
- 并发: `1, 2, 4`
- 场景: `zero-shot`, `prompt-latents`, `reference-latents`
- 队列参数: `coalesce_ms=0,1,2,5`
- IPC 模式: `bridge`, `to_thread`

提交 benchmark 结果时，建议把下面三类结论一起写出来:

1. `queue_coalesce_ms` 的最佳区间和副作用
2. `bridge` 相比 `to_thread` 在 TTFB / 吞吐上的实际收益
3. 哪个模型或场景已经主要受限于 `AudioVAE.decode` 或 GPU decode 路径
