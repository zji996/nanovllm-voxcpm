# CUDA Graph Runtime Notes

这份文档记录当前仓库里 `CUDA graph` 路径的真实状态、推荐默认值，以及 GPU 0 专机托管时的建议配置。

## 当前结论

- `decode` 路径已经支持 `CUDA graph capture/replay`
- 推荐默认组合:
  - `enforce_eager=False`
  - `gpu_memory_utilization=0.92`
  - `NANOVLLM_ATTENTION_BACKEND=sdpa`
- 这组默认值已经在本地 `VoxCPM2 + GPU 0` 上做过最小真实 smoke

## 运行机制

- `prefill` 仍然走 eager
- `decode` 在下面条件满足时会走 graph replay:
  - `enforce_eager=False`
  - 当前 step 不是 prefill
  - batch size `<= 512`
- graph 只会在启动期 capture 一次，后续 decode step 复用

这意味着:

- `enforce_eager=True` 更保守，但会放弃 decode graph 的 launch overhead 优化
- `enforce_eager=False` 才能吃到当前这条优化路径

## 为什么推荐 `sdpa`

当前仓库里 `flash-attn` 是可选依赖，`sdpa` 更容易在本地和部署环境里稳定落地。

对现在这条 `CUDA graph` 路径来说，`sdpa` 的优势是:

- 本仓库已经为 `VoxCPM2 + SDPA decode` 修过 graph-safe 路径
- 本地开发和服务部署不需要额外处理 `flash-attn` 安装
- 当机器主要用于托管时，减少一层可变依赖会更稳

如果后续要切回 `flash`，建议单独做一轮回归，而不是默认直接切换。

## 为什么默认 `gpu_memory_utilization=0.92`

当前默认值从 `0.9` / `0.95` 收敛到 `0.92`，目的是在“尽量多给 KV cache”和“给模型加载、图捕获、服务运行留余量”之间取一个更稳的中间值。

对 GPU 0 专机托管来说，`0.92` 的特点是:

- 通常比 `0.9` 多拿到一点 KV cache 空间
- 比 `0.95` 更不容易在启动期或运行期贴着显存上限
- 对 `CUDA graph` 这种需要启动期先 warmup/capture 的路径更友好

如果后续机器负载非常单一，也可以继续手动上调，但建议以 `0.92` 作为默认基线。

## 推荐启动方式

Python API:

```python
from nanovllm_voxcpm import VoxCPM

server = VoxCPM.from_pretrained(
    model="/path/to/VoxCPM2",
    devices=[0],
    gpu_memory_utilization=0.92,
    enforce_eager=False,
)
```

FastAPI service:

```bash
NANOVLLM_MODEL_PATH=./models/VoxCPM2 \
NANOVLLM_SERVERPOOL_DEVICES=0 \
NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION=0.92 \
NANOVLLM_SERVERPOOL_ENFORCE_EAGER=false \
NANOVLLM_ATTENTION_BACKEND=sdpa \
uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8010
```

## 最小 smoke

下面这条命令可以直接验证:

- 服务能初始化
- 启动期 graph capture 没失败
- 生成期 decode replay 能正常出音频 chunk

```bash
NANOVLLM_ATTENTION_BACKEND=sdpa \
uv run python benchmark/bench_inference.py \
  --model ./models/VoxCPM2 \
  --devices 0 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.92 \
  --concurrency 1 \
  --warmup 0 \
  --iters 1 \
  --max-generate-length 3 \
  --target-text 'Please speak clearly and stop after a very short sentence.' \
  --gpu-sample-interval-ms 0
```

如果想顺手确认“非精确 capture 档位”的 replay，也可以再跑一轮:

```bash
NANOVLLM_ATTENTION_BACKEND=sdpa \
uv run python benchmark/bench_inference.py \
  --model ./models/VoxCPM2 \
  --devices 0 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.92 \
  --concurrency 3 \
  --warmup 0 \
  --iters 1 \
  --max-generate-length 2 \
  --target-text 'Please speak clearly and stop after a very short sentence.' \
  --gpu-sample-interval-ms 0
```

## 常见问题

### 1. `enforce_eager` 会不会更快

通常不会。

`enforce_eager=True` 的价值是“更保守、更容易避开 graph 兼容性问题”，不是更高上限。只要 graph 路径已经跑通，decode 吞吐上限一般还是 `enforce_eager=False` 更值得作为默认。

### 2. 启动期就报错

优先检查:

- 显存是否够
- 是否错误切到了 `flash` 后端
- `max_model_len` / `max_num_seqs` / `gpu_memory_utilization` 是否给得过激进

推荐先退到这组排查基线:

- `NANOVLLM_ATTENTION_BACKEND=sdpa`
- `gpu_memory_utilization=0.92`
- `max_num_seqs=4`
- `max_model_len=1024`

### 3. 运行中 OOM

优先降这些:

- `max_num_seqs`
- `max_model_len`
- `max_num_batched_tokens`
- `gpu_memory_utilization`

如果只是想先恢复可用，也可以临时开启 `enforce_eager=True` 做保底，但这不应该作为长期默认。
