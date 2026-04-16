# Server Runtime 收敛与性能优化路线图

## 背景

这条任务线覆盖三件事:

- 把 `voxcpm` 和 `voxcpm2` 里重复的 server lifecycle 逻辑收敛到共享层
- 给部署默认值、Docker 文档和测试补齐统一基线
- 在不改模型行为的前提下，先做一轮低风险 runtime 优化，并把下一轮性能工作拆成明确批次

目标不是一次性把所有抽象都做完，而是先把“稳定、可测、可继续推进”的骨架搭起来。

## 当前状态

- `Batch 1`: 已完成，等待分批提交
- `Batch 2`: 已完成，等待分批提交
- `Batch 3`: 已完成，等待分批提交
- `Batch 4`: 已完成，等待分批提交
- `Batch 5`: 已完成，等待分批提交
- `Batch 6`: benchmark 面板与实验命令已完成，待填实测数据

## 实施批次

### Batch 1: 本地工作流与文档入口收敛

状态: `已完成`

目标:

- 给仓库补一个稳定的本地开发入口
- 统一 `models/VoxCPM2`、`GPU 0`、`8010`、`sdpa` 这组默认值
- 建立 `docs/roadmap`、`docs/tasks`、`docs/reference` 三分法路由

主要文件:

- `README.md`
- `.gitignore`
- `.env.example`
- `manage.sh`
- `scripts/manage/*`
- `docs/README.md`
- `docs/reference/runtime-baseline.md`
- `docs/roadmap/repo-alignment.md`
- `docs/tasks/voxcpm2-local-run.md`

验收标准:

- 新人只看根 README 就能找到本地启动入口
- `./manage.sh setup env`、`./manage.sh setup model`、`./manage.sh dev api` 可作为推荐路径
- `models/` 明确不进入版本管理
- 文档检查脚本能覆盖最小 docs 路由契约

验证方式:

```bash
./manage.sh check docs
./manage.sh check quick
./manage.sh help
```

### Batch 2: Docker 默认值、部署文档与锁定测试

状态: `已完成`

目标:

- 将部署镜像收敛到 `sdpa-first` 的默认运行面
- 把 Docker 默认端口从 `8000` 统一到 `8020`
- 明确镜像入口、健康检查和推荐运行命令

主要文件:

- `deployment/Dockerfile`
- `deployment/README.md`
- `docs/reference/docker-deployment.md`
- `deployment/tests/test_dockerfile_defaults.py`
- `deployment/tests/test_deployment_docs.py`

验收标准:

- Dockerfile 默认使用 runtime CUDA 镜像
- 容器暴露端口与运行命令一致
- 文档中的 build/run/ready 命令可直接复用
- 关键默认值由测试锁定

验证方式:

```bash
uv run pytest deployment/tests/test_dockerfile_defaults.py deployment/tests/test_deployment_docs.py -q
uv run python -m compileall deployment docs
```

### Batch 3: 共享 server lifecycle 抽取

状态: `已完成`

目标:

- 把主 loop、子进程 RPC、ready 握手、stop/kill、prompt pool、负载选择统一到共享 runtime
- 让 `voxcpm` 和 `voxcpm2` 只保留模型差异
- 后续需要修改 server/pool 行为时只改一处

主要文件:

- `nanovllm_voxcpm/models/server_runtime.py`
- `nanovllm_voxcpm/models/voxcpm/server.py`
- `nanovllm_voxcpm/models/voxcpm2/server.py`
- `nanovllm_voxcpm/llm.py`

验收标准:

- `voxcpm` / `voxcpm2` 的 async/sync server 行为保持一致
- `prompt_id`、LoRA、`encode_latents`、`get_model_info` 都还能走通
- 共享层对空 server pool、ready 失败、非正常退出有清晰行为

验证方式:

```bash
uv run pytest tests/unit/test_llm_from_pretrained.py tests/unit/test_voxcpm2_server_encode_latents.py deployment/tests/test_deployment_app.py deployment/tests/test_core_config.py deployment/tests/test_routes_additional_coverage.py -q
uv run python -m compileall nanovllm_voxcpm/models/server_runtime.py nanovllm_voxcpm/models/voxcpm/server.py nanovllm_voxcpm/models/voxcpm2/server.py
```

### Batch 4: 低风险 runtime 热路径优化

状态: `已完成`

目标:

- 先收掉明显的重复 CPU 开销和读写告警
- 避免每次 HTTP 请求都重复走 metadata 子进程 RPC
- 减少 runner decode 路径上的重复 dtype/device 搬运

主要文件:

- `nanovllm_voxcpm/models/server_runtime.py`
- `nanovllm_voxcpm/models/voxcpm/runner.py`
- `nanovllm_voxcpm/models/voxcpm2/runner.py`
- `nanovllm_voxcpm/utils/torch_numpy.py`
- `deployment/app/api/routes/generate.py`
- `tests/unit/test_server_runtime.py`
- `tests/unit/test_torch_numpy.py`
- `tests/unit/test_voxcpm2_chunk_sizes.py`
- `tests/unit/test_voxcpm2_server_encode_latents.py`

验收标准:

- `get_model_info()` 在 pool ready 后可复用缓存
- 只读 NumPy buffer 不再触发 `torch.from_numpy` 的 writable warning
- `feat`、`temperature`、`cfg_value` 的输入搬运不再走多次转换
- 输出行为与现有测试期望保持一致

验证方式:

```bash
uv run pytest tests/unit/test_server_runtime.py tests/unit/test_torch_numpy.py tests/unit/test_voxcpm2_server_encode_latents.py tests/unit/test_voxcpm2_chunk_sizes.py deployment/tests/test_deployment_app.py deployment/tests/test_routes_additional_coverage.py -q
uv run python -m compileall nanovllm_voxcpm/models/server_runtime.py nanovllm_voxcpm/models/voxcpm/runner.py nanovllm_voxcpm/models/voxcpm2/runner.py
```

### Batch 5: 共享层下一步抽象

状态: `已完成`

目标:

- 继续压缩 `VoxCPMServerImpl` / `VoxCPM2ServerImpl` 的重复
- 把模型级 metadata、LoRA plumbing、`encode_latents` 前后处理抽到更薄的共享层

建议拆分:

1. 抽 `BaseModelServerImpl`，统一 `health/get_model_info/cancel/step/is_finished`
2. 只把模型特有的 config、audio loader、engine factory 留在子类
3. 审视 `ModelInfoResponse` 差异，决定是共用 schema 还是维持扩展字段

验收标准:

- server impl 层重复显著下降，但不牺牲可读性
- `voxcpm2` 的 `encoder_sample_rate/output_sample_rate` 等扩展信息仍可表达
- LoRA 和 prompt 流程不回退

风险:

- 如果抽象过头，模型差异会被隐藏，后续排障反而更慢
- `voxcpm` 与 `voxcpm2` 的音频编码依赖不同，不能强行合并成一条路径

### Batch 6: 性能基线与第二轮优化

状态: `benchmark 面板已完成，实测待执行`

目标:

- 用真实指标判断下一轮优化优先级，而不是只凭代码观感
- 把 HTTP 层、子进程 IPC、GPU decode 路径分别量出来

本批次当前交付:

- `benchmark/bench_inference.py` 已支持 zero-shot / prompt latents / reference latents 三类场景
- 新增 `benchmark/bench_runtime_matrix.py`，可批量扫描 `coalesce_ms`、`recv_queue_mode`、模型和并发
- `AsyncServerProcess` 已支持 `NANOVLLM_RECV_QUEUE_MODE=bridge|to_thread`，默认 `bridge`
- 新增 `docs/reference/runtime-benchmarking.md` 作为复现实验入口

建议任务:

1. 建一个最小 benchmark 面板，记录 `TTFB`、总生成时长、音频秒吞吐、GPU 利用率
2. 比较 `NANOVLLM_QUEUE_COALESCE_MS` 在 `0 / 1 / 2 / 5` 下的影响
3. 评估 `recv_queue` 从 `asyncio.to_thread(queue.get)` 换成常驻 bridge thread 的收益
4. 评估 prompt/ref latent 复用场景，决定是否需要服务端缓存或显式 `prompt_id` 推广
5. 检查 `AudioVAE.decode` 是否已经成为 decode 主瓶颈

建议验证矩阵:

- 模型: `VoxCPM1.5`、`VoxCPM2`
- 并发: `1 / 2 / 4`
- 场景: zero-shot、prompt latents、reference latents
- 输出: 平均值 + p95 + 失败案例备注

关闭条件:

- 有一份可复现的 benchmark 命令集
- 下一轮优化项按收益/风险排序
- 不再凭主观猜测决定优化顺序

## 提交建议

为了让历史更清晰，建议按下面顺序提交:

1. `docs/manage` 基线与任务路由
2. `deployment` Docker 默认值与文档测试
3. 共享 `server_runtime` 抽取
4. `sdpa` / `torch_numpy` / runner 热路径优化

## 回归检查清单

- README、deployment README、docs 入口互相可跳转
- `./manage.sh check docs` 通过
- 受影响的 pytest 子集通过
- `compileall` 通过
- Docker 默认端口、服务默认端口、本地默认端口三者不混淆
