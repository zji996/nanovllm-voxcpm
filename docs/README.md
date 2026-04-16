# Docs

这个仓库采用轻量三分法文档路由，目标是让人和 AI 都能快速找到当前真实信息。

## 快速入口

1. 先读 `AGENTS.md`
2. 再读 `docs/tasks/README.md`
3. 按当前焦点任务继续读取对应任务文档
4. 需要确认现状时再读 `docs/reference/`

常用参考:

- Docker 构建与运行: `docs/reference/docker-deployment.md`
- HTTP API 参考: `docs/reference/http-api.md`
- 当前运行基线: `docs/reference/runtime-baseline.md`
- CUDA graph 与部署建议: `docs/reference/cudagraph-runtime.md`
- Runtime benchmark 面板: `docs/reference/runtime-benchmarking.md`

当前执行面板:

- `docs/tasks/runtime-performance-roadmap.md`
- `docs/tasks/voxcpm2-local-run.md`

## 文档分区

- `docs/roadmap/`: 为什么这样收敛，记录长期方向和结构性决策
- `docs/tasks/`: 当前在做什么，如何验收
- `docs/reference/`: 系统现在如何运行，写真实实现而不是未来设想

## 治理命令

```bash
./manage.sh check docs
```
