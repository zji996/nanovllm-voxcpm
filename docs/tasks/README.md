# Tasks

## 🎯 当前焦点

- [Server Runtime 收敛与性能优化路线图](./runtime-performance-roadmap.md)
- [VoxCPM2 本地下载与 GPU0 服务启动](./voxcpm2-local-run.md)

## 进行中

- `Batch 1`: 本地工作流和 docs 路由收敛
- `Batch 2`: Docker 默认值、部署说明和测试锁定
- `Batch 3`: `voxcpm` / `voxcpm2` 共享 server lifecycle 抽取
- `Batch 4`: 低风险 runtime 热路径优化与只读 NumPy 兼容修正
- `Batch 5`: 建立性能基线与下一轮优化任务面板

## 已完成

- 确认 `deployment` 服务可通过 `config.json` 自动分发到 `voxcpm` 或 `voxcpm2`
- 确认 `8000` 端口已被占用，本地默认端口改为 `8010`
- Docker 部署文档已收敛到 `docs/reference/docker-deployment.md`，容器默认端口改为 `8020`
- 确认 `GPU 0` 空闲充足，适合作为默认推理卡

## 状态规则

- `进行中`: 当前回合正在推进
- `已完成`: 已经落地并可验证
- `归档`: 不再是当前焦点，但保留历史记录

## 使用方式

1. 先看 `🎯 当前焦点`
2. 再看对应任务文档里的 `实施批次`、`验收标准` 和 `验证方式`
3. 提交前把任务状态同步回这里
