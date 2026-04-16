# Repo Alignment

## 已落地

- 增加了根入口 `manage.sh`
- 增加了 `scripts/manage/` 统一环境、模型下载、启动与检查命令
- 增加了 `docs/roadmap`、`docs/tasks`、`docs/reference` 三分法文档路由
- 增加了面向后端场景的默认值: `models/VoxCPM2`、`GPU 0`、`8010`

## 仍可继续优化

- 为 `nanovllm_voxcpm/` 补一份模块级 `README.md`，说明核心包职责和关键子目录
- 给 `deployment/README.md` 增加 `VoxCPM2` 优先的本地运行路径
- 将 `./manage.sh check quick` 扩展为更完整的测试矩阵
- 后续如果需要多实例部署，再补面向服务运维的 `docs/reference/deployment.md`

## 为什么是这个版本

这轮脚手架收敛的核心价值主要在统一入口、文档路由、验证内建和边界清晰。这个仓库是后端项目，不需要为了套模板而重做目录树，所以本轮只落最能提升协作效率的部分。
