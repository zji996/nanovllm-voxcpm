# VoxCPM2 本地下载与 GPU0 服务启动

## 背景

当前仓库已经同时支持 `voxcpm` 和 `voxcpm2`，但默认文档和示例对本地服务入口、模型落点、端口选择还不够统一。目标是把本地运行路径收敛成一个面向后端仓库的稳定入口。

## 目标

- 将 `VoxCPM2` 下载到仓库内的 `models/VoxCPM2`
- 将 `models/` 保持为仓库内调试目录，并明确由 `.gitignore` 忽略
- 使用 `GPU 0` 启动服务
- 避开已被占用的 `8000` 端口，默认使用 `8010`
- 给出稳定命令入口，减少重复记忆成本

## 范围

- 统一本地环境安装命令
- 统一模型下载命令
- 统一 API 服务启动命令
- 补充当前后端项目的文档路由

## 设计收敛

- 不重构现有源码目录，不把仓库强行改成多应用结构
- 通过 `manage.sh` 提供薄入口，真正逻辑放在 `scripts/manage/`
- 使用 `docs/tasks/` 保存当前执行任务，使用 `docs/reference/` 写当前真实运行基线

## 实施步骤

1. 执行 `./manage.sh setup env`
2. 执行 `./manage.sh setup model`
3. 如需显式指定 ModelScope，可执行 `./manage.sh setup modelscope`
4. 如需切到 Hugging Face，可执行 `./manage.sh setup huggingface`
5. 执行 `./manage.sh dev api`
6. 访问 `http://127.0.0.1:8010/ready`
7. 访问 `http://127.0.0.1:8010/docs`

## 验收标准

- 模型文件落在 `models/VoxCPM2`
- `models/` 不进入 git 追踪
- 服务启动读取本地模型路径
- 服务仅使用 `GPU 0`
- 端口为 `8010`
- `/ready` 返回成功

## 验证方式

```bash
./manage.sh check docs
./manage.sh check quick
./manage.sh setup model
./manage.sh setup modelscope
./manage.sh setup huggingface
./manage.sh dev api
curl http://127.0.0.1:8010/ready
```

## 关闭条件

- 本地下载与启动路径可重复执行
- README 和 docs 已能把人引导到统一入口
