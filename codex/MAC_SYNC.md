# 在 Mac mini 同步项目

GitHub 私有仓库为 `DavidZhang-HT/body-schema-research`，本项目位于 `codex/` 子目录。当前云会话与 Mac 上登录的 Codex 是独立执行环境，登录本身不提供远程终端通道。

在 Mac mini 的 Codex 中打开本地目录后，可执行以下任务：

> 将 GitHub 私有仓库 DavidZhang-HT/body-schema-research 克隆或快进同步到 ~/Projects/body-schema-research。项目在 codex/ 子目录。已有目录如有本地改动或属于其他仓库，请保留并报告，不要 reset 或覆盖。进入 codex 后运行 python3 scripts/verify_snapshot.py，报告本地项目路径及 git rev-parse HEAD 的结果。先不重跑训练。

也可以在 Mac 终端执行：

```bash
mkdir -p ~/Projects
git clone https://github.com/DavidZhang-HT/body-schema-research.git ~/Projects/body-schema-research
cd ~/Projects/body-schema-research/codex
python3 scripts/verify_snapshot.py
```

Mac 需具有该私有仓库的 GitHub 读取授权；这与登录 Codex 是不同的授权。克隆后，日常同步运行 `bash scripts/sync_on_mac.sh`，只允许快进更新，不覆盖本地改动。

可选仿真环境：已检查 PyPI 提供固定版本的 Python 3.12 原生 wheel。运行环境支持 Apple Silicon、macOS 14+；Intel Mac 仍可同步和查看项目，但该固定运行环境没有完整原生 wheel。

```bash
bash scripts/setup_macos.sh
bash scripts/run.sh schema_v2_checks.py
```

安装脚本只在项目目录建立虚拟环境，不安装 Homebrew 或修改系统 Python。`run.sh` 在 Mac 预先选择 CGL，让冻结脚本保持原样。项目已包含相对路径资产，无需重建 XML。运行 `build_model.py` 会生成本机路径并使完整快照校验失配；同步不需要这一步。
