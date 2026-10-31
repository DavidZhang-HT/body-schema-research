# Pi Plus 身体图式与动力学学习实验

基于 Pi Plus URDF 和 MuJoCo 的可复现实验：关节零位辨识、注意力动力学模型、地面接触图式、统一身体部位目标，以及单肢到四肢的课程学习。

本项目位于 `DavidZhang-HT/body-schema-research` 仓库的 `codex/` 子目录。下列项目命令均在 `codex/` 中运行。

**当前没有实现可靠的纯神经策略翻身、坐起或持续行走。** 代码、成功与失败轨迹、模型、数据分割和负结果均保留。

## 最新结果

第二版课程采集 4,740 段物理运动。用力排序相关为 0.726，动作幅度基线为 0.568；跨肢体组合预测尚未优于简单相加。16 个小幅抬举目标中，直接策略成功 0/16、普通想象规划 1/16、省力规划 1/16；尚无同等成功条件下的省力收益证据。

- [第二版肢体课程实验](SCHEMA_CURRICULUM_STUDY.md)
- [预测与控制图](schema_v2_results/study_summary.png)
- [实际／预测组合关系图](schema_v2_results/combination_relations.png)
- [真实物理录像](schema_v2_results/control_examples.mp4)
- [翻身／坐起补测](ROLL_SIT_RESULTS.md)
- [统一身体部位目标实验](GOAL_SCHEMA_STUDY.md)
- [地面接触与起身实验](GROUND_GETUP.md)
- [早期注意力动力学实验](ATTENTION_DYNAMICS.md)
- [主动关节零位辨识](ZERO_IDENTIFICATION.md)
- [早期步行实验及来源说明](WALKING_EXPERIMENT.md)

## 同步与运行

Mac 的同步、完整性校验和可选安装步骤见 [MAC_SYNC.md](MAC_SYNC.md)。代码、数据与录像均可直接克隆；固定 Mac 仿真依赖面向 Apple Silicon、macOS 14+、Python 3.12。

```bash
python3 scripts/verify_snapshot.py
bash scripts/setup_macos.sh
bash scripts/run.sh scripts/smoke_test.py
```

Linux 使用原有 CPU 环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-attention.txt
bash scripts/run.sh scripts/smoke_test.py
```

仿真和模型训练使用 CPU 即可。大规模重跑会覆盖对应实验结果，应在新实验目录或分支进行。模型和报告可直接检查，无需重新训练。

## 内容与来源

`schema_v2_*.py` 是最新数据采集、模型、训练、策略、评估和审计代码；`schema_v2_results/` 包含模型、数据、全部 48 条闭环轨迹及重放结果。其余 `*_results/` 保留历史实验数据与结果。

`upstream/` 只保留复现所需的 Pi Plus 机器人、参考动作和原始许可。虚拟环境、缓存、重复 ZIP 与控制台日志没有纳入版本控制。单个文件均小于 GitHub 普通文件上限，本快照无需 Git LFS。

冻结的训练代码及检查点保持不变。`assets/` 中的模型仅将云端绝对路径改成相对路径；详见 [第三方来源](THIRD_PARTY_NOTICES.md) 和 [发布说明](PUBLICATION_NOTES.json)。
