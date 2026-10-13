# 物欲：完整体内的动作学习与持续状态实验

2026-10-12 · 1.1.0 · 基于已整合的 01–08 与后续研究继续发展

当前主干为 `whole`。新机制从运行起就属于同一个具身个体，共用身体、库存、旧经验技能、预测与行动反馈，并进入完整存档。A/B/C/D 是同一实现内的条件。先读本轮 `WUYU_Whole_Report_2026-10-12.md`、`docs/WHOLE_MECHANISMS.md` 和 `HANDOFF.md`。

默认使用 A：修正了进展语义的完整基线，沿用原生工坊世界。B/C/D 是显式开启的动作学习候选，是否应推广以本轮报告为准。原 01–08、integrated、workshop_bridge 与各研究入口仍保留，不把历史独立模块全部宣称为当前同时激活。

## 安装与继续同一个体

解压新的 Source 包即可运行全部源码和所需既有模型。把本轮所有 Data 分包解压到同一父目录，可恢复全部新实验原始轨迹。各包共享 `WUYU_Whole_2026-10-12/` 顶层目录、无重叠成员。

验证环境：Python 3.12.14，NumPy 2.3.5。图表工具另需 matplotlib。跨硬件/数值库不承诺浮点逐字节相同。

```bash
python3 -m pip install -r requirements.txt
python3 -B -m wuyu_unified list
python3 -B -m wuyu_unified run --seed 990001 --steps 384 --out runs/whole_384
python3 -B -m wuyu_unified resume runs/whole_384/state.wuyu --steps 384 --out runs/whole_768
```

默认等同 `--profile whole --mode A --case native`。暂停不推进时间，续跑不重置身体。输出目录必须不存在；已到声明 horizon 的存档不会偷偷延长寿命。

```bash
python3 -B -m wuyu_unified run --mode D --case challenge --seed 990101 --horizon 2560 --steps 768 --out runs/candidate_768
python3 -B -m wuyu_unified resume runs/candidate_768/state.wuyu --steps 1792 --out runs/candidate_final
python3 -B -m wuyu_unified fork runs/candidate_768/state.wuyu --intervention suppress_work_state --steps 64 --out runs/state_probe
```

`challenge` 的普通 CLI 默认为开发世界映射；正式 holdout 由下述冻结实验脚本明确指定。B 清空新增状态、C 使用 GRU、D 使用可保持/写入/清除状态；旧图网络和旧记忆都保留。诊断分支不会把结果写回父生命。

## 旧存档显式升级

```bash
python3 -B -m wuyu_unified migrate-whole examples/combined_384/state.wuyu --mode A --steps 16 --out runs/migrated_400
```

迁移验证原冻结来源，保留旧身体、库存、图网络状态、技能历史与动作随机流；重标可重建的证据样本，重初始化并用实际保留回放近似重建神经预测器。它不是无损的学习状态转换。已交付真实旧存档迁移示例 `examples/whole_migrated_400/`，迁移损失写入 lineage。

## 重现本轮正式评价

源码和协议在正式执行前冻结。以下命令新建另一次结果目录，使用同样的种子、配置和固定样本量；不会覆盖本次证据。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B tools/run_whole_experiment.py prepare --out runs/whole_reproduction --split holdout --seed-start 1400000 --n 64 --initializations 0 1 --cases challenge stable --rate .001 --diagnostic-n 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B tools/run_whole_experiment.py run --out runs/whole_reproduction --workers 6
python3 -B tools/analyze_whole.py runs/whole_reproduction
python3 -B tools/audit_whole_results.py runs/whole_reproduction
```

总计 1,024 条主生命、2,621,440 次转移，另有固定诊断分支。执行时间取决于机器。每个条件共享物理预算；没有教师动作或隐藏路线反馈。正式样本以世界聚类，不以时间步充当样本。当前预算不足以精确排除 0.01 级的小效应，见开发决定和冻结协议。

## 验证与目录

```bash
python3 -B -m unittest discover -s tests -v
python3 -B tools/run_original_tests.py
python3 -B -m wuyu_unified verify-sources
```

| 路径 | 用途 |
|---|---|
| `wuyu_unified/whole.py` | 完整体观察、决策、实际反馈与状态整合 |
| `wuyu_unified/action_learning.py` | 相同参数预算下的三种动作学习条件 |
| `wuyu_unified/whole_world.py` | 连续物理事件、可行性检查与成本账本 |
| `wuyu_unified/whole_migration.py` | 旧语义存档的明确迁移 |
| `experiments/whole_holdout/` | 冻结协议、原始记录、分析、检查点与审计 |
| `experiments/whole_development*/` | 两轮开发数据、失败结果与对应源码快照 |
| `experiments/semantic_correction/` | 语义修正与冻结原版的独立工程对照 |
| `source_projects/` | 原 13 个项目及所需既有模型，原文件不修改 |
| `reference_runtime_v1/` | 旧整合层的来源验证副本 |
| `docs/WHOLE_MECHANISMS.md` | 机制、信号、学习逻辑、迁移与解释边界 |

Source 包包含全部原 Python/JavaScript 源码、原生运行所需检查点、旧整合验收材料和本轮摘要。历史大体积模型/原始研究数据仍由上一轮的 7 个历史 Data 包提供；它们的顶层目录是 `WUYU_Unified_2026-10-12/`，不要与本轮 Data 分包混淆。`verify-sources --full` 需要补全这些历史数据；普通 `verify-sources` 核验全部源码与协议。

旧报告和 `docs/README_V1.md`、`docs/HANDOFF_V1.md` 是历史证据，不代表当前默认入口或当前效果结论。
