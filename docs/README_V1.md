# 物欲：源码、机制、运行逻辑与效果整合

2026-10-12 · 版本 1.0.0 · WUYU 01–08 + 既有整合版 + 后续四轮实验

本工程将完整来源、统一运行状态和实际机制组合一起交付。先读 `WUYU_Integration_Report_2026-10-12.html` 与 `docs/MECHANISM_INTEGRATION.md`。组合候选不自动替代已有参照；实际验收结果决定其地位。

## 安装与启动

解压 Source 包即可使用全部源码、默认历史检查点、选定模型、新验收数据和回归样本。把全部 Data 包解压到同一父目录，才得到全量历史模型与原始记录。所有包都有同一个顶层目录 `WUYU_Unified_2026-10-12/`，无文件覆盖冲突。

在该目录运行。验证环境为 Python 3.12.14、NumPy 2.3.5。可选图表需要 matplotlib 3.10.8。浮点逐字节一致性只在验证环境内保证。

```bash
python3 -m pip install -r requirements.txt
python3 -B -m wuyu_unified list
python3 -B -m wuyu_unified run --profile workshop_bridge --mode combined --seed 990001 --horizon 1536 --steps 384 --out runs/combined_384
python3 -B -m wuyu_unified resume runs/combined_384/state.wuyu --steps 384 --out runs/combined_768
```

第二次调用承接第一份身体、世界、记忆和随机流，不重置个体。每次输出目录必须新建。有限 horizon 是预先声明的实验终点，`--steps` 可以提前暂停；暂停不会执行 WAIT 或自动推进世界。

默认 `run` 是 `integrated/prior`；`workshop_bridge` 默认 `shadow`。`combined` 是明确指定的候选组合。它把图网络、关系/技能记忆、神经在线学习与前缀回放、证据模型、E/U 接在同一条真实生命中，不提供已经形成“人”或新价值的结论。

直接从解压工程运行，或使用 `pip install -e .`。`source_projects/` 是运行依赖，不能只拷贝 `wuyu_unified/`；不支持丢弃来源树的单独 wheel 分发。

## 原实验与诊断入口

```bash
python3 -B -m wuyu_unified run --profile 08 --steps 64 --out runs/stage08
python3 -B -m wuyu_unified run --profile revision --mode BEU --case local_swap --seed 71000 --steps 1100 --out runs/evidence_1100
python3 -B -m wuyu_unified fork runs/evidence_1100/state.wuyu --intervention erase_model --action 1 --steps 64 --out runs/evidence_diagnostic
python3 -B -m wuyu_unified run --profile exploration --mode least_visited --horizon 108 --steps 108 --seed 31000 --out runs/explored
python3 -B -m wuyu_unified evaluate-history runs/explored/state.wuyu --case need_shift --variant learned --steps 12 --out runs/evaluation
```

`evaluate-history` 是从真实采集史派生的隔离测试，明确重置测试身体与位置；它不是连续生命的续接。`fork` 是研究者干预的新分支，父状态不变。01–08 保留原批次和宏/微动作语义，不能把每个统一 event 都当成同等行动预算。

`import-native` 支持完整旧整合存档、旧 01–08 envelope、持续经历原生完整存档和经验证的证据修订存档；不完整的分析快照拒绝导入。`migrate08` 保留旧个体的声明内容，但原批随机流切换为新私有流，不能承诺迁移前后同一随机未来；迁移完成后的保存恢复是精确的。

## 核验与原实验重跑

```bash
python3 -B -m unittest discover -s tests -v
python3 -B tools/run_original_tests.py
python3 -B tools/verify_frozen_revision.py
python3 -B -m wuyu_unified verify-sources
python3 -B -m wuyu_unified data-status
python3 -B -m wuyu_unified verify-sources --full
```

最后一项需要全部 Data 包。131 项历史测试与 22 项新集成测试已执行通过；证据修订回归覆盖 seed 71000 的 5×6 条完整生命，72,000 步逐字段和最终状态一致，不是重新统计全部 128 种子的独立重复研究。

```bash
python3 -B -m wuyu_unified reproduce --profile revision
python3 -B -m wuyu_unified reproduce --profile goals --execute --out runs/original_goals_reproduction
python3 -B tools/validate_integrated_effects.py --out runs/new_acceptance --workers 4
```

`reproduce` 不加 `--execute` 只列出原实验命令；执行时先建立独立副本。重评估使用随包选定模型，不等同重新训练全部历史模型。完整重评估可能需要 Data 包中的额外模型。验收脚本提前冻结参数、源码与场景，不根据结果挑样本；本轮结果在 `validation/integrated_effects/`。

## 目录与状态约定

| 路径 | 内容 |
|---|---|
| `wuyu_unified/` | 生命周期、无 pickle 完整存档、原实验适配、迁移、可复用证据模型和实际组合 |
| `source_projects/` | 16 个原始来源包中的 12,373 个文件，字节保持；含 291 个 Python 源文件 |
| `provenance/` | 原始包 SHA256、每个成员来源/大小/哈希、1–8 源码覆盖 |
| `docs/` | 机制映射、运行顺序、边界、理论和历史方案 |
| `tests/`、`validation/` | 新/旧测试、精确回归、完整验收原始轨迹和结果 |
| `tools/` | 来源重建、验证、效果验收、打包及报告生成 |
| `HANDOFF.md` | 当前任务优先级、实际结果、已知缺口、后续接续说明 |

新检查点存储结构化 JSON/gzip，数组保留 dtype/shape，元组键、队列、集合和 RNG 均有类型编码；内部摘要和源码/配置锁阻止损坏与静默迁移。SHA 是完整性校验，不是认证签名。以后修改代码需显式迁移检查点，不能删掉校验以假装兼容。

本项目未启用后台常驻服务。执行、暂停、分支均由明确调用发生。
