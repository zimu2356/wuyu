# 物欲统一工程

已确认的 WUYU_01—08 完整来源在 `legacy/`。统一入口在 `wuyu_core/`。这是研究工程整合，不把已有实验当成最终模型，也不把数字生命或意识当成已证实结果。

运行环境：Python 3.10+、NumPy（本次验证为 Python 3.12、NumPy 2.3.5）。CPU 即可；没有外部 API、GPU 或后台服务依赖。

```bash
python -m pip install -r requirements.txt
python -m wuyu_core list
python -m wuyu_core run --profile integrated --steps 128 --save runs/person_128.npz
python -m wuyu_core resume runs/person_128.npz --steps 128 --save runs/person_256.npz
```

默认 `prior` 沿用 08 的已训练图网络和在线经验记忆。`online` 和 `replay` 是新接通的可塑预测器实验，未证明提升生存或长链行为；`shadow` 学习但不影响动作，用于因果对照。

```bash
python -m wuyu_core run --mode replay --steps 256 --save runs/replay_256.npz --trace runs/trace.jsonl
python -m wuyu_core run --profile 03 --steps 32 --save runs/version03.npz
python -m wuyu_core resume runs/version03.npz --steps 32 --save runs/version03_next.npz
```

`--profile 01` 到 `08` 保留原世界及动作语义。除 01 新建世界外，默认读取各版已保存的真实个体；02/03 使用明确的新外生事件流，之后会随统一存档一起保存。01—03 的 `--horizon` 是新流长度；04—08 恢复原有限世界，不凭此参数扩展原实验。输出只展示少数个体的简短轨迹，但存档保存整个原批次。

迁移 08 的既有经历到新运行层：

```bash
python -m wuyu_core migrate08 legacy/WUYU_08/results/checkpoint_graph_varied_0_384.npz --individual 0 --mode replay --save runs/migrated_384.npz
python -m wuyu_core resume runs/migrated_384.npz --steps 64 --save runs/migrated_448.npz
```

迁移保留身体、世界、网络、循环状态、经验记忆；新增的预测器从未训练状态开始。旧版共用的批随机流无法等价拆成独立流，因此迁移的随机未来不保证与旧版一致；来源及这一变化写入存档。迁移后的保存恢复则可精确延续。01—07 的世界语义不相容，保留各自运行档案，不伪造“权重合并”。

验证与复现：

```bash
python -m unittest discover -s tests -v
python experiments/run_integration.py --workers 3
python experiments/diagnose_prediction.py
python experiments/diagnose_dependencies.py
python experiments/audit_sources.py
```

为了精确复现本机浮点路径，建议设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。保存恢复在相同 NumPy/BLAS 环境下验证，不保证跨硬件逐位一致。个体的世界时长用尽后停止；修改时长会改变相位时序，属于新实验。程序退出后不会继续后台生活。

目录：

| 位置 | 内容 |
|---|---|
| `wuyu_core/` | 统一存档、八版适配、新身体闭环、私有预测器、迁移工具 |
| `legacy/WUYU_01`—`08` | 保留原源代码、原协议与证据；不当作最终成品 |
| `docs/MECHANISMS.md` | 每版机制的明确去向和未接通处 |
| `PROTOCOL.md` | 新正式实验的固定条件与评价口径 |
| `experiments/` | 正式试验、失败诊断、来源审计与打包脚本 |
| `results/` | 新原始轨迹、配对指标、代表个体状态 |
| `validation/` | 旧版与新整合测试记录 |
| `provenance/` | 原包指纹、新文件来源、逐文件指纹 |
| `REPORT.html` / `REPORT.md` | 中文结果与下一步路线 |

运行包保留全部源码、说明、默认存档、选定模型和新实验数据；完整包另保留恢复到的历史大体积实验产物。某些历史重新训练脚本依赖这些历史数据，运行包不声称包含它们；使用完整包复现相关历史训练。
