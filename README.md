# 物欲 WUYU：完整体源码与整合验收

本仓库包含 01–08、后续研究、当前 Whole 1.2 主干、运行所需模型与本次验收记录。可直接在仓库根目录运行；无需下载或合并源码分包。

已通过 170 项回归测试、153 种公开模式与情境配置及 6 组补充运行，共 316,354 个主事件；10 条旧存档迁移路径也已验证。工程验收通过，长期稳定生存仍未解决。

详见 [完整验收报告](WUYU_Complete_Verification_2026-10-13.md) 和 [分包空目录复原核验](WUYU_Complete_Package_Verification_2026-10-13.json)。报告中的 ZIP 合并章节用于离线分发，GitHub 工作副本直接从本目录运行。

## 快速运行

```bash
python3 -m pip install -r requirements.txt
python3 -B -m wuyu_unified verify-sources
python3 -B -m wuyu_unified run --profile whole --mode R10 --case r_combined --seed 1770100 --horizon 2560 --steps 768 --out runs/current_768
python3 -B -m wuyu_unified resume runs/current_768/state.wuyu --steps 1792 --out runs/current_final
```

`--out` 必须是尚不存在的目录。当前运行需要 Python 3.12+ 和 NumPy；整合验收工具的 JavaScript 语法检查另需 Node.js。大规模旧实验原始轨迹不是当前运行与验收的依赖，源码、所需模型与验收记录均包含在仓库中。

---

# 物欲：完整体内的可达关系学习

2026-10-12 · 主干 1.2.0 · 继承 01–08、整合层和 Whole 1.1

2026-10-13 已完成当前完整体的全量整合验收：170 项回归测试、全部 153 种公开配置和 6 组补充运行、十种旧存档迁移路径均通过工程检查。源码分包合并与运行说明、机制覆盖及生存边界见 `WUYU_Complete_Verification_2026-10-13.md`。工程通过不等于长期生存已经解决。

当前主干仍是 `WholeAgent`。可达关系学习和旧策略权重调节直接进入同一个体的观察、旧图网络、动作适配器、真实反馈与完整存档，从第一次开发实验起就在主体中共同运行。先读 `WUYU_Relation_Report_2026-10-12.md`、`docs/RELATION_MECHANISMS.md` 和 `HANDOFF.md`；实验效果以报告为准。

默认入口保持 `whole/A/native`。新实验明确选择 R 条件，不在更新时静默改变已有模式。旧 01–08、integrated、workshop_bridge 和各研究入口仍保留；全部历史模块不等于在当前每一步都同时激活。

## 条件

| 条件 | 读取实际移动学得的关系 | 旧策略权重 | 动作适配器 |
|---|---|---|---|
| A | 否 | 原组合策略 | 无新增适配器 |
| R00 | 否，后台学习作参照 | 1 | 原 C 的 GRU |
| R10 | 是 | 1 | 同一 GRU |
| R01 | 否，后台学习作参照 | 从 0.9 开始学习 | 同一 GRU |
| R11 | 是 | 从 0.9 开始学习 | 同一 GRU |
| RT | 是 | 固定 0.9 | 同一 GRU |

五种 R 条件均分配 22,515 个参数，关闭的门参数不实际参与学习；分配数相同不代表有效容量相同。旧 B/C/D 仍可运行。新关系来自可见的起点、动作和下一位置，不读取世界封路标记或未来事件。六个位置身份、原环形几何、证据衰减和距离代价转换是明确给定的设计先验。

## 运行与接续

Source 包含完整源码及运行所需既有模型，可单独运行。所有本轮 Data 分包解压到同一父目录，恢复全部新实验原始记录；共享 `WUYU_Whole_2026-10-12/` 顶层目录且成员无重叠。分布清单给出路径、大小和 SHA256。

验证环境：Python 3.12.14、NumPy 2.3.5；图表另需 matplotlib。跨硬件或数值库不承诺浮点逐字节相同。

`requirements-reproduction.txt` 记录本次恢复与验证实际使用的数值库和绘图库版本；完整环境记录在 `experiments/relation_validation/recovery_runtime_2026-10-13.json`。希望尽量贴近该环境时，可使用前者安装依赖。

```bash
python3 -m pip install -r requirements.txt
python3 -B -m wuyu_unified list
python3 -B -m wuyu_unified run --profile whole --mode R10 --case r_combined --seed 990201 --horizon 2560 --steps 768 --out runs/relation_768
python3 -B -m wuyu_unified resume runs/relation_768/state.wuyu --steps 1792 --out runs/relation_final
```

暂停不推进时间，续跑不重置身体、知识、优化器、未完成学习缓存或随机流。输出目录必须不存在，声明的生命终点不会偷偷延长。CLI 使用开发世界映射，正式世界由冻结实验脚本指定。

新环境为 `r_stable`、`r_block`、`r_noise`、`r_reopen`、`r_relocate`、`r_combined`：稳定、封路、偶发失败、重开、障碍换位和组合挑战。长生命保留同一个体至 5,120 步，死亡吸收、不复活；2,560 步后延续原改变后的世界，不额外加入新挑战。

## 旧个体显式升级

支持来源已验证的 Whole 1.1 A/C 存档升级到 R 条件：

```bash
python3 -B tools/migrate_relation_checkpoint.py experiments/relation_migration/old_C_t1097.wuyu --mode R10 --out runs/migrated_R10.wuyu
python3 -B -m wuyu_unified resume runs/migrated_R10.wuyu --steps 256 --out runs/migrated_continuation
```

保留身体、库存、旧技能、旧神经状态和随机流；C 的动作权重、Adam 和未完成缓存继续保留。新增关系只用实际保留的近期移动记录重建，不假装拥有全部历史。A 升级时新建 C 动作适配器。跨模式的未来轨迹不保证相同。

可直接加载的当前来源锁示例是 `experiments/relation_migration/final_R00_t1097.wuyu` 和 `final_R10_t1097.wuyu`，完整验证见 `final_verification.json`。更早的开发迁移文件保留作历史证据，不能冒充当前冻结版本的存档。

## 重现实验

首要比较 R10−R00 在两轮开发后、任何正式成绩之前确定。固定样本，不因成绩扩样。以下命令使用相同世界与算法重新运行；输出目录必须是新目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B tools/run_relation_experiment.py prepare --out runs/relation_main --split holdout --seed-start 1600000 --n 128 --initializations 0 1 --modes A R00 R10 R01 R11 RT --cases r_stable r_combined --horizon 2560 --diagnostic-n 12 --primary-contrast R10-R00 --min-effect .05
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B tools/run_relation_experiment.py run --out runs/relation_main --workers 7
python3 -B tools/analyze_relations.py runs/relation_main
python3 -B tools/audit_relations.py runs/relation_main --workers 7
```

辅助情境改为 `--seed-start 1601000 --n 16 --cases r_block r_noise r_reopen r_relocate --diagnostic-n 8`；长期改为 `--seed-start 1602000 --n 16 --modes R00 R10 --cases r_combined --horizon 5120 --diagnostic-n 0`。各自指定新的 `--out`。其余主选项相同。

正式共 3,904 条生命、10,158,080 次主生命转移；两轮开发另有 480 条。全部协议、固定模型初始化、配置和来源锁保存在 `experiments/relation_holdout_*/protocol.json`。

上述 `prepare` 命令会记下新的目录名和创建时间，因此新协议摘要以及包含运行来源的最终状态哈希不会与本次文件相同；比较物理轨迹和结果时，应区分这些来源元数据与数值行为。2026-10-13 的证据恢复则保留原协议原文，才能核对有留存摘要的原轨迹与最终状态。恢复经过和范围见 `docs/RELATION_RECOVERY_2026-10-13.md`，不将恢复计算当成额外独立样本。

`analysis.json` 为完整分析。脚本终端兼容输出中的 `primary` 字段沿用旧标签 R11−R00；本研究始终以协议的 `primary_contrast=R10-R00` 和完整 `contrasts` 中该项为首要比较。报告和判定脚本读取正确的冻结比较。

## 验证与目录

```bash
python3 -B -m unittest discover -s tests -v
python3 -B -m wuyu_unified verify-sources
```

| 路径 | 内容 |
|---|---|
| `wuyu_unified/whole.py` | 完整体行动与真实反馈循环 |
| `wuyu_unified/relations.py` | 可见移动转移的学习、查询和存档 |
| `wuyu_unified/action_learning.py` | 原动作学习与可关闭的旧策略权重输出 |
| `wuyu_unified/relation_world.py` | 六类连续物理情境 |
| `docs/RELATION_MECHANISMS.md` | 信号流、机制、先验及解释边界 |
| `docs/RELATION_DEVELOPMENT_DECISION.md` | 开发结果和固定样本量依据 |
| `experiments/relation_holdout_*` | 三个正式协议、摘要、轨迹、存档、诊断和审计 |
| `experiments/relation_development*` | 全部开发结果和源码快照 |
| `experiments/relation_start_snapshot` | Whole 1.1 起点，用于明确验证旧存档 |
| `experiments/relation_validation` | 当前测试、继承源码哈希与最终证据汇总 |
| `source_projects` | 全部原项目源码及运行所需既有模型 |

`docs/README_Whole_1_1.md` 和旧报告是历史证据。上一轮 Whole 1.1 原始轨迹继续由旧 Whole Data 包提供；更早大数据仍由原 7 个历史 Data 包提供，顶层目录为 `WUYU_Unified_2026-10-12/`。本轮不重复旧轨迹；所有原源码、所需模型和旧摘要保留。`verify-sources --full` 需要历史大数据，普通 `verify-sources` 核验源码与协议。

上一轮 Whole 六个原始数据分包的既有入口保存在 `docs/HISTORICAL_WHOLE_1_1_DOWNLOADS.md`。若需要直接执行冻结的 Whole 1.1 起点，先运行 `python3 tools/restore_snapshot_links.py` 恢复快照对共享原源码的相对链接；当前主干和随包的迁移示例不依赖此步骤。
