# WUYU：有限自主探索研究

本包于 2026-10-12 完成。研究问题是：撤掉任意位置的平衡教学后，从实际起点取得的有限经历能否支持适当选择？本包不是新的完整数字生命架构。

先阅读 `REPORT.md` 或离线 HTML 报告。恢复研究时先阅读 `HANDOFF.md`。

## 核心结果

正式留出 128 个种子。在预定的 432 次交互主比较中，随机和定向探索均有 128/128 通过四情境，未发现定向优势。108 次的次要比较为随机 86/128、定向 128/128。随机的 42 个失败个体中，24 个未发现一种结果，另有 18 个发现了两种结果但缺少已观察的备用路线。

同历史的 learned 与 no_goal_buffer 在全部 4,608 对完整行为轨迹上相同。结果支持这个任务中作用学习和状态重估的功能，不支持新价值标准形成或目标缓冲的额外必要性。

## 可复现运行

Python 3.12.14 为本次验证环境。核心、运行、分析和审计仅使用标准库，不需要网络、模型权重或第三方服务。后续 Python 版本的位级浮点／随机兼容性未额外验证。

从解压后的本目录运行，输出目录必须尚不存在：

```bash
python -m unittest discover -s tests -v
python run_study.py --cohort holdout --output outputs/local_reproduction
python analyze.py outputs/local_reproduction
python audit_data.py outputs/local_reproduction
python compare_runs.py outputs/holdout outputs/local_reproduction --output validation/local_reproducibility.json
```

`CODE_FREEZE.json` 和 `PROTOCOL_FREEZE.json` 验证科学源码／协议。不要重新运行 `freeze_code.py` 覆盖既有冻结；该脚本会拒绝已有文件。创建新研究应复制到独立分支、声明修改和新协议。

图表与离线报告已随包提供，无需安装依赖即可阅读。重新生成呈现层需要 matplotlib 3.10.8：

```bash
python reporting/build_report.py
```

报告生成器不会运行新实验，但会重新写入图表、报告和描述性失败诊断。交付文件的原始哈希随后会相应改变；应保留原分发包以核对。

## 记录格式

| 文件 | 一条记录／行代表什么 |
|---|---|
| acquisition.jsonl.gz | 一个采集动作：seed、collector、stream_budget、step、Experience |
| knowledge.csv | 一份种子／采集器／预算的经历摘要，共 1,152 行正式记录 |
| initial_learners.jsonl.gz | 一份经历和一种决策版本的测试前快照，共 5,760 条 |
| episodes.csv | 一个独立测试分支，共 23,040 行正式记录 |
| test_traces.jsonl.gz | 一次测试的完整逐步决策、实际反馈和结果后的额外选择 |
| summary.json | 预先指定的种子级统计；没有按测试次数虚增独立样本量 |
| collector_timing.csv | 一个完整采集流的进程 CPU 时间；自主组只记录 1,728 次的总采集时间 |
| run_manifest.json | 参数、种子范围、源码与原始输出哈希、起止时间、记录数量 |
| data_audit.json | 从原始经历重建模型与分支、全分母核验的结果 |

自主组短预算复用长轨迹前缀；教学组各预算分别生成，因此采集文件中的 stream_budget 不能一概当作独立的学习预算。所有版本的 history_sha256 必须与对应 knowledge.csv 一致。

在正式记录中，732,672 条采集经历来自 640 个实际采集流，而不是把各预算前缀重计为新交互。71,614 次测试行动与采集交互分开计数；完成后的额外决策不调用世界，不增加环境行动数。

## 本包的范围

保留了本轮全部源码、测试、协议、开发与正式原始数据、报告、冻结清单、复现比较，以及原目标形成研究 ZIP。旧核心字节未改。

`references/` 包含本轮依据的理论稿、此前研究建议和历史源码索引。此前 01—08、统一工程、LAAP 的大型源码归档没有在本包重复复制；其位置与完整性证据见源码索引。本包无需那些历史归档即可复现本轮。

## 必须保留的解释边界

位置连续来自实际行动；每次采集前身体重置仍然存在。采集与使用经历是两个阶段。精确位置、结果类别、固定理想点、平方偏差、图搜索和探索偏向均由设计者提供。测试只涉及同一拓扑的不同标识、动作映射和噪声。

完整重复运行的七项科学输出逐字节一致，CPU 时间和执行时间戳不要求一致。该结果是工程可复现性检查，不增加独立统计样本。
