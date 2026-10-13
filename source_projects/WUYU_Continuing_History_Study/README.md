# WUYU：持续状态与经历作用实验

2026-10-12。独立新实现，旧阶段源码不改动。本轮完成持续状态、WAIT、状态条件预测、保存恢复、经验删除／无关删除／恢复、环境变化后的在线学习对照。尚无技能生成、新目标类别、价值标准形成或社会关系。

优先打开 `WUYU_Continuing_History_Report_2026-10-12.html`；文字版为同名 `.md`。精确给定条件见 `GIVEN_AND_LEARNED.md`；分析口径见 `ANALYSIS_DEFINITIONS.md`。

## 运行条件

科学实验只需 Python 标准库；本次为 Python 3.12.14。图表生成另需 matplotlib，本次为 3.10.8。跨 Python/平台的逐字节一致性未验证。

在本目录执行：

```bash
python -m unittest discover -s tests -v
python run.py --cohort holdout --output outputs/my_reproduction
python analyze.py outputs/my_reproduction
python audit.py outputs/my_reproduction
python compare_runs.py outputs/holdout outputs/my_reproduction
```

输出目录必须不存在，程序拒绝覆盖已有实验。`FORMAL_FREEZE.json` 会验证冻结源码及协议；不要为了运行修改其中哈希。测试输出默认到终端，不覆盖冻结的 `validation/unit_tests.txt`。

## 一个可以暂停、继续的个体

```bash
python tools/continue_life.py --seed 51000 --ticks 600 --output my_individual_1
python tools/continue_life.py --resume my_individual_1/state.json --ticks 600 --output my_individual_2
```

第二条从第 600 步继续，身体、模型、访问次数和随机状态均不重置。每次输出完整新增事件及状态。该便捷接口在正式冻结后增加，不参与正式实验；跨进程的保存恢复另有集成核验。默认世界在第 1600 步反转工具作用，代理不接收变化标签。运行时长由调用者指定，不代表代理的死亡或主观经历。

## 内容与责任

| 路径 | 内容 |
|---|---|
| core.py | 世界、给定评分、有限经验窗模型、代理与记忆干预 |
| experiment.py | 连续主轨迹、诊断副本、预测探针、冻结模型分支 |
| run.py | 固定预算运行、原始日志、源码和文件校验 |
| analyze.py / audit.py | 按种子统计、逐条审计与独立正规方程核验 |
| calibrate_and_freeze.py / compare_runs.py | 开发规则选定协议、复现实验文件比较 |
| development_protocol.json / DEVELOPMENT_FREEZE.json | 开发运行前固定的任务参数与选择规则 |
| formal_protocol.json / FORMAL_FREEZE.json | 正式运行前选定的第 400 步、128 个种子及源码冻结 |
| outputs/development | 32 个开发个体的完整原始数据和分析 |
| outputs/holdout | 128 个留出个体的完整原始数据、分析、审计、复现凭证 |
| outputs/reproduction | 包内只留运行凭证和派生结果；逐字节相同的大型原始副本不重复打包 |
| reporting | 绘图、报告和探索性定位脚本；不影响冻结实验 |
| validation | 13 项机制测试、运行日志、审计及数据完整性事件记录 |
| references | 承接方案、理论底稿及旧源码地图；旧档案未强行合并到本模型 |

## 数据定义

- `events.jsonl.gz`：每次可见状态、五个动作的预测与分数组成、选择、后果；`event_id` 可关联模型记忆。
- `checkpoints.jsonl.gz`：完整世界及代理状态。只有评估器可读工具真值；代理只接收观测。
- `interventions.jsonl.gz`：删了什么、删前／删后／恢复哈希及诊断起始代理。
- `predictions.jsonl.gz`：条件与同经历均值预测、无噪声参照；不给主轨迹反馈。
- `final_states.jsonl.gz`：每个个体在线主轨迹与冻结分支的最终状态。
- 三份原始 CSV：诊断分支、预测探针和生命周期窗口；`individual_effects.csv` 保留每个种子的效应。

主轨迹始终连续。只有诊断副本将身体、疲劳、材料对齐；分支不计作独立个体。bootstrap 使用种子为抽样单位。次要区间未做多重比较校正。

## 完整性与复现说明

首次正式运行完成后，一个轨迹压缩文件被发现尾部缺失。独立整轮重跑产生的文件哈希与**首次完成清单中原有的哈希**相同，随后用其恢复原路径；没有改写预期哈希或结果。原始失败审计与解决过程见 `validation/RAW_LOG_INCIDENT.json` 和报告。损坏副本作为本地诊断证据保留，包内提供通过完整审计的恢复文件。

`checksums.json` 校验交付包内各文件。完整实验只需本包；旧项目不作为执行依赖。原始数据体积较大，重复运行的相同字节仅存一份，保留两次运行清单、日志与 11 项逐字节复现比较。
