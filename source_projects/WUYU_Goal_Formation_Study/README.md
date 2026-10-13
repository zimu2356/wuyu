# WUYU / 物欲：目标形成局部实验

这是独立的研究原型，验证给定调节准则下的具体结果学习与目标选择。不是最终物欲，不包含语言模型、完整人格或主观体验的实现。

需要 Python 3.10 或更高版本，核心、实验和测试仅使用标准库。不需要 GPU、网络、模型权重、API 密钥或额外安装包。

```bash
cd WUYU_Goal_Formation_Study
python3 -m unittest discover -s tests -v
python3 run_study.py --output reproduced_results
python3 make_report.py --results reproduced_results
```

阅读顺序：`REPORT.md`（结果）→ `SPECIFICATION.md`（假设与范围）→ `protocol.json`（运行前方案）→ `wuyu_goal_study/core.py`（机制）→ `run_study.py`（实验）。

`results/episodes.csv` 是全部情境结果；`history_pairs.csv` 是配对历史结果；`summary.json` 是统计汇总；`training.jsonl.gz` 保存实际训练经历；`traces.jsonl.gz` 保存每步观察、选择、分数、结果和达成后的额外决策；`learned_states.jsonl.gz` 保存学习后状态；`run_manifest.json` 记录运行前源码哈希及输出哈希。gzip 文件可以用 Python 标准库 gzip 读取。

五个版本：learned（已学作用＋当前状态评价＋显式目标缓冲）、no_goal_buffer（去除显式目标缓冲，逐步重选）、cached_value（训练价值缓存）、erased_effects（保留规划能力、清除评价关联）、fixed_target（设计者指定具体目标）。

成本敏感性分析与主实验均事先写入方案。不要把此小世界中稳定或满分的结果外推为复杂环境能力。

所有源码均随包提供；`make_report.py` 仅生成静态中文 Markdown/HTML 报告，没有远程脚本或远程资源。
