# WUYU_05

沿 WUYU_03/04 推进的持续学习实验：同一循环神经核心、工具动作、256 条经验、8 事件重算回放和两个可学习更新强度。旧工程未修改。

```bash
python3 -m unittest discover -s tests -v
python3 run_experiment.py
python3 audit_results.py
python3 make_report.py
python3 demo.py
```

依赖见 requirements.txt。先看 REPORT.html（手机可读）与 PROTOCOL.md。完整训练包括三个策略家族各 48 个候选、独立验证选择和 32 个正式测试个体；每个体有三个配对分支，统计不将分支当独立样本。

results 保存原始数组、搜索记录、选定调节器、完整检查点、独立核验结果和冻结源文件哈希。

不预填新规则槽位，不永久冻结旧知识（frozen 仅为对照）。仍保留 A/B 情境输入、二元完整反馈、显式学习目标与固定调节接口。本项目是有限机制实验，不是开放式数字生命或通用模型。
