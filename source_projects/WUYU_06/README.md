# WUYU_06

共享神经核心 + 连续身体状态 + 有界经历目标 + 个体后果学习。

主报告：`REPORT.html`（离线可读，图表已内嵌）。主要结论：连续运行闭环已实现；五项预定效益判据中只有“预测输入对已训练策略有作用”通过。普通循环代理与完整核心都达到存活指标上限，目标与回放的额外必要性尚未证实。

## 运行

Python 3.12。依赖见 requirements.txt；仅运行 demo 需要 NumPy，生成报告另外需要 Matplotlib。

```bash
cd WUYU_06
python -m pip install -r requirements.txt
python demo.py --steps 48
python demo.py --steps 64 --save continued.npz
python demo.py --checkpoint continued.npz --steps 64
python -m unittest discover -s tests -v
```

默认从真实正式生命的第 128 事件恢复一个 32 个体批次，打印第 0 个体。`--individual 3` 可查看另一个个体。每次从同一检查点开始会得到相同后续；使用 `--save` 才生成继续状态。仿真有有限预设时长，示例不会跨过其终点。HTML 报告不会在手机后台运行模型。

## 两种包

- 轻量运行包：代码、报告、六个选中模型、完整训练选择日志、一个真实检查点、摘要与核验结果；可直接运行 demo。
- 完整实验包：还包括全部训练候选、主/压力测试逐步数组、每种条件终态、三个版本的中途状态、WUYU_05 诊断及初始不完整文件的恢复证据。

轻量包不含完整原始数据，所以不能直接重新生成全部图表或完成原始数据审计。可依次运行以下脚本生成缺少的数据：

```bash
python run_train.py
python run_eval.py
python run_probes.py
python diagnose_wuyu05.py
python run_stress.py
python audit_results.py
python make_report.py
```

`run_train.py` 会复用已经存在的 training_*.json；这不是从中断状态继续训练。重新训练请在另一解压副本中先移走 results。BLAS 线程建议设为 1，以降低运行差异。原 WUYU_05 的额外对照项只在原工程文件可用时核验；诊断所需的原核心源码与已选参数已经包含在 inputs。

## 文件

- `PROTOCOL.md`：实现前固定方案。
- `TRAINING_CONFIG.json`：训练前固定数值配置。
- `SUPPLEMENT_01.md`：看到主指标上限后固定的探索性压力方案。
- `wuyu06/world.py`：资源、位置、通道与失能的仿真定律。
- `wuyu06/network.py`：显式 NumPy 循环 actor-critic 与后果预测梯度。
- `wuyu06/agent.py`：目标候选、在线快速参数、有限记忆和动作执行接口。
- `wuyu06/state.py`：数组 + JSON 的完整状态保存，读取不使用 pickle。
- `results/frozen_before_training.json`：原始协议/核心哈希。
- `results/audit.json`：独立账本和真实检查点核验。
- `results/recovery.json`：四个初始不完整 NPZ 的重建记录；原始指标完全匹配。

`raw_*.npz` 中第一维是时间，第二维是世界。`alive` 表示该事件开始时仍能行动，`body` 是事件后的资源；`pred`/`target` 的前两项为资源变化除以 .36，第三项为是否移动。`goal_origin` 和 `replay_origin` 是早于当前事件的真实来源。`phase` 是评估器字段，从不传入代理。

## 解释边界

全局模型有 3,386 参数；个体在线更新的是 693 个快速预测参数。正式阶段没有继续训练全局动作权重。目标表示、八槽分类、最长 24 事件、资源价值和记忆选择规则仍有人为先验。这是有限连续个体原型，不是已证明的数字生命、人或通用模型基底。

训练的六次运行、3 × 32 正式完整核心轨迹和全部对照均实际执行；按世界配对的统计区间对这三个训练版本条件化。完整核验保留负结果和数据修复记录。
