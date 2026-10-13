# WUYU_04

本轮是“保留多个规则并依据证据选择”的机制实验，不是完整数字生命，不是已训练好的通用模型基底。主体为 NumPy，CPU 即可运行。

## 运行

Python 3.12；安装 `requirements.txt` 中的 NumPy、matplotlib。

```bash
python3 -m unittest discover -s tests -v
python3 demo.py --checkpoint results/checkpoint_active_mid.npz --steps 128
```

恢复后继续执行会消耗仿真事件；状态不保存在进程外，除非显式指定新 checkpoint：

```bash
python3 demo.py --steps 256 --save continued_state.npz
```

请在复制出来的目录中复现实验，避免覆盖交付的原始数据：

```bash
python3 -u run_experiment.py
python3 make_report.py
```

默认 demo 从首次新规则 C 的学习中途恢复，包含世界随机状态；世界总长 6656 事件（学徒期 2048＋隐藏期 4608），到末尾自然结束。它不在下载后自动运行，也不是手机后台常驻服务。

## 文件

- `PROTOCOL.md`：事前写定的设计、种子、阈值、比较方式与限制。
- `wuyu04/legacy_neural.py`：逐字复制的 WUYU_03 RNN；出处哈希见 `inputs/provenance.json`。
- `wuyu04/diagnostic.py`：已知情境的原核心诊断、回放与模块对照。
- `wuyu04/latent.py`：独立的有限规则记忆核心、隐藏情境推断、探查与完整保存恢复。
- `results/selection.json`：验证集参数选择，不能据最终结果改变。
- `results/diagnostic_raw.npz`：每步正确性、无更新评估、反事实情境探针。
- `results/hidden_raw.npz`：普通线索、答案、预测概率、正确性、探查选择、先后信念、净收益等完整逐步记录。
- `results/checkpoint_*_mid.npz`、`*_final.npz`：完整 B 运行状态，可用 `demo.py` 读取；规则、权重、时钟、RNG 都在其中。
- `results/diagnostic_weights_*.npz`：仅 A 诊断的最终权重与隐藏状态，**不是**可精确继续的完整运行 checkpoint。
- `results/manifest.json`：运行环境、时间和实验开始时的源文件哈希；之后新增报告/demo 不改变被冻结核心。
- `REPORT.html`：适合手机阅读的自包含图文报告；`REPORT.md` 为文字副本。

## 数据形状与统计单位

A：`{method}__correct` 为 `[4096,32]`，`evaluations` 为 `[2,2,32]`（中途/最终、A/B、个体）。联合训练的“中途”包含两种规则，不能叫 A-only。

B：`{method}__correct/query/reward/...` 为 `[4608,32]`；信念为 `[4608,32,3]`。9 段各 512 步，A/B/A/C/A/B/C/A退化/A恢复。规则身份仅保存在世界 checkpoint 和统计脚本中；passive/active 的决策接口不接收它。

所有统计区间以 32 个体为单位配对重采样；没有把数千步当作独立样本。B 的几种方法共享普通事件和探查噪声流，但主动方法会获得额外实际观测，须连同探查成本解读。

## 设计者已经给出的帮助

离散输入、二元可反推标签的反馈、学徒期真实情境、固定三个模块、两个冻结旧模块、一个预留可塑槽、固定 hazard、手写探查策略。模块内部的答案是训练得到的，不是预填的；模块划分与冻结机制则明确由设计者提供。不能据此声称无限任务发现、人格、意识或生命。

报告中区分原 RNN 的失败诊断与独立新核心的能力，不能把后者成绩当成原 RNN 已修复，也不能把固定冻结带来的保留说成涌现。
