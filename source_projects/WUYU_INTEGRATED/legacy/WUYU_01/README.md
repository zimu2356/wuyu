# WUYU-01：持续交互中的历史调节实验

这是物欲研究的第一轮机制实验，不是数字人、语言模型或完整生命基底。包含预先固定的协议、独立编写的 NumPy 实现、训练参数、逐轨迹测试结果、检查点和中文报告。

## 从哪里读

1. `REPORT.md`：实际结果及可支持的结论。
2. `PROTOCOL.md`：运行前固定的问题、对照、指标和失败条件。
3. `wuyu/core.py`：环境、个体状态、学习规则与检查点。
4. `run_experiment.py`：固定预算训练、独立测试、因果干预与长期压力测试。
5. `results/`：原始 CSV、模型、训练历史、干预数据、完整检查点与运行哈希。
6. `SUPPLEMENT_01.md` / `supplement.py`：主结果之后声明的诊断，用新种子检查固定步长范围和环境的学习需求。

## 运行

Python 3.10+，依赖 NumPy 和 Matplotlib。原实验使用 CPU，没有使用 GPU、PyTorch 或外部模型接口。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py --mode adaptive --steps 256
```

完整重跑（会覆盖 `results/` 中同名输出，保留原始结果请先复制目录）：

```bash
python -u run_experiment.py
python supplement.py
python make_report.py
```

只使用已有参数重新评估：

```bash
python -u run_experiment.py --reuse-models
python make_report.py
```

`--reuse-models` 不重新训练；重新评估同一测试集不能算新的独立验证。若开展下一轮研究，应另外固定协议、数据种子与输出目录。

## 检查点恢复

```python
import numpy as np
from wuyu.core import load_checkpoint, rollout
world, agent = load_checkpoint('results/checkpoint_before_intervention.npz')
out = rollout(world.scenario, agent.mode, agent.parameter[0],
              initial=(world, agent), start=world.t)
print(out['metrics']['reward'].mean())
```

检查点包含预测权重、历史统计、访问计数、近期反馈、环境位置与存量，以及已生成的后续外生过程。NPZ 加载不启用 pickle。保存恢复仅验证计算连续性，不回答主观身份问题。

## 模型究竟有多大

- 6 个在线预测值，可解释为独热资源输入的线性预测权重。
- adaptive 的调节器是 9 参数逻辑单元；外层搜索后固定这些参数。
- `fast`、`slow`、`change` 是手工定义、在线更新的反馈历史统计。它们调节学习步长；本轮没有让更新算法自身无限递归改写。
- 每个方法都有 42 个基础状态标量（含整数计数），部分方法不使用其中的历史统计；memory 另存 6 × 窗口长度的反馈。
- 共同动作规则、探索机制、资源恢复以及奖励均为人工设计。

不要把不同缓存容量误称为严格参数匹配。本轮检验的是一个具体调节机制的收益，不是证明“权重记忆优于状态记忆”。`state_equivalent` 专门展示相同更新的状态写法可以获得相同行为。

## 研究诚信与许可

本代码独立编写，引用研究思路，不包含其他仓库的复制代码。`PROTOCOL.md` 列出参考文献。训练、验证和测试使用分开的轨迹种子；仅验证集用于选择模型，测试集不用于事后调参。

自有代码以 MIT 许可提供，见 `LICENSE`。文中不宣称任何生命、欲望或意识已经产生。
