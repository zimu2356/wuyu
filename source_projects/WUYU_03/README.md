# WUYU_03：持续感知学习原型

本轮已实际实现并在 CPU 上运行。核心由 1,033 个可训练参数的循环感知网络、250 个 Q 值的工具控制器、学到的后果模型、持续隐状态和经验缓存组成。它自行读取线索并回答，从答案正确反馈学习；原 WUYU_02 的程序代答仅保留在旧任务诊断中。

结论：已接通感知—行动—反馈学习；跨时刻记忆和规则反转后的再学习得到支持。跨情境旧技能保留未达到预设标准。长历史采样的独立种子补充实验有所改善，仍未达到保留判据。本项目不是语言模型、数字生命或意识实现。

## 阅读

- `REPORT.html`：可离线阅读的完整中文报告，含图表与结论边界。
- `REPORT.md`：同一分析的文本版本。
- `PROTOCOL.md`：主实验之前固定的协议。
- `SUPPLEMENT_01.md`：主实验后、补充实验前固定的假说。
- `VALIDATION.md`、`results/audit.json`：必要验证与源文件哈希复核。

## 环境

实际运行：Python 3.12.14 / NumPy 2.3.5 / Matplotlib 3.10.8 / CPU。
不依赖 PyTorch、GPU、API key 或外部模型。普通 Python 环境可按 `requirements.txt` 安装依赖。

```bash
python -B -m unittest discover -s tests -v
python -B demo.py --context A --steps 256
python -B demo.py --checkpoint results/supplement/checkpoint_reservoir.npz --context B --steps 256
```

`demo.py` 载入交付的真实训练状态，在一段新的随机经历中继续运行；默认不保存修改。所有批次个体继续运行，屏幕仅显示所选个体。要保存新状态，显式指定一个新路径：

```bash
python -B demo.py --context A --steps 512 --save continued_state.npz
```

该演示不是新的总体性能证据。运行程序停止时，模拟也停止；不存在自动部署的长期在线服务。

## 复现全部实验

输入中已包含旧实验诊断所需的真实轨迹和初态，不需要额外下载 WUYU_02。

```bash
python -B run_experiment.py --output reproduced_main
python -B run_supplement.py --output reproduced_supplement
python -B make_report.py
```

主/补充运行器拒绝覆盖已有输出。`make_report.py` 从交付的默认 `results/main` 与 `results/supplement` 重建报告；若要显示另一次复现，需在副本中明确调整其读取路径。`prepare_inputs.py` 仅用于从同目录上一级的 WUYU_02 原工程重建诊断输入，通常不必执行，且拒绝覆盖已有输入。

主运行器的源文件集合检查曾把导入时新生成的 Python 编译缓存也计入输入，从而使其 `source_unchanged` 聚合字段为 false；原始记录中每个协议、源文件与输入的 SHA-256 均复核匹配。详见 `VALIDATION.md`，未修改原始清单。使用 `-B` 可避免新写入该缓存。已有缓存作为审计附件保留，不是学习器的知识来源。

## 实验与局限

A：64 对旧历史个体，3 种算法 × 2 种支持条件 × 2 种证据条件。每分支支持期/撤回期各 1,024 步，部分分支另有 8 次受控探测。这里仍用旧版宏动作环境。

B：5 个方法各 32 个配对初始化/随机流个体，每个体连续 12,288 事件，课程为 A、B、A 返回、A 反转、A 恢复、B 返回。每个有效事件有 4 个神经微时刻。真实个体不因课程切换而清零；评估使用副本。受控感知评估固定基本通道，线索正确率 .60，不能与自然工具使用中约 .88 的准确率直接比较。

补充：新的 32 个种子，ring 与 reservoir 两种 256 条容量缓存，比较长历史样本保留与适应冲突。补充结果不能替换主 B4 的失败。

规则和情境接口、动作空间、外部奖励、答案类型、反向传播目标都由设计者规定。二元正确反馈可推出标签；这不是完全无监督学习。跨事件保留隐状态，但梯度只跨一个事件的 4 个时刻。网络冻结对照未经元学习，不代表所有固定权重循环网络。六阶段远不足以证明长期可塑性已解决。

## 文件结构

- `wuyu03/neural.py`：循环网络、手写梯度与短期缓存。
- `wuyu03/control.py`：Q、经验回放与 Dyna。
- `wuyu03/core.py`：无代答的环境接口、持续核心、完整状态保存恢复。
- `wuyu03/legacy.py`：旧任务证据与计算诊断。
- `wuyu03/reservoir.py`：独立补充的长期均匀采样缓存。
- `inputs/`：旧实验所需片段、来源记录和原环境源码。
- `results/main/`：参数选择、主实验原始逐事件数据、检查点与指标。
- `results/supplement/`：补充原始数据与检查点。
- `figures/`：可单独使用的科学图表。
- `CHECKSUMS.sha256`：包中文件完整性。

检查点保存网络、控制器、模型、隐状态、经验、上一反馈与内部随机状态。要逐项重现未来，需提供同一未来外生随机事件；演示入口会生成新的事件流。
