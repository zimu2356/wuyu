# WUYU：证据复验与局部修订

这是独立于上一轮冻结源码的新研究。五方法为 B0（原算法）、BE（动作年龄复验）、BU（持续矛盾后的局部降权）、BEU（组合）、BR（固定概率随机复验）。六条件为稳定、局部正负工具对换、备用通道悄然变好、噪声脉冲、一次可见外生身体/疲劳冲击、A→B→A 回归。

## 运行环境

Python 3.12、NumPy；绘图额外依赖 matplotlib。脚本主动将 OpenBLAS/OMP 线程设为 1，`run.py` 默认 4 个独立进程。并行单位为独立种子，不改变种子内随机流。底层 BLAS/平台变化可能改变极小浮点误差，本环境中单独进程复现检验覆盖全部正式输出。

```bash
python3 -m unittest -v test_mechanisms
python3 run.py --config formal_protocol.json --out my_reproduction --start 71000 --count 128 --workers 4
python3 analysis.py --config formal_protocol.json --data my_reproduction
python3 audit.py --config formal_protocol.json --data my_reproduction
```

本轮正式 N=128，与 `formal_protocol.json` 中 `holdout_seeds.count` 一致。输出目录必须不存在；不能覆写原实验。中断时已有 `block_*.json` 为已完成种子的回执，本版不自动合并半成品运行。

## 文件与数据

- `core.py`：可见状态、世界物理、条件岭回归、逐动作证据、E/U 与随机对照。
- `experiment.py`：持续主轨迹、诊断探针和真实更新的阻断/恢复副本。
- `protocol.py`、三个冻结 JSON、开发记录：何时决定什么。
- `diagnostics.py`：同检查点等量取样、相同转移流更新、同数据预测比较、单步真值预测替换，以及可达性检查。
- `run.py`、`storage.py`：确定性压缩记录、种子完成回执、哈希与完整性。
- `analysis.py`：冻结的唯一主比较、稳定代价约束及次要分析。
- `audit.py`：独立表达的逐步物理、全部文件哈希、最终加权正规方程、真实事件来源、干预恢复校验。
- `test_mechanisms.py`：13 项机制测试，包括与上一轮原代码 600 步动作/预测兼容。
- `development/`、`holdout/`：所有个体、方法、场景、全部原始步骤。不存在成功样本筛选。
- `reproduction/`：独立重跑的清单、分析和审计。为免重复占用，打包时可省略与原件哈希完全相同的重复轨迹/模型文件，省略规则和校验结果单列。

每条生命保存 `{seed}_{case}_{method}.npy.gz`，是 gzip 包裹的无 pickle NumPy structured array，字段见 `storage.DTYPE`。所有状态与预测均为 float64；每行对应决策前绝对 `tick=行号`，`after` 为该动作后身体两维和疲劳。材料可由种子与时钟重建。动作、选择来源、修订、模型版本、更新前预测、残差、样本质量与年龄均保留。`n`、`mass`、`ess`、`version` 是更新前指标；`streak` 为该步更新后连续矛盾数。最近状态覆盖距离记录在定期探针中，没有声称其为概率置信度。

配套 `.json.gz` 含最终完整世界/代理、探针、主结果及适用时的干预全部状态和事件。每个种子的 `block_*.json` 保存完整文件清单与 SHA256；`MANIFEST.json` 汇总。`RUN_START.json` 记录实际配置和科学源文件哈希；`COMPLETE.json` 只有完成且源文件不变时生成。

```python
from storage import read_events, read_gz_json
events = read_events('holdout/71000_local_swap_BEU.npy.gz')
metadata = read_gz_json('holdout/71000_local_swap_BEU.json.gz')
```

## 继续一个个体

```bash
python3 tools/continue_life.py --config formal_protocol.json --state holdout/71000_local_swap_BEU.json.gz --steps 100 --out continued.json
python3 tools/continue_life.py --config formal_protocol.json --state continued.json --steps 100 --out continued_again.json
```

工具校验源文件及配置后，恢复身体、疲劳、时间、所有模型样本及加权矩、探测随机流和选择随机流。主轨迹从不重置；只有明确标出的诊断副本分叉或使用匹配状态。

## 解释边界

普通 Agent 只接收时间、身体、疲劳、材料和可用动作数量，不接收世界类别、改变时点、工具轴/方向或评估标记。真值单步参照只在评估器中出现，并非公平学习方法或全局最优解。其参照是无噪声且截断后的后果，在边界附近不等于随机截断后结果的数学期望。

固定零理想点、评分、六特征、动作通道、复验偏向和检测规则均为设计者给予。此轮研究学习系统如何修订证据依据，不声称生成新欲望、技能、价值标准、社会关系或主观体验。
