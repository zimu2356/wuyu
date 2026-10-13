# WUYU_07

连续经历、有限操作技能、依赖组合、需要重新估值的研究代码。含一个显式规划的混合控制器，以及一个部署时完全由循环网络输出动作的学生。

学生借助混合控制器的模仿标签训练，正式运行不查询教师。其共享动作核心固定，168 个有限关联/技能参数在个体生活中继续更新。不是开放式生命、完整终身神经学习或已完成的新模型基底。

## 运行

```bash
python -m pip install -r requirements.txt
python demo.py --steps 48
python demo.py --steps 64 --save continued_07.npz
python demo.py --checkpoint continued_07.npz --steps 64 --save continued_07.npz
python -m unittest discover -s tests -v
```

默认从实际神经学生第 384 事件的完整检查点继续，包含 32 个不同世界/个体；屏幕展示第一个。超过原实验 1,536 事件后沿用最后阶段物理规律，是探索运行，不计入正式成绩。程序只在执行期间前进；HTML 是报告与已录制轨迹，不是后台进程。

## 完整复现

```bash
python run_train.py
python run_student.py
python run_refine.py
python run_eval.py
python run_probes.py
python analyze.py
python audit.py
python diagnose_student.py
python audit_diagnostic.py
python make_report.py
```

各训练脚本会复用已存在的结果文件；完全重训请在单独副本中移走 results 目录。轻量包仅含选中模型、一个真实检查点、统计与报告，完整审计/重放需完整包。完整包保留候选模型、训练日志、正式逐步数据、全部检查点和旁路轨迹。仅依赖 NumPy 与用于图表的 Matplotlib，不需要 GPU。

## 阅读顺序与边界

- `PROTOCOL.md`：最初协议与五个研发判据。
- `ENGINEERING_LOG.md`：工程试跑发现及限制。
- `STUDENT_ADDENDUM.md`、`REFINEMENT_PROTOCOL.md`：正式集封存期间加入的神经学生与自身轨迹训练。
- `EVALUATION_DETAILS.md`：参数冻结的能力测试和身体条件干预；受提示操作与自主生活严格区分。
- `SUPPLEMENT_DIAGNOSTIC.md`：正式结果之后针对神经需要/组合失败的探索性定位，未据此重新训练或选模。
- `results/summary.json`、`results/audit.json`：全部成绩和核验。

六个设备、两个准备阶段、三按钮是设计的有限世界；技能槽按设备分配，保留能力受益于参数分离，不能视作解决共享大网络遗忘。基础身体价值和混合控制器的递归规划来自设计者。物理成本是模拟成本，不是实际计算机电量。普通 RNN 与神经学生的参数、先验、训练方法和外部记忆不相同，二者比较只能说明本轮系统表现，不能归因于单一机制。

06 原有代码与数据未修改。07 是可复查实验分支，不替代 06，也不把之前实现视为最终完成品。
