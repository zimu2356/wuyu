# 关键性质验证

执行：`python -m unittest discover -s tests -v`

本轮训练前实际执行，6 项通过（运行约 0.171 秒）：

- `test_action_changes_future_and_hidden_state_not_observed`：动作改变位置与存量；观察接口只有存量与位置。
- `test_bounds_and_finite_outputs`：三种主方法的预测有界、有限；单步全知动作差距非负。
- `test_complete_checkpoint_resumes_exactly`：完整 NPZ 保存、恢复后的后续轨迹与不间断运行逐项相同。
- `test_intervention_changes_only_specified_mediators`：仅清空指定历史统计，保留预测、计数、年龄、近期反馈和模型参数；干预可影响步长。
- `test_reproducible_and_state_equivalent`：重复运行及等价递归状态写法产生一致轨迹。
- `test_single_candidate_agrees_with_batched_candidates`：候选独立运行与批量训练运行的结果一致。

这些检查用于验证实现与实验归因的关键前提，不验证生命、意识或人格。
