"""Semantics are explicit: profiles share lifecycle, not a fictitious task scale."""
DESCRIPTIONS = {
    "01": "价值记忆与学习率调节", "02": "工具维护与 Q 学习宏动作",
    "03": "私有循环感知与控制", "04": "隐情境推断与付费观测",
    "05": "连续学习、个体随机流与回放", "06": "身体、经验目标与行动后果",
    "07": "技能、依赖组合与神经学生", "08": "关系图神经动作与需要变化",
    "integrated": "既有 1–8 具身整合主循环",
    "goals": "给定调节准则下的学得结果选择",
    "exploration": "有限自主经历采集；保留每步身体重置",
    "history": "连续身体状态与经历作用",
    "revision": "证据复验与局部修订；五方法六场景",
    "workshop_bridge": "旧具身核心与通用证据模型的工程接通",
    "whole": "连续完整体：修正进展语义、动作可塑性与持续状态候选",
}
CONTINUOUS = {"integrated", "history", "revision", "workshop_bridge", "whole", "06", "07", "08"}
MODES = {
    "integrated": ("prior", "online", "replay", "shadow"),
    "goals": ("learned", "no_goal_buffer", "cached_value", "erased_effects", "fixed_target"),
    "exploration": ("random", "least_visited"),
    "history": ("online", "frozen"),
    "revision": ("B0", "BE", "BU", "BEU", "BR"),
    "workshop_bridge": ("shadow", "online", "evidence", "update", "revision", "combined"),
    "whole": ("A", "B", "C", "D", "R00", "R10", "R01", "R11", "RT"),
}
DEFAULT_MODES = {k: v[0] for k,v in MODES.items()}
DEFAULT_MODES["exploration"] = "least_visited"
DEFAULT_MODES["revision"] = "B0"
CASES = {"revision": ("stable", "local_swap", "opportunity", "noise_pulse", "visible_shock", "recurrence"),
         "goals": ("quiet", "blocked", "need_shift", "satiated"),
         "whole": ("native", "challenge", "stable", "r_stable", "r_block", "r_noise", "r_reopen", "r_relocate", "r_combined")}

def describe(profile):
    if profile not in DESCRIPTIONS: raise ValueError(f"Unknown profile: {profile}")
    return dict(profile=profile, description=DESCRIPTIONS[profile],
                continuous_body=profile in CONTINUOUS,
                modes=MODES.get(profile, ("native_checkpoint",)),
                default_mode=DEFAULT_MODES.get(profile,"native_checkpoint"),
                cases=CASES.get(profile, ()),
                step_unit="macro action" if profile=="02" else "native event" if profile.isdigit() else "primitive transition or explicit stop decision",
                claim="engineering integration" if profile in ("workshop_bridge","whole") else "original study semantics")
