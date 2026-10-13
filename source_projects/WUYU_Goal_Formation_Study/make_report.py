#!/usr/bin/env python3
"""Build matching, standalone Chinese Markdown and HTML reports (stdlib only)."""
import argparse
import gzip
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
LABELS = {"learned": "学得作用＋当前评价", "no_goal_buffer": "去除显式目标缓冲",
          "cached_value": "缓存训练时价值", "erased_effects": "清除作用关联",
          "fixed_target": "预设追求 A"}
SCENARIOS = {"quiet": "常态选择", "blocked": "短路阻断后换路", "need_shift": "内部状态改变后换目标", "satiated": "无调节需要时停止"}


def pct(x):
    return f"{100*x:.1f}%"


def inline(s):
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    protocol = manifest["protocol"]
    groups = {(g["variant"], g["scenario"]): g for g in summary["groups"] if g["cost"] == protocol["action_cost"]}
    md, web = [], []

    def heading(s, level=2):
        md.append("#"*level+" "+s+"\n")
        web.append(f"<h{level}>{inline(s)}</h{level}>")

    def paragraph(s, emphasis=False):
        md.append(s+"\n")
        web.append(f'<p class="{"callout" if emphasis else ""}">{inline(s)}</p>')

    def table(headers, rows):
        md.append("| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+
                  "\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)+"\n")
        web.append('<div class="table-wrap"><table><thead><tr>'+"".join("<th>"+inline(x)+"</th>" for x in headers)+
                   '</tr></thead><tbody>'+"".join('<tr>'+"".join('<td>'+inline(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+
                   '</tbody></table></div>')

    heading("物欲：具体目标形成的第一轮局部实验", 1)
    paragraph("2026-10-11 · Python 标准库研究原型 · 理论底稿版本 2 的局部落实")
    paragraph("**已完成实现与实际运行。结果支持给定调节准则下的工具性目标选择；没有证明新的评价标准、主观欲望或人格形成。**", True)

    heading("这轮最重要的发现")
    paragraph("学习结果对内部状态的作用，再根据当前状态重新评价，足以在这个简单世界中选择合适结果、换路、换目标和停止。清除作用关联后，选择正确结果的比例接近随机；冻结训练价值则无法处理内部状态变化。")
    paragraph("**去掉显式目标缓冲后，所有主要结果保持不变。** 这轮没有获得显式目标保持机制必要性的证据。持续行动可以由保留的作用知识和每一步重新评价解释，不能单凭它称为承诺或持续欲望。", True)
    paragraph("预设追求 A 的版本也能完成常态选择和换路。这直接说明：在几个表面行为上表现得像有目标，并不足以证明目标自主形成。")

    heading("我们实际给了系统什么")
    table(["明确给予", "通过经历获得", "尚未解释"], [
        ["两维内部偏差、零点与平方偏差准则", "结果标识与内部变化的关联", "评价准则为什么值得维护"],
        ["精确位置标识、离散动作和已分割结果", "观察到的位置—动作转移", "对象、概念和意义怎样形成"],
        ["图搜索、比较规则、显式目标缓冲", "具体结果的选择随知识和当前状态变化", "规划、反思与标准修订怎样发展"],
        ["平衡呈现全部状态—动作对", "训练样本内的结果作用估计", "自主探索与连续成长"]])
    paragraph("目标目录开始为空，只有实际经历过的结果才能成为候选；但世界只有两个预设结果。因此本轮学到的是具体结果的工具性价值，尚未创造新的结果类别。调节变量是模拟变量，没有将宿主电量、受苦或死亡作为实验条件。")
    paragraph("评价采用 V(g,b)=||b||²−||b+μ(g)||²，再减去已知路径的步成本。μ(g) 来自观察，评价公式、理想点和规划算法由设计者提供。内部状态改变导致价值变化，不等于评价标准改变。")

    heading("实验规模与运行方式")
    paragraph(f"主实验：{protocol['n_seeds']} 个种子 × 5 个版本 × 4 个情境，共 2,560 条轨迹；加上步成本 0 和 0.03 的两组预定敏感性分析，总计 {summary['episode_count']:,} 条轨迹。另有 {summary['history_pair_count']} 对首次目标选择的历史比较。主成本为 0.01。")
    paragraph("每个种子随机化标识、动作映射、训练顺序和作用噪声，世界拓扑相同。它们不是 128 个丰富世界，也不是 7,680 个独立生命。每条测试轨迹从相同训练状态重新开始。训练是平衡呈现 432 条经历，不是自主探索；测试最多 12 步。没有语言目标提示，也没有独立的提示撤除实验。")
    paragraph("方案在首次批量运行前本地冻结并记录哈希；没有公开预注册。未根据首批结果调整参数。正确性检查使用与主实验不同的种子；完整复现结果另存于验证记录。")

    heading("主实验结果")
    rows = []
    for variant in protocol["variants"]:
        rows.append([LABELS[variant]]+[f"{groups[variant,s]['success_count']}/{groups[variant,s]['n']}（{pct(groups[variant,s]['success_rate'])}）" for s in protocol["scenarios"]])
    table(["版本"]+list(SCENARIOS.values()), rows)
    paragraph("各列成功定义：常态/阻断到达 A；内部状态改变后到达 B；无需要时首次决策停止。上述 A/B 只用于实验者读表，学习版本收到随机化标识。预设 A 版本明确获得实验者指定目标。")
    paragraph("满分并不表示强泛化：状态完全可观测，训练覆盖全部转移，作用差异大，规划能力现成，替代路线也已学过。它验证的是这些规则在受控小世界中的功能后果。")
    table(["代表性比例", "Wilson 95% 区间"], [
        [f"学得作用版本常态：{pct(groups['learned','quiet']['success_rate'])}",
         " — ".join(pct(v) for v in groups['learned','quiet']['wilson_ci95'])],
        [f"清除作用关联常态：{pct(groups['erased_effects','quiet']['success_rate'])}",
         " — ".join(pct(v) for v in groups['erased_effects','quiet']['wilson_ci95'])],
        [f"清除作用关联状态改变：{pct(groups['erased_effects','need_shift']['success_rate'])}",
         " — ".join(pct(v) for v in groups['erased_effects','need_shift']['wilson_ci95'])]])
    contrast = next(c for c in summary["contrasts"] if c["scenario"] == "quiet" and c["comparison"] == "learned - erased_effects")
    paragraph(f"常态中，学得作用版本相对清除关联版本提高 {100*contrast['mean_difference']:.1f} 个百分点；按种子配对 bootstrap 95% 差值区间为 {100*contrast['ci95'][0]:.1f}—{100*contrast['ci95'][1]:.1f} 个百分点。全部配对比较见 summary.json。对于完全相同或完全分离的结果，bootstrap 区间可能退化，不能解读为对其他环境毫无不确定性。")

    heading("换路、停止和历史的具体证据")
    paragraph("学得作用版本常态平均用 3 步取得结果；短路线被阻断后平均 5 步：包含 1 次未预知的失败和 4 步已知替代路线。智能体只有观察到自环失败后才更新转移图，没有提前得到阻断消息。一次失败即可覆盖旧转移，是本轮给予的确定性环境先验。")
    paragraph("在常态实际取得一次结果后，另外调用一次决策：学得作用和无目标缓冲版本均在 128/128 个种子停止，其余三个版本均未停止。此指标来自真实额外决策，未把 runner 的终止当作停止行为。")
    table(["版本", "交换训练历史后首次目标改变", "两种历史都选择相应结果"], [
        [LABELS[h["variant"]], pct(h["changed_rate"]), pct(h["both_follow_history_rate"])] for h in summary["history"]])
    paragraph("历史对照保持当前位置和身体输入相同，只交换先前经历中 A/B 的内部作用。第一次选择前没有新的结果反馈。学得作用、无目标缓冲和缓存价值都随历史变化，预设目标和清除关联不变。这说明已学知识具有因果作用，但不支持人格、主观经历归属或身份形成。")
    sensitivity_changes = [g for g in summary["groups"] if g["cost"] != protocol["action_cost"]
                           and g["success_rate"] != groups[g["variant"],g["scenario"]]["success_rate"]]
    paragraph(f"两组预定步成本敏感性分析中，成功率发生变化的条件数为 {len(sensitivity_changes)}。这只是很窄的成本范围检查，没有检验新拓扑、长期噪声转移、未知对象或复杂社会互动。")

    heading("这对物欲理论意味着什么")
    table(["主张", "本轮判定"], [
        ["具体目标的选择可以受经历塑造", "在给定调节准则和两个已分割结果的范围内得到实现支持"],
        ["当前需要变化可改变同一结果的工具性价值", "得到支持；属于固定准则下的状态重估"],
        ["绕路能证明自主目标", "不成立：预设目标对照也能绕路"],
        ["持续行为需要显式目标保持变量", "本轮没有支持：删除目标缓冲后表现相同"],
        ["产生了自己的评价标准、主观欲望或人格", "没有检验，也没有证据"],
        ["已经建立神经模型基底", "没有：本轮是小型表格模型与图搜索"]])
    paragraph("这些结果是透明机制的合成验证，未与人类或动物数据拟合，也没有提出已证实的新认知理论。旧代码结果不计入本轮证据。")

    heading("下一步应怎样改变研究")
    paragraph("首先保留这一版作为可审查的基准，不因行为满分就扩展模块。下一轮优先去掉“全部转移和结果都由实验者平衡教完”的支撑，研究有限经历与自主探索下，作用关联能否可靠形成。这样可以检验当前成功是否依赖理想训练条件。")
    paragraph("显式目标缓冲暂不作为已经证明必要的核心。以后若研究跨时项目，应加入目标竞争、中断和恢复，并比较显式保持、逐步重估、习惯等解释；相同表面行为仍需内部过程对照。评价标准如何改变和新目标类别如何形成继续保留为独立理论缺口。")

    heading("复现、文件与核验")
    paragraph("核心与实验只需 Python 3.10+ 标准库，无 GPU、联网、额外包或模型权重。完整源码、11 项正确性测试、运行前方案、训练经历、每步轨迹、学习后状态和统计结果都在压缩包内。运行已结束，不是在后台持续生长的个体。")
    table(["文件", "用途"], [
        ["README.md / SPECIFICATION.md / protocol.json", "运行方法、全部假设和冻结方案"],
        ["wuyu_goal_study/core.py / run_study.py", "完整机制实现和实验入口"],
        ["results/episodes.csv / history_pairs.csv", "逐条结果与配对历史选择"],
        ["results/training.jsonl.gz / traces.jsonl.gz", "实际训练数据与每一步决策证据"],
        ["results/learned_states.jsonl.gz / summary.json", "学习后状态与统计汇总"],
        ["results/run_manifest.json / VALIDATION.json", "运行记录、哈希与复现核验"],
        ["tests/test_mechanisms.py / tests_output.txt", "正确性检查源码与实际输出"]])
    paragraph("运行前 protocol.json 的 SHA-256："+manifest["source_sha256_before_run"]["protocol.json"])

    heading("研究参照及边界")
    sources = [
        ("Keramati & Gutkin (2014), Homeostatic reinforcement learning", "https://doi.org/10.7554/eLife.04811", "提供将结果价值联系到内部状态的参照；本轮不是论文复现。"),
        ("Colas et al. (2022), Autotelic Agents", "https://doi.org/10.1613/jair.1.13554", "用于区分目标表征、生成、选择和达成；本轮只实现其中有限部分。"),
        ("Barandiaran, Di Paolo & Rohde (2009), Defining Agency", "https://doi.org/10.1177/1059712309343819", "个体性与规范来源仍需解释；本轮不检验完整能动性理论。")]
    for title, url, note in sources:
        md.append(f"- [{title}]({url})：{note}\n")
        web.append(f'<p><a href="{html.escape(url)}">{html.escape(title)}</a>：{html.escape(note)}</p>')
    (ROOT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    style = """body{margin:0;background:#f4f6f7;color:#172b34;font:17px/1.8 system-ui,-apple-system,'Noto Sans CJK SC',sans-serif}main{max-width:960px;margin:auto;padding:38px 22px 70px;background:white}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:40px;border-bottom:2px solid #d6e5e7;padding-bottom:8px}p{overflow-wrap:anywhere}strong{color:#075563}.callout{background:#e9f4f2;padding:18px;border-left:4px solid #198577}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.65}th,td{border:1px solid #d4dfe3;padding:12px;text-align:left;min-width:105px}th{background:#e9f0f3}tr:nth-child(even){background:#f8fafb}a{color:#076784}@media(max-width:600px){body{font-size:16px}main{padding:22px 16px}h1{font-size:25px}h2{font-size:21px}th,td{padding:9px}}"""
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物欲：目标形成局部实验报告</title><style>'+style+'</style></head><body><main>'+"\n".join(web)+'</main></body></html>'
    (ROOT / "REPORT.html").write_text(document, encoding="utf-8")
    print("Created REPORT.md and REPORT.html")


if __name__ == "__main__":
    main()
