"""Generate the Chinese report and static figures from saved, unmodified results."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
NAMES = {"memory":"窗口记忆", "fixed":"固定可塑性", "adaptive":"历史调节可塑性",
         "constant_gate":"匹配平均步长", "fixed_002":"固定步长 0.02", "fixed_001":"固定步长 0.01",
         "frozen":"冻结预测（不学习）", "greedy_quality":"已知当步质量的贪心策略"}
KINDS = {"in_distribution":"分布内", "rapid":"更快变化", "high_noise":"高噪声",
         "drift":"连续漂移", "stationary":"长期静态", "mixed":"混合条件"}


def read(name):
    return json.loads((OUT/name).read_text())


def ci(value):
    return f"{value['mean']:+.6f} [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}]"


def table(headers, rows):
    md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n"
    md += "\n".join("| " + " | ".join(map(str,row)) + " |" for row in rows)
    ht = "<div class='table-wrap'><table><thead><tr>"+"".join(f"<th>{html.escape(h)}</th>" for h in headers)+"</tr></thead><tbody>"
    ht += "".join("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return md, ht+"</tbody></table></div>"


def main():
    ev, model, intervention = read("evaluation.json"), read("models.json"), read("interventions.json")
    long, sup, manifest = read("long_run.json"), read("supplement.json"), read("run_manifest.json")
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "font.size":10, "savefig.dpi":170})
    kinds = list(KINDS)
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    yy = np.arange(len(kinds))
    means = np.array([ev[k]["adaptive_minus_fixed"]["mean"] for k in kinds])
    lower = np.array([ev[k]["adaptive_minus_fixed"]["ci95"][0] for k in kinds])
    upper = np.array([ev[k]["adaptive_minus_fixed"]["ci95"][1] for k in kinds])
    ax.errorbar(means, yy, xerr=[means-lower,upper-means], fmt="o", color="#267373", capsize=4)
    ax.axvline(0,color="#777",linewidth=1)
    ax.set_yticks(yy, ["In distribution","Rapid changes","High noise","Continuous drift","Stationary","Mixed"])
    ax.invert_yaxis()
    ax.set_xlabel("Adaptive minus fixed (0.02): mean reward per event")
    ax.set_title("Main evaluation: paired differences and 95% intervals", loc="left", pad=16)
    ax.grid(axis="x",alpha=.2)
    fig.savefig(OUT/"main_differences.png"); plt.close(fig)

    labels = ["adaptive", "constant_gate", "fixed_002", "fixed_001", "frozen", "greedy_quality"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    values = [sup["overall"]["means"][k] for k in labels]
    colors = ["#267373", "#789ca0", "#a1aeb6", "#a1aeb6", "#b16b43", "#818490"]
    ax.barh(np.arange(6), values, color=colors)
    ax.set_yticks(np.arange(6), ["Adaptive", "Matched constant", "Fixed 0.02", "Fixed 0.01", "Frozen (no learning)", "Quality-aware greedy"])
    ax.invert_yaxis(); ax.set_xlim(0, .225)
    for i,v in enumerate(values): ax.text(v+.002, i, f"{v:.5f}", va="center",fontsize=9)
    ax.set_xlabel("Mean reward per event (new independent seeds)")
    ax.set_title("Post-main diagnostic: learning did not improve reward", loc="left",pad=16)
    fig.savefig(OUT/"supplement_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    for mode,color in (("adaptive","#267373"),("fixed","#8b8e95")):
        vals=long[mode]["segment_rewards"]
        ax.plot(np.arange(1,len(vals)+1)*256,vals,label=mode,color=color,marker=".")
    ax.set_xlabel("Events per life"); ax.set_ylabel("Mean reward in 256-event segment")
    ax.set_title("4096-event stress test: finite states, no lifetime guarantee",loc="left",pad=16)
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.savefig(OUT/"long_run.png"); plt.close(fig)

    md = ["# WUYU 第一轮实验报告\n\n2026-10-10 · 机制实验 · 已完成实际训练与独立评估"]
    ht = ["<div class='eyebrow'>WUYU / EXPERIMENT 01 / 2026-10-10</div><h1>历史能否改变<br>之后的学习方式？</h1><p class='subtitle'>一轮完成的机制实验，以及没有得到支持的核心假说。</p>"]

    def section(title):
        md.append("## "+title); ht.append("<h2>"+html.escape(title)+"</h2>")

    def paragraph(text, box=False):
        md.append(text)
        ht.append("<p"+(" class='callout'" if box else "")+">"+html.escape(text)+"</p>")

    def addtable(headers, rows):
        a,b=table(headers,rows); md.append(a); ht.append(b)

    def figure(name,caption):
        md.append(f"![{caption}](results/{name})")
        data=base64.b64encode((OUT/name).read_bytes()).decode()
        ht.append(f"<figure><img src='data:image/png;base64,{data}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>")

    section("先看结论")
    paragraph("这一轮没有证明历史调节带来额外适应收益。它完成了持续交互、在线更新、训练、独立测试和因果干预的可运行闭环，但结果不支持现在扩大模型。补充检查表明，在当前环境与行动规则下，完全不更新的对照表现更好。",True)
    paragraph("这不是数字生命、欲望、意识或人格已经产生的证据，也不是这些目标不可能实现的证据。应保留实验工具，重新校准环境与对照，再研究更复杂的机制。")

    section("实际做了什么")
    paragraph("6 个资源点组成环。个体移动并采集；收益受隐藏质量、当前存量、噪声和移动成本影响。采集会耗损资源，资源逐步恢复，位置与全部学习状态持续保留。每步是一次移动及采集事件。系统只能在采集后观察该点带噪质量，不能读取隐藏质量或变化时间。")
    paragraph("本实现是小型预测与控制系统：6 个在线预测权重，9 参数的学习步长调节器，以及反馈历史统计。动作评分和探索规则由我们指定。调节器通过外层搜索训练，评估时参数固定；预测权重与历史统计继续更新。它不是从零自然演化的生命体，也不是完整仿脑网络。")
    addtable(["方法","在线变化","离线选择"],[
        ["窗口记忆","写入反馈，窗口平均形成预测","窗口长度；选中 32"],
        ["固定可塑性","误差修正预测权重","固定步长；选中 0.02"],
        ["历史调节可塑性","权重、历史统计及由其决定的步长","9 个调节器参数；3 次搜索中选验证最优"],
        ["匹配平均步长","相同误差修正，步长不随事件变化",f"使用主模型训练平均步长 {model['mean_training_gate']:.6f}"],
        ["等价递归状态","与历史调节模型相同方程，换成状态解释","不训练新模型；检查轨迹等价"]])
    paragraph("可塑预测更新为 q ← clip(q + α × (观察反馈 − q), 0, 1)。adaptive 的 α = 0.01 + 0.94 × sigmoid(θ·特征)。历史特征包括快慢误差统计、反馈变化、访问间隔等；这些特征的形式是人工选择，不应称为内生价值。")
    paragraph("各方法共享观测、动作评分、探索规则和事件预算，但有效容量不完全相同。memory 存 6×32 个反馈值；adaptive 有额外的调节参数与统计记忆。这不是严格容量匹配的架构竞赛。")

    section("训练与评估范围")
    paragraph(f"训练 24 条轨迹，每条 384 事件；验证 24 条新轨迹，每条 512 事件。调节器进行 3 次独立搜索，每次 60 代、48 候选。根据验证集选择种子 {model['selected']['seed']}，主测试前固定参数。主评估为 6 类×128 条＝768 条独立轨迹，每条 1024 事件；各方法使用相同外生过程。另有每类 32 条、4096 事件的压力测试。")
    paragraph(f"原主实验在 CPU 上实际运行约 {manifest['elapsed_seconds']:.1f} 秒。高吞吐来自小规模状态和 NumPy 批量计算；这不代表大模型训练只需同样时间。没有 GPU、PyTorch 或外部语言模型参与训练。训练搜索共评估约 7963 万个模拟候选事件，它们不是 7963 万个不同个体，也不是那么多独立样本。")
    paragraph("置信区间为按环境类别分层、轨迹配对的 2000 次 bootstrap；类别等权。它们描述给定模型在这些环境样本上的不确定性，不涵盖所有架构、训练随机性或现实世界情况。分类指标及次指标用于解释，不做事后挑选。")

    section("主评估结果")
    overall=ev["overall"]
    addtable(["方法","平均净收益/事件","所选点预测 MSE","平均步长"],[
        [NAMES[k],f"{v['reward']:.6f}",f"{v['prediction_mse']:.6f}",f"{v['alpha']:.6f}"]
        for k,v in overall["means"].items()])
    paragraph("窗口记忆行的步长列记录 1/有效窗口长度，仅供描述平均操作，不是突触更新步长；窗口移除旧反馈时，更新也不等同于简单误差修正。预测 MSE 针对各方法实际访问的点，因此访问分布不同，不能当成相同预测样本上的直接能力排名。")
    addtable(["比较：adaptive 减去对照","平均差及 95% 区间"],[
        [NAMES[k],ci(overall["adaptive_minus_"+k])] for k in ("fixed","memory","constant_gate")])
    paragraph("主模型比原定 fixed 网格中选出的模型略好，名义上满足 H1 的统计判定；但它没有优于匹配平均步长的简单对照，H3 未获支持。原网格最低为 0.02，而主模型平均约为 0.011，比较包含搜索范围差异。因此不能把主差值归因于历史调节。")
    figure("main_differences.png","主评估：对原定固定步长的分类差值。范围差异限制了对优势来源的解释。")
    addtable(["训练搜索种子","验证收益","测试收益"],[
        [r["seed"],f"{r['validation']:.6f}",f"{next(x['test_reward'] for x in ev['training_seed_sensitivity'] if x['training_seed']==r['seed']):.6f}"]
        for r in model["runs"]])
    paragraph("三次搜索的测试表现有差异；其中一次更弱。主模型按验证集选定，没有因为另一搜索在测试中略好而替换。这些结果不足以宣称该学习方法普遍优越。")

    section("干预结果：有影响，未证明有益")
    paragraph("在第 384 事件复制完整状态，只清空一份的快误差、慢误差和反馈变化统计。保留两份的预测权重、访问计数、年龄与近期反馈。先给予完全相同的强制后续反馈，再另外比较自由行动分支。强制反馈分析不计入主闭环性能。")
    addtable(["检查","实际结果"],[
        ["相同后续反馈下，步长平均绝对差",f"{intervention['forced_tail_mean_abs_alpha_difference']:.8f}"],
        ["相同后续反馈下，预测平均绝对差",f"{intervention['forced_tail_mean_abs_prediction_difference']:.8f}"],
        ["保留统计 − 清空统计：自由行动收益差",ci(intervention['intact_minus_erased_free_tail_reward'])],
        ["可塑权重与等价状态写法：轨迹最大差",f"{intervention['state_equivalent_max_trace_difference']:.1f}"]])
    paragraph("H2 的窄义机制检查成立：历史统计对后续更新有非零作用，但效应很小；收益差区间覆盖零，没有证据表明保留这些统计在这里有益。等价状态实现产生相同轨迹是方程等价的检查，不是新发现，也不说明意识或生命性。")

    section("事后补充检查：环境的学习需求未通过校准")
    paragraph("以下不是主协议的预注册结果。发现步长范围问题后，先另写 SUPPLEMENT_01.md，再使用新的独立种子：6 类×64 条＝384 条，每条 1024 事件。不重新训练，不修改主模型，也不覆盖主评估。")
    addtable(["方法","新独立样本平均净收益/事件"],[
        [NAMES[k],f"{v:.6f}"] for k,v in sup["overall"]["means"].items()])
    paragraph("adaptive − fixed 0.01："+ci(sup["overall"]["adaptive_minus_fixed_001"])+"。更公平的较低固定步长对照没有更差。")
    paragraph("adaptive − 冻结预测："+ci(sup["overall"]["adaptive_minus_frozen"])+"。冻结预测在这套配置下更好，因而当前实验不能作为历史学习带来收益的有效展示。",True)
    figure("supplement_comparison.png","事后补充检查使用新的样本。冻结预测对照表现更好；该结果应约束后续路线。")
    paragraph("可能的解释是资源恢复与移动代价使稳定、较均匀的采集策略已经很强，而当前预测与行动规则不能有效利用所学信息。这是解释假说，本轮没有单独操纵各因素验证。即使可读取当步真实质量的贪心策略也没有明显优于冻结预测；它不是长期最优策略，不能据此断言信息无用或学习不可能获益。")

    section("长期运行与计算连续性")
    paragraph(f"4096 事件压力测试：adaptive 平均收益 {long['adaptive']['mean_reward']:.6f}，fixed 为 {long['fixed']['mean_reward']:.6f}；最终预测均为有限值并处于 [0,1]。有界性部分来自显式裁剪，不能称为自发自维持，也不能由有限时长推出永久学习。")
    figure("long_run.png","分段收益不单调增长。数值稳定与持续获得新能力应分开判断。")
    paragraph("6 项针对关键风险的自动检查通过：行动改变环境；观测不暴露隐藏质量；相同种子与等价写法一致；完整检查点恢复后轨迹逐项一致；干预只改指定统计；批量与单候选运行一致及数值有界。检查结果记录在 VALIDATION.md。")

    section("本轮能支持什么")
    paragraph("可以支持：该机制实验闭环可运行并可复现；历史统计确实进入学习更新；可以通过检查点和选择性清空实施干预；主分析与事后诊断能够分开保存。")
    paragraph("不能支持：历史调节优于匹配的固定步长；系统形成了自主目标或内生价值；出现数字生命、欲望、意识、人格；当前小控制器可以直接扩大成 0.xB 基底；资源预算与神经参数等价于生物代谢。")
    paragraph("建议暂不扩大参数。下一轮先校准任务：确认在相同观察/行动接口下，可利用信息的学习或规划基线稳定优于冻结预测，再比较固定可塑性与历史调节。需要增加延迟后果或必须记住的隐变量时，仍须保留简单基线，不能故意设计只有预定模型才会成功的环境。")

    section("交付与复现")
    paragraph("完整包包括 PROTOCOL.md、SUPPLEMENT_01.md、README.md、独立源代码、6 项检查、训练参数与日志、主评估逐轨迹 CSV、干预原始数组、完整检查点、压力测试结果、图表和本报告。运行方式见 README.md。预先记录的协议及代码 SHA-256 位于 results/run_manifest.json；补充检查有单独时间与哈希。")
    paragraph("本轮报告只使用本轮真实输出。旧 WUYU_0A 的偏好强化结果未混入。")
    section("参考来源")
    links=[("Backpropamine：神经调制可塑性","https://arxiv.org/abs/2002.10585"),
           ("Evolving Plasticity：个体内学习规则","https://arxiv.org/abs/1904.01709"),
           ("Sensorimotor Lenia：自组织个体与扰动检验","https://arxiv.org/abs/2402.10236"),
           ("Loss of plasticity：持续学习的长期问题","https://www.nature.com/articles/s41586-024-07711-7")]
    for label,url in links:
        md.append(f"- [{label}]({url})")
        ht.append(f"<p><a href='{url}'>{html.escape(label)}</a></p>")
    paragraph("这些来源支持机制选择与研究问题。本代码为独立简化实现，不声称复现上述论文的实验结果。")
    (ROOT/"REPORT.md").write_text("\n\n".join(md)+"\n",encoding="utf-8")
    css="""*{box-sizing:border-box}body{margin:0;background:#f3f1eb;color:#233133;font-family:system-ui,-apple-system,'Noto Sans CJK SC','Microsoft YaHei',sans-serif;line-height:1.8}main{max-width:920px;margin:auto;padding:48px 36px 64px;background:#fffdfa}.eyebrow{font-size:12px;letter-spacing:2px;color:#42736d}h1{font-size:40px;line-height:1.35;letter-spacing:-1px;margin:16px 0}.subtitle{color:#637270;font-size:18px}h2{margin-top:44px;padding-top:18px;border-top:1px solid #d8dfd9;font-size:23px}p{margin:16px 0}.callout{padding:20px 22px;background:#eaf1ea;border-left:4px solid #387770;border-radius:0 8px 8px 0;font-weight:600}.table-wrap{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65}th,td{text-align:left;border-bottom:1px solid #dce1dc;padding:12px 14px;vertical-align:top}th{background:#edf1ec;white-space:nowrap}td{min-width:120px}figure{margin:28px 0}img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:#637270;padding-top:10px}a{color:#286f69;text-decoration-thickness:1px;text-underline-offset:3px}@media(max-width:600px){main{padding:28px 18px 44px}h1{font-size:31px}h2{font-size:21px}.subtitle{font-size:16px}p{font-size:15px}th,td{padding:10px;font-size:13px}.callout{padding:16px}}"""
    page="<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>WUYU 第一轮实验报告</title><style>"+css+"</style></head><body><main>"+"\n".join(ht)+"</main></body></html>"
    (ROOT/"REPORT.html").write_text(page,encoding="utf-8")
    print("Created REPORT.md and REPORT.html from saved results.")


if __name__ == "__main__":
    main()
