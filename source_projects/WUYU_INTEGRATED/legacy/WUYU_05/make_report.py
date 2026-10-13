from pathlib import Path
import base64
import hashlib
import html
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures'
MODES=['fixed','heuristic','learned','matched_mean','shuffled','frozen']
NAMES={'fixed':'固定双更新率','heuristic':'误差驱动公式','learned':'可学习双调节','matched_mean':'匹配平均更新率','shuffled':'打乱训练期调节序列','frozen':'分支停止神经学习'}
SHORT=['Fixed','Error rule','Learned','Mean match','Shuffled','Frozen']
KINDS=['return','change','noise'];KNAMES={'return':'旧规律返回','change':'规律反转','noise':'噪声增加'}


def fmt(x,percent=False):
    scale=100 if percent else 1;digits=1 if percent else 4;suffix='%' if percent else ''
    return f"{x['mean']*scale:.{digits}f}{suffix} [{x['low']*scale:.{digits}f}, {x['high']*scale:.{digits}f}]"


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])


def charts(summary):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#6d7e8d','#9aabb9','#067e74','#87b8b0','#c8b28b','#b5b8bb']
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),sharey=True)
    for ax,kind,title in zip(axs,KINDS,['Old rule returns','Rule changes','Feedback gets noisy']):
        xs=[summary['methods'][m]['branches'][kind]['all']['reward'] for m in MODES]
        means=np.array([x['mean'] for x in xs]);low=means-np.array([x['low'] for x in xs]);high=np.array([x['high'] for x in xs])-means
        ax.bar(np.arange(6),means,color=colors,yerr=[low,high],capsize=3,width=.72)
        ax.set_xticks(range(6),SHORT,rotation=40,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].set_ylabel('Reward per environment event');fig.suptitle('WUYU_05 | Three paired branches, 32 individuals',fontsize=14)
    fig.tight_layout();fig.savefig(FIG/'rewards.png',dpi=170);plt.close(fig)
    with np.load(OUT/'raw_learned.npz') as d:
        fig,axs=plt.subplots(2,3,figsize=(13,6),sharex=True,sharey=True)
        for col,kind in enumerate(KINDS):
            for row,key in enumerate(['rate_current','rate_replay']):
                values=d[kind+'__'+key];phase=d[kind+'__phase'];curves=[]
                for j in range(values.shape[1]):
                    seq=values[phase[:,j]==0,j]
                    curves.append([b.mean() for b in np.array_split(seq,24)])
                curves=np.array(curves);mean=curves.mean(axis=0);se=curves.std(axis=0,ddof=1)/np.sqrt(32)
                ax=axs[row,col];x=np.linspace(0,1,24)
                ax.plot(x,mean,color='#067e74',lw=2);ax.fill_between(x,mean-se,mean+se,color='#067e74',alpha=.2)
                ax.grid(alpha=.2);ax.set_ylim(0,.12)
                if row==0:ax.set_title(['Old rule returns','Rule changes','Feedback gets noisy'][col])
                if row==1:ax.set_xlabel('Relative time within first phase')
        axs[0,0].set_ylabel('Current-example rate');axs[1,0].set_ylabel('Replay rate')
        fig.suptitle('Learned modulation | Mean ± 1 SE; time normalized within each individual',fontsize=13)
        fig.tight_layout();fig.savefig(FIG/'rates.png',dpi=170);plt.close(fig)


def render(md):
    def inline(s):
        s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
        s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
        return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',s)
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('#'):
            marks,title=line.split(' ',1);out.append(f'<h{len(marks)}>{inline(title)}</h{len(marks)}>');i+=1;continue
        if line.startswith('!['):
            match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line);p=ROOT/match.group(2)
            out.append('<figure><img alt="'+html.escape(match.group(1))+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+inline(match.group(1))+'</figcaption></figure>');i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            out.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        if line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        out.append('<p>'+inline(line)+'</p>');i+=1
    css='''body{margin:0;background:#f1f5f3;color:#203632;font:16px/1.85 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:960px;margin:28px auto;padding:38px 42px;background:white;border-top:6px solid #087d70}h1{font-size:1.95rem;line-height:1.45}h2{font-size:1.25rem;margin-top:2.2em;padding-top:.5em;border-top:1px solid #dce7e2}strong,a{color:#066f63}p{margin:1em 0}code{background:#edf3f0;padding:2px 4px;overflow-wrap:anywhere}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:570px;width:100%;font-size:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #dce7e2;vertical-align:top}th{background:#edf5f0}figure{margin:26px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#5c716b}li{margin:8px 0}@media(max-width:650px){main{margin:0;padding:22px 18px}h1{font-size:1.55rem}h2{font-size:1.15rem}}'''
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_05 实验报告</title><style>'+css+'</style></head><body><main>'+''.join(out)+'</main></body></html>'


def main():
    summary=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text());charts(summary)
    gates=summary['gates'];all_pass=all(gates.values());comp=summary['comparisons']
    verdict='四项事前门槛全部通过，当前任务支持学习到的经历调节带来额外收益。' if all_pass else '实现、训练与正式评估已经完成，但未满足全部事前收益门槛；不能宣布学到的经历调节已经优于简单策略。'
    rows=[[NAMES[m],fmt(summary['methods'][m]['reward']),fmt(summary['methods'][m]['old_b'],True),f"{summary['methods'][m]['mean_current_rate']:.4f}",f"{summary['methods'][m]['mean_replay_rate']:.4f}"] for m in MODES]
    diffs=[[NAMES[m],fmt(comp['learned_minus_'+m])] for m in MODES if m!='learned']
    branch_rows=[[KNAMES[k],fmt(comp['branches_learned_minus_fixed'][k])] for k in KINDS]
    stage_rows=[]
    for k in KINDS:
        for m in ('fixed','heuristic','learned'):
            b=summary['methods'][m]['branches'][k]
            stage_rows.append([KNAMES[k],NAMES[m],fmt(b['first']['accuracy'],True),fmt(b['late']['accuracy'],True),fmt(b['recovery']['accuracy'],True)])
    gate_rows=[['G1：全程收益优于 fixed',gates['G1_net_reward']],['G2：三分支无超过 .02 的明确收益损失',gates['G2_each_branch']],['G3：旧 B 无超过 3 个百分点的明确损失',gates['G3_old_b']],['G4：优于平均率与打乱序列对照',gates['G4_history_correspondence']]]
    resources=[]
    for m in MODES:
        run=json.loads((OUT/('runtime_'+m+'.json')).read_text());r=run['resources']
        resources.append([NAMES[m],r['policy_parameters'],f"{r['agent_arrays_bytes_per_individual']/1024:.1f}",r['gradient_calls'],f"{run['seconds']:.2f}"])
    md=f'''# WUYU_05：经历如何调节持续学习

2026-10-10 · 沿 WUYU_03/04 的实际实现继续推进

## 这次的结论

{verdict}

learned 相对 fixed 的每步环境收益差为 **{fmt(comp['learned_minus_fixed'])}**。与误差驱动公式的差为 **{fmt(comp['learned_minus_heuristic'])}**。这两项必须一起看。

本轮实际完成了一个共享神经核心、工具动作、重算回放及两个可训练调节量。两个输出分别决定新反馈与旧回放的更新强度。当前没有永久冻结旧规则，也没有为新规则预留额外专家槽位；frozen 只作为干预对照。

## 实际运行了什么

- 神经核心沿用 17 输入、24 隐状态的 1033 参数 RNN；工具控制保留 250 个 Q 值和 Dyna 更新。
- 经验容量 256；每条经验保存最多 8 个前序事件。用当前权重重算回放初态，包含其间没有有效观察的事件。
- learned 为 16 参数双输出调节器，使用当前误差、历史误差统计、预测熵和旧样本误差等 8 个特征。调节器参数经外层训练后固定；神经权重、隐状态、记忆、快慢误差统计在个体运行中继续更新。
- fixed、heuristic、learned 各搜索 48 个候选，训练个体各 8 个；每家族的 3 个候选再在另 8 个体上验证。正式测试 32 个独立种子，各有 3 个配对分支。
- 共同前缀 1536 事件，每个分支 1536 事件。分支统计先按个体合并；没有将 96 条分支当成 96 个独立个体。
- 整个训练、验证、派生对照和正式运行记录用时 {manifest['seconds']:.2f} 秒。运行时间受机器影响，不代表其他设备速度。

## 三种经历及恢复

共同学过 A/B 后，分别进入旧 A 返回、A 规律反转、A 反馈噪声增加。每支随后经过正常 B，再返回原 A。各个体的变化时刻不同，正式阶段更长。真实原因与噪声率不提供给代理，但 A/B 情境输入仍可见。

{table(['方法','全程奖励/步，95%区间','最终旧 B 准确率','当前更新率均值','回放更新率均值'],rows)}

旧 B 的读数来自独立克隆上的受控无参数更新测试，不是自主工具表现。更新率均值只统计相应梯度有效的正式分支事件。

![三种分支的全程环境收益；误差棒为个体级 95% bootstrap 区间。](figures/rewards.png)

## 是否超过简单解释

{table(['比较对象','learned − 对照：每步奖励差'],diffs)}

matched_mean 使用 learned 训练轨迹的两种平均率；shuffled 从训练轨迹成对抽取更新量，切断与正式经历的对应。它们不使用正式测试的未来调节序列。各条件都从出生即使用自身策略，因此比较包含发展轨迹差异，不能只归因于挑战阶段的即时调节。

{table(['分支','learned − fixed：全程奖励差'],branch_rows)}

旧 B 保留差为 {fmt(comp['old_b_learned_minus_fixed'],True)}（区间以百分点计）。这里显示的差值百分数对应准确率差，不是相对百分比变化。

## 适应的时间过程

{table(['经历','方法','首段前128步','首段末128步','恢复段末128步'],stage_rows)}

表中准确率以取得有效观察的回答为分母。维护和失效动作仍在前面的每步奖励中计入成本，因此不能只看这张准确率表。

![learned 在三种首段经历中的调节轨迹；每个体按阶段相对时间归一，阴影为均值上下一个标准误，仅作描述。](figures/rates.png)

更新量随经历变化本身不证明有用；是否有增益以同预算对照和事前门槛为准。

本轮轨迹显示：噪声首段里，learned 的当前更新强度逐渐提高；规律反转首段则在开始后逐渐降低。它尚未形成期望中的可靠分工。正式全程当前更新率均值约为 0.0302，fixed 为 0.0545。这提供了“调节是否过度保守、是否把持续误差当成应该加强学习的信号”的诊断线索，但曲线不足以确定原因。

## 事前门槛

{table(['门槛','结果'],[[x,'通过' if y else '未通过'] for x,y in gate_rows])}

区间为 32 个体级配对 bootstrap，4000 次。多个次级比较没有多重校正。全部门槛、搜索预算和种子在训练前写入 PROTOCOL.md；正式期间的数值源文件哈希未变。

本轮四项门槛均未通过，但含义不同：G1、G4 的差值区间包含零，表示额外收益证据不足；规律反转分支的差值区间整体低于零，对应 G2 中的具体弱项；G3 的旧 B 差值均值接近零但区间较宽，因此不能保证损失小于预定容忍范围，也不能直接断言旧 B 已受到明确损害。噪声分支的均值改善同样包含零，不能宣布它已经学会抑制噪声。

## 资源与工程验证

{table(['方法','调节参数数','代理数组 KiB/个体','单支含前缀梯度调用','正式用时秒'],resources)}

每条件同样计算当前与回放梯度，零更新量也计算梯度。参数数只指调节器，核心另有 1033 神经参数和 250 个 Q 值。数组内存是批量运行中的均摊数字数组，不是整个进程或全部世界存储；shuffled 还携带训练更新量池。梯度调用列为单支含公共前缀，不是三个分支全部计算总量。

8 项机制测试通过。独立核验从原始动作重算了工具次数、可用观察、噪声线索、反馈与收益；重新计算 learned 的历史统计和更新率；中途恢复后的 256 个事件与正式记录逐项相等，继续至分支结束后的全部网络数组相等。完整状态再保存/恢复后也逐项一致。

## 这一步的意义与限制

这次把双学习调节真正接入了感知—行动—反馈过程，可以据实比较是否有用。所有结果，包括未达标项，都保留在原始数组中。

需要保留的限制：可见 A/B 情境、二元正确性反馈、固定容量与前序长度、手工定义的调节输入和学习目标、8 个训练及 8 个验证个体，以及很窄的异或映射家族。正式种子的随机极性独立，但不意味着出现了训练时没有过的映射类型。因此它尚未验证全新复杂任务中的学习方法迁移。

replay 的前序状态在当前样本更新前重算；随后先做当前更新，再做回放更新。这是一次有限近似，所有条件一致，不能把它描述成对完整历史的精确重新解释。

外层 CEM 学到调节参数，内层继续改神经权重及持续统计；本轮没有证明个体能自行创造学习目标、修改学习算法代码、无限扩展技能或形成欲望与意识。

## 下一步如何由结果决定

若 learned 有稳定额外收益，下一步应从相同前缀检查点切换调节方式，进一步隔离调节本身的因果贡献，再转入未见任务结构。若 learned 未超过简单策略，先分析它在哪种经历中吃亏，检查历史统计、回放相关性与训练范围，不直接扩大参数。

依据本轮结果，下一步优先建议是：从同一已学检查点出发，只替换两个更新量，比较动态调节、固定较快更新及不同回放强度；先区分“更新幅度不足”“旧回放产生干扰”“调节输入不足以区分变化与噪声”三个解释。这是后续诊断建议，本轮没有追加运行或据此改写正式成绩。

复现入口是 run_experiment.py；独立核验入口是 audit_results.py；demo.py 从已保存的实际个体运行 128 个事件。默认都是有限仿真，不是后台常驻服务。

## 机制参考

- [R2D2 作者页：循环回放的状态陈旧与表示漂移](https://willdabney.com/publication/r2d2/)
- [Differentiable plasticity：可塑性本身可训练](https://proceedings.mlr.press/v80/miconi18a.html)
- [Learning to reinforcement learn：循环动态中的学习过程](https://arxiv.org/abs/1611.05763)

这些是机制参考；本实现采用小型 CEM 外层搜索与显式梯度更新，没有复现上述论文。
'''
    (ROOT/'REPORT.md').write_text(md)
    (ROOT/'REPORT.html').write_text(render(md))
    print(json.dumps({'report':str(ROOT/'REPORT.html'),'gates':gates,'reward_difference':comp['learned_minus_fixed']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
