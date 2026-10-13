"""Build static scientific figures and a self-contained Chinese report."""
import base64
from collections import defaultdict
import gzip
import html
import io
import json
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze import read_csv,estimate
from run import write_json


def fmt(x,n=4):return f'{x:.{n}f}'
def interval(r,n=4):return f"{fmt(r['mean'],n)} [{fmt(r['lower95'],n)}, {fmt(r['upper95'],n)}]"
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def render(md):
    def inline(s):
        s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
        return s
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>');i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):rows.append([x.strip() for x in lines[i].strip('|').split('|')]);i+=1
            out.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in rows[0])+'</tr></thead><tbody>')
            for row in rows[2:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        image=re.fullmatch(r'!\[([^]]+)\]\(([^)]+)\)',line)
        if image:
            data=base64.b64encode((ROOT/image[2]).read_bytes()).decode();out.append(f'<figure><img alt="{html.escape(image[1])}" src="data:image/png;base64,{data}"><figcaption>{html.escape(image[1])}</figcaption></figure>');i+=1;continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>');i+=1;continue
        if line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>');continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ','```','![')):para.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(para))+'</p>')
    return '\n'.join(out)


def make_report():
    output=ROOT/'outputs/holdout';c=json.loads((ROOT/'formal_protocol.json').read_text());s=json.loads((output/'summary.json').read_text())
    audit=json.loads((output/'audit.json').read_text());repro=json.loads((output/'reproduction.json').read_text())
    assert audit['status']==repro['status']=='passed'
    d=json.loads((ROOT/'reporting/derived/exploratory_diagnostics.json').read_text());p=s['checkpoints']['400'];n=s['n']
    rows=read_csv(output/'branches.csv');individual=read_csv(output/'individual_effects.csv')
    bidx={(int(r['seed']),int(r['checkpoint']),r['intervention'],int(r['task'])):float(r['body_loss']) for r in rows}
    seeds=range(41000,41000+n);absolute={}
    for label in ('retain','related','irrelevant','restore'):
        values=[]
        for seed in seeds:
            ivs=[('retain' if label=='retain' else 'restore_axis0' if label=='restore' else f'erase_axis{k if label=="related" else 1-k}') for k in (0,1)]
            values.append(statistics.mean(bidx[seed,400,ivs[k],k] for k in (0,1)))
        absolute[label]=estimate(values,c)
    timeline={name:dict(loss=[0.]*48,wait=[0.]*48,count=[0]*48) for name in ('life','frozen_after_change')};example=[];example_events=[]
    with gzip.open(output/'events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['seed']==41000:example_events.append(r)
            name=r['branch']
            if name not in timeline:continue
            tick=r['before']['tick'];bucket=tick//50;v=timeline[name]
            v['loss'][bucket]+=sum(x*x for x in r['after']['body']);v['wait'][bucket]+=r['action']==0;v['count'][bucket]+=1
            if r['seed']==41000 and name=='life':example.append(r)
    for v in timeline.values():
        for metric in ('loss','wait'):v[metric]=[x/count if count else float('nan') for x,count in zip(v[metric],v['count'])]
    figdir=ROOT/'reporting/figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    ticks=c['checkpoints'];x=list(range(len(ticks)))
    for i,t in enumerate(ticks):
        vals=[float(r['selective_effect']) for r in individual if int(r['checkpoint'])==t]
        jitter=[i+(j%17-8)*.007 for j in range(len(vals))];axes[0].scatter(jitter,vals,s=7,color='#263d44',alpha=.14,zorder=1)
        r=s['checkpoints'][str(t)]['selective_effect'];color='#197866' if r['mean']>0 else '#b94a37'
        axes[0].errorbar(i,r['mean'],yerr=[[r['mean']-r['lower95']],[r['upper95']-r['mean']]],fmt='o',capsize=5,color=color,markersize=7,zorder=3)
    axes[0].axhline(0,color='#8a9899',lw=1);axes[0].axvline(2.5,color='#b94a37',ls='--',lw=1)
    axes[0].set(xticks=x,xticklabels=['400*','800','1200','2000','2400'],xlabel='Checkpoint (* primary)',ylabel='Selective interaction S',title='A  Useful history can become harmful')
    names=['retain','related','irrelevant','restore'];xx=range(4);means=[absolute[k]['mean'] for k in names]
    axes[1].bar(xx,means,color=['#197866','#b94a37','#b8a66a','#628c9b'],width=.62)
    axes[1].errorbar(list(xx),means,yerr=[[absolute[k]['mean']-absolute[k]['lower95'] for k in names],[absolute[k]['upper95']-absolute[k]['mean'] for k in names]],fmt='none',capsize=4,color='#203237')
    for i,v in enumerate(means):axes[1].text(i,v+.007,f'{v:.4f}',ha='center',fontsize=10)
    axes[1].set(xticks=list(xx),xticklabels=['Retain','Erase\nrelated','Erase\nother','Restore'],ylabel='12-step mean body loss J (lower better)',ylim=(0,.20),title='B  Selective cost at tick 400')
    fig.savefig(figdir/'history_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained');xs=[25+50*i for i in range(48)]
    for name,color,label in [('life','#197866','Online learning'),('frozen_after_change','#b94a37','Frozen learned models')]:
        axes[0].plot(xs,timeline[name]['loss'],label=label,color=color,lw=2)
        axes[1].plot(xs,timeline[name]['wait'],label=label,color=color,lw=2)
    axes[0].set(ylabel='Mean body loss',title='A  Continuing lives: 128 seeds, 50-step bins');axes[0].legend(loc='upper left')
    axes[1].set(ylabel='Fraction WAIT',ylim=(-.03,1.05),title='B  Waiting continues to advance the world')
    axes[2].plot([r['after']['tick'] for r in example],[r['after']['body'][0] for r in example],label='Body 0',color='#197866',lw=.9)
    axes[2].plot([r['after']['tick'] for r in example],[r['after']['body'][1] for r in example],label='Body 1',color='#a68031',lw=.9)
    axes[2].plot([r['after']['tick'] for r in example],[r['after']['fatigue'] for r in example],label='Fatigue',color='#738490',lw=.6,alpha=.65)
    axes[2].set(ylabel='State',xlabel='Life tick',title='C  Prespecified example: seed 41000 (complete main life)');axes[2].legend(loc='upper left',ncol=3)
    for ax in axes:ax.axvline(1600,ls='--',color='#b94a37',lw=1);ax.set_xlim(0,2400);ax.grid(alpha=.12)
    fig.savefig(figdir/'continuing_lives.png',dpi=160);plt.close(fig)
    with open(ROOT/'reporting/derived/example_41000_events.jsonl.gz','wb') as raw:
        with io.TextIOWrapper(gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0),encoding='utf-8') as f:
            for row in example_events:f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
    # Explicitly exploratory sensitivity; the primary estimate includes all 128.
    matched=[float(r['selective_effect']) for r in individual if int(r['checkpoint'])==400 and int(r['full_tool_buffers'])]
    sensitivity=estimate(matched,c);write_json(ROOT/'reporting/derived/report_metrics.json',dict(primary_absolute=absolute,matched_118_sensitivity=sensitivity))
    checkpoint_table=table(['检查点','S 均值 [95% 区间]','S>0／S≤0','缓冲全满','样本数不等'],[
        [str(t)+('（主检验）' if t==400 else ''),interval(s['checkpoints'][str(t)]['selective_effect']),
         f"{s['checkpoints'][str(t)]['selective_effect']['positive']}／{n-s['checkpoints'][str(t)]['selective_effect']['positive']}",
         f"{s['checkpoints'][str(t)]['full_buffer_count']}/{n}",str(n-s['checkpoints'][str(t)]['equal_removed_count'])] for t in ticks])
    absolute_table=table(['处理','平均 J [95% 区间]','与保留的配对差'],[
        ['保留全部经历',interval(absolute['retain']),'—'],['删除当前任务相关模型',interval(absolute['related']),interval(p['related_minus_retain'])],
        ['删除另一任务模型',interval(absolute['irrelevant']),interval(p['irrelevant_minus_retain'])],['删除后完整恢复',interval(absolute['restore']),'逐步行为完全一致']])
    life_table=table(['时段','在线 J','冻结 J','在线 WAIT 占比'],[
        [label,fmt(s['life_windows'][f'online/{label}']['mean_body_loss']),fmt(s['life_windows'][f'frozen/{label}']['mean_body_loss']) if f'frozen/{label}' in s['life_windows'] else '—',
         f"{s['life_windows'][f'online/{label}']['waits']/(int(label.split('-')[1])-int(label.split('-')[0])):.1%}"] for label in ('1200-1600','1600-1700','1700-2000','2000-2400')])
    pred_table=table(['检查点','删除的预测选择性','均值 MSE − 条件 MSE'],[[t,interval(s['checkpoints'][t]['prediction_selectivity'],6),interval(s['checkpoints'][t]['conditional_mse_improvement'],6)] for t in ('400','800','1200','2000','2400')])
    first={r['branch']:r for r in reversed(example_events) if r['before']['tick']==400 and r['branch'].startswith('probe_400_')}
    case=[]
    for iv,label in [('retain','保留'),('erase_axis0','相关删除'),('erase_axis1','另一轴删除'),('restore_axis0','恢复')]:
        row=first[f'probe_400_{iv}_0'];case.append([label,str(row['action']),fmt(bidx[41000,400,iv,0]),','.join(str(k) for k in row['decision']['knowledge_samples'])])
    case_table=table(['种子 41000，任务 0','首动作通道','12 步 J','动作 0–4 可用样本数'],case)
    md=f'''# WUYU：经历既能支持行动，也可能妨碍重新学习

持续状态与经历干预实验 · 2026-10-12 · 32 个开发个体／128 个正式留出个体

**本轮完成了一个能持续运行、保存恢复、依据实际经历更新模型的最小个体，并验证了模型中的经历具有任务选择性的因果作用。环境变化后，这些旧经历又会造成明显的适应迟滞。**

这是一项可复查的低维学习与控制结果。它没有实现新欲望、技能生成、社会关系或价值标准形成，也不构成人格或体验已经出现的证据。

## 1. 最重要的三个结果

- **稳定期，相关经历确实在参与选择。** 第 400 步主交互 S={interval(p['selective_effect'])}，128/128 个体 S>0。相关删除的平均身体损失由 {fmt(absolute['retain']['mean'])} 升至 {fmt(absolute['related']['mean'])}；删除另一任务模型只有较小影响。
- **变化后，旧经历可能有害。** 工具作用在第 1600 步反转。第 2000 步 S={interval(s['checkpoints']['2000']['selective_effect'])}，120/128 个体 S<0；相关删除反而改善诊断表现。这是次要、描述性结果。
- **持续更新有用，但远未恢复原水平。** 最后 400 步在线 J={fmt(d['late_mean'])}，冻结模型 J={fmt(s['life_windows']['frozen/2000-2400']['mean_body_loss'])}；在线表现仍是变化前平均损失的 {d['ratio_of_means']:.1f} 倍。127/128 个体晚期损失高于自身变化前。

![经历效应与主检验绝对表现。散点为独立种子，误差线为按种子 bootstrap 的 95% 区间。虚线左右对应变化前后；只有第 400 步为正式主检验。](reporting/figures/history_effects.png)

## 2. 实际做成了什么

世界有两个连续身体读数、一个会影响操作幅度的疲劳变量、四个不透明工具动作和 WAIT。WAIT 仍使时间与身体变化，并恢复疲劳。每个工具的作用轴、方向和幅度对代理隐藏。材料条件在 0.2、0.8 之间切换；诊断采用 0.5。

每个动作维护最近 24 次实际经历，以给定特征 `[1,b0,b1,f,m,f*m]` 拟合下一步变化。代理用预测的身体与疲劳变化、活动成本和给定的访问不足偏向选择动作，收到真实后果后再更新。没有主轨迹身体重置，也没有传入训练／考试或工具变化标签。

给定的零理想点、平方偏差评分、特征结构、模型族、可塑性规则和探索偏向依然承担核心解释工作。学得的是具体作用关系；“最初为什么在乎这些数值”仍未解释。

保存内容包括世界、有限经历窗、拟合统计量与参数、访问次数、决策随机状态。除 13 项机制测试外，命令行跨进程的 50+50 步与一次 100 步也得到完全相同的事件和最终状态。

## 3. 干预为何能区分一般损伤与任务相关作用

在第 400、800、1200、2000、2400 步保存主轨迹。评估器复制个体，分别保留经历、清空轴 0 的两个工具模型、清空轴 1 的两个工具模型、清空轴 0 后完整恢复。WAIT 模型、其他模型、访问次数及随机状态均保留。

两个诊断任务分别将身体对齐为 `(0.6,0)` 和 `(0,0.6)`，疲劳 0.25、材料 0.5。每个分支继续实际行动和学习 12 步，外生噪声由种子与绝对时刻索引。诊断副本的重置不反馈到主轨迹。

`J = 12 个动作后 (b0²+b1²) 的平均；越低越好。`

`S = [(J(删轴0,任务0) − J(删轴1,任务0)) + (J(删轴1,任务1) − J(删轴0,任务1))] / 2。`

正 S 表示相关删除的代价更高；负 S 表示相关删除相对更有利。每个种子只贡献一个配对交互，分支和时刻不扩充样本量。

## 4. 主检验：支持有限而明确的历史作用

{absolute_table}

上表先在每个个体内平均两个任务，再跨 128 个体估计；绝对 J 区间为描述性补充。预测选择性为 {interval(p['prediction_selectivity'],6)}：误差增加主要位于被清空的工具通道，其他通道保持。恢复分支与保留分支的 12 步预测、选择和后果完全一致；全 5 个检查点共核验 1,280 对轨迹。

118/128 个体在主检验时四工具缓冲全满，两个删除各清空 48 条经历。另 10 个个体容量相同但样本数不完全相等，**全部仍计入主结果**。探索性完整缓冲子集 S={interval(sensitivity)}，不能替代全体主检验；第 800 步全体样本匹配时也出现正向效应。

恢复的完全一致是工程可逆性检查，不能作为第二份独立的心理形成证据。此处的因果干预直接清空模型与样本，支持的是这一架构内学得模型的功能作用；不等同于主体主动遗忘或经历归属。

## 5. 全部检查点与失败分母

{checkpoint_table}

第 2000 步相关删除相对保留的配对差为 {interval(s['checkpoints']['2000']['related_minus_retain'])}，显示减少旧模型影响可以暂时改善这个诊断任务。第 2400 步的 S 区间仍跨 0；不能声称所有个体已经恢复稳定的有益历史作用。

未全满、样本不等和 S≤0 描述不同问题，可重叠；没有按结果筛掉任何个体。主检验之外的区间没有多重比较校正，不能逐项当作新的确认性发现。

## 6. 作用反转后：学习优于冻结，不等于适应成功

{life_table}

冻结分支从第 1600 步的同一世界与同一代理状态开始，只关闭模型更新；访问次数与决策继续运行。它随后处在不同动作造成的状态中，因此长期配对差是这整套在线更新机制的总效果，不是单一参数的局部效果。

变化后前 100 步，在线个体约 96.7% 的时间选择 WAIT；世界仍在漂移。等待可以恢复能力，却不能替代对已经改变的工具重新取样。最后 400 步相比变化前的配对损失增量为 {interval(d['late_minus_prechange'])}（探索性）。

![连续主轨迹、冻结对照与预定个体示例。前两图是 128 个体每 50 步平均；第三图完整显示预先指定的最低留出种子 41000。竖线为工具作用反转。](reporting/figures/continuing_lives.png)

探索性记忆审查发现：第 2000 步工具模型样本中，平均 {d['checkpoints']['2000']['mean_pre_change_sample_fraction']:.1%} 仍来自变化前；512 个工具模型中有 {d['checkpoints']['2000']['total_tool_models_not_revisited']} 个完全未在变化后重新访问。到第 2400 步，旧样本占比仍为 {d['checkpoints']['2400']['mean_pre_change_sample_fraction']:.1%}。这支持继续检查“访问减少—旧模型保留—重新采样不足”的反馈，但尚未通过独立机制干预确认其全部因果链。

在第 2400 步的实际主轨迹状态上，四工具反事实无噪声预测 MSE 均值为 {d['checkpoints']['2400']['mean_tool_mse_actual_state']:.5f}，当时所选动作对应值只有 {d['checkpoints']['2400']['mean_chosen_mse_actual_state']:.5f}。两者所覆盖动作不同，不能直接作优劣统计；它提醒我们，**只看被选动作的低误差，可能漏掉被回避工具的错误模型**。

## 7. 条件模型并未证明全面优于简单模型

{pred_table}

均值对照和条件模型使用同一动作完全相同的最近经历；对照只比较预测，不是另跑一套均值决策代理。第 400 步的差值区间跨 0，只有 30/128 个体的条件模型更好；第 2000、2400 步均值模型的平均误差反而较低。不能把“接入了条件模型”写成“证实了条件模型优势”。

给定特征包含正确类型的疲劳×材料交互，但数据来自自主选择而非全面覆盖。身体边界截断、状态稀疏区、更新样本混合及错误的条件外推都可能有关；当前实验没有把这些因素单独分离。材料 0.5 仅在材料维度上是插值，不保证整个诊断状态处于每个工具的已见分布中。

## 8. 一个未按结果挑选的完整个体

示例固定为正式种子起点 41000，选择规则在正式运行前写入。上图第三幅包含其完整主轨迹；`reporting/derived/example_41000_events.jsonl.gz` 包含它的全部 3,680 个事件，包括诊断和冻结分支。

{case_table}

通道 0 为 WAIT，1–4 是个体尚需从经历学习其作用的工具接口。该示例只用于定位具体决策，不代表全体结论；全量原始记录保留每一步的五个候选预测、评分分解、访问量、动作与真实后果。

## 9. 开发选择、审计和一次完整性故障

开发前固定任务参数、选择规则与预算；32 个开发个体中，第 400 步有 29/32（90.625%）工具缓冲全满，按“最早达到 90%”规则选为主时点。样本量按开发 S 的标准差和 0.025 的目标半区间选定：128 个体预期半宽 0.00628，因此没有增加到 256 或 512。未根据效应最大的时点选题，也未在正式结果出来后调整参数或扩样。

正式种子 41000–41127，开发种子 31000–31031；每条主生命 2,400 步。正式数据共 471,040 个事件、768 份检查点、2,560 次干预、32,000 项预测探针及 5,120 条诊断轨迹。全部审计通过：连续性、近期真实经历来源、参数正规方程、选择分数、干预选择性、恢复一致性、冻结模型不更新、统计汇总回算和原始文件哈希。

首次运行完成后，事件压缩文件发生尾部不完整，能够读取 470,847 条而非 471,040 条。检测后未改写预期哈希：独立整轮重跑产生的完整事件文件与首次完成清单的原始 SHA256 完全相同，再据此恢复。其他 7 项原始输出也匹配；加上分析、个体效应及审计，共 11 项文件逐字节一致。失败审计日志和处理记录随包提供。文件损坏的底层原因未定位，不能假装这次故障没有发生。

源码与协议的本地时间戳、哈希证明包内固定与运行顺序，不等于外部注册平台的预注册。审计中的物理重放复用同一世界实现，另以解析机制测试和独立正规方程计算补充核验；它仍不是第二套独立模拟器。

## 10. 对物欲理论推进了哪里

已增加的证据是：持续状态下，真实经历形成的有限作用模型可以选择性地改变未来预测和行动；历史的作用还可能随世界变化由有益变为有害。外部日志与代理实际可用记忆已经区分，保存与恢复也有可检验接口。

仍未解释：评价方向怎样形成；对象和技能怎样出现；新目标内容怎样产生；关系或共同标准怎样获得地位；个体怎样理解和修改自身；这些过程是否伴随体验。本实验的固定二维稳态标准无法承担这些问题。

“删除相关模型有影响”不足以证明独特个人历史；当前效果可由学习的工具作用关系解释。研究者在诊断中选择删除哪些模型，代理没有主动选择改造自己的经历。疲劳只是影响动作效果的环境变量，不是主观疲劳或痛苦的证据。

## 11. 下一步的具体优先级

**先把“知道自己可能已经错了，并愿意重新取得证据”做成可区分的机制，再进入技能与目标内容扩展。** 本轮失败已经提供了比扩大模型更明确的工程问题。

| 下一项实验 | 具体对照 | 进入后续工作的门槛 |
| --- | --- | --- |
| 错误发现与重新取样 | 当前累计访问偏向、固定小概率取样、近期未访问偏向、预测矛盾触发取样；原始动作预算一致 | 在未知变化时点下缩短恢复时间，同时记录稳定世界代价；不能把变化标签喂给代理 |
| 有选择地更新旧知识 | 最近窗口、时间衰减、矛盾后局部更新；简单均值与条件模型共享数据 | 分离采样问题、模型外推问题与更新问题；旧规律再次出现时也需测试保留代价 |
| 状态条件模型的实际价值 | 数据量匹配的均值模型、条件模型和覆盖不足时回退模型 | 在需要状态条件的任务上获得可复现优势；不能由模型名称推断理解能力 |
| 技能支持的可达结果扩展 | 可复用序列与原始动作基线，展开动作逐步计费 | 新候选确由经历和能力产生，并在检验前改变选择；另论评价来源 |

下一轮需要同时保留稳定世界、真正规律改变、短暂噪声和旧规律回归四类情况，避免把过度遗忘包装成适应。上表是后续设计建议，本轮未运行这些新方法。社会关系和评价形成仍在总路线中，不能用控制任务成功替代。

## 12. 复查入口

- `README.md`：从头运行、暂停恢复与数据说明。
- `formal_protocol.json`、`FORMAL_FREEZE.json`：唯一主时点、预算及冻结内容。
- `outputs/holdout/summary.json`：精确统计；`individual_effects.csv`：全部个体效应。
- `outputs/holdout/audit.json`、`reproduction.json`：审计与 11 项逐字节复现。
- `validation/RAW_LOG_INCIDENT.json`：日志尾部缺失、原哈希及按原哈希恢复的记录。
- `GIVEN_AND_LEARNED.md`：全部给予条件及结论边界。
- `references`：承接方案、理论工作底稿与旧源码清单；旧阶段编号不在本轮重新定义。

本报告根据本次代码、协议和原始数据生成。没有借外部文献替这次实验作理论背书。完整包提供开发与留出的全量独有原始数据；逐字节相同的重跑大文件不重复存储，两轮清单与复现凭证均保留。
'''
    name='WUYU_Continuing_History_Report_2026-10-12';(ROOT/(name+'.md')).write_text(md)
    css='''*{box-sizing:border-box}body{margin:0;background:#edf2ee;color:#183438;font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.8}main{max-width:1100px;margin:0 auto;padding:58px 64px 80px;background:#fffdf8;box-shadow:0 0 50px #162f3810}h1{font-size:34px;line-height:1.35;letter-spacing:-.5px;margin:10px 0 26px;max-width:900px}h2{font-size:23px;margin-top:50px;padding-top:20px;border-top:2px solid #ccdcd5;color:#176854}p,li{font-size:16px}strong{color:#0c6355}code{background:#edf2ed;padding:2px 5px;border-radius:3px;font-size:.9em;overflow-wrap:anywhere}pre{overflow:auto;padding:20px;background:#edf2ed}.table-scroll{overflow-x:auto;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6}th{background:#1b514b;color:white;text-align:left}th,td{padding:12px 14px;border-bottom:1px solid #d8e1d9;vertical-align:top}tbody tr:nth-child(even){background:#f0f4ed}figure{margin:30px 0}figure img{width:100%;display:block;background:#fafbf9}figcaption{font-size:13px;color:#607573;margin-top:12px}.eyebrow{font-size:12px;letter-spacing:3px;font-weight:700;color:#53786c}.badge{display:inline-block;padding:4px 10px;border:1px solid #8eafa2;border-radius:20px;margin-right:8px;font-size:12px}.footer{margin-top:48px;font-size:12px;color:#73857c}@media(max-width:700px){main{padding:30px 20px}h1{font-size:28px}h2{font-size:21px}p,li{font-size:15px}th,td{min-width:110px;padding:10px}figure{margin-left:-10px;margin-right:-10px}}@media print{body{background:white}main{box-shadow:none;padding:0}h2{break-after:avoid}figure,tr{break-inside:avoid}}'''
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>WUYU 持续状态与经历作用实验</title><style>'+css+'</style></head><body><main><div class="eyebrow">WUYU · RESEARCH RECORD · 2026.10.12</div><p><span class="badge">源码与协议冻结</span><span class="badge">128 个留出个体</span><span class="badge">全量审计＋独立复现</span></p>'+render(md)+'<p class="footer">WUYU Continuing History Study · 本文件包含全部图像，可独立打开。</p></main></body></html>'
    (ROOT/(name+'.html')).write_text(page)
    print(json.dumps(dict(report=name,characters=len(md),html_bytes=len(page.encode()),figures=2,example_events=len(example_events),main_example_events=len(example)),ensure_ascii=False))


if __name__=='__main__':make_report()
