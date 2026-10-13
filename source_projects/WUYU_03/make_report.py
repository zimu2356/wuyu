"""Rebuild a self-contained Chinese report and scientific plots from saved data."""
from pathlib import Path
import base64
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wuyu03.stats import interval

ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'results/main';SUP=ROOT/'results/supplement';FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
A=json.loads((MAIN/'legacy.json').read_text());B=json.loads((MAIN/'core.json').read_text())
S=json.loads((SUP/'summary.json').read_text());M=json.loads((MAIN/'manifest.json').read_text())
SM=json.loads((SUP/'manifest.json').read_text());SEL=json.loads((MAIN/'selection.json').read_text())
LABEL={'online':'在线学习','repeat':'重复当前经历','replay':'近期回放','frozen':'首阶段后冻结','no_memory':'移除跨时刻记忆'}
COLORS={'online':'#087f8c','repeat':'#ad8540','replay':'#7758a6','frozen':'#ce5d45','no_memory':'#839298'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})


def pct(x):return f'{100*x:.1f}%'
def pp(x):return f'{100*x:.2f}'
def ci(x,scale=100,suffix='个百分点'):
    return f"{x['mean']*scale:.2f}（95% 区间 {x['low']*scale:.2f}～{x['high']*scale:.2f}）{suffix}"


def plots():
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
    xx=np.arange(3)
    for j,(support,col,label) in enumerate([(0,'#86999e','Dependency intact'),(1,'#087f8c','External support')]):
        vals=[A['conditions'][f'{m}__{support}__1__support_phase'] for m in ['q','replay','dyna']]
        for ax,key,scale in [(axes[0],'maintenance',100),(axes[1],'reward',1)]:
            y=np.array([v[key]['mean'] for v in vals])*scale
            lo=np.array([v[key]['low'] for v in vals])*scale
            hi=np.array([v[key]['high'] for v in vals])*scale
            ax.bar(xx+(j-.5)*.34,y,.32,color=col,label=label)
            ax.errorbar(xx+(j-.5)*.34,y,yerr=[y-lo,hi-y],fmt='none',color='#243d42',capsize=3)
            ax.set_xticks(xx,['Q','Replay +8','Dyna +8']);ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Preferred-tool maintenance (%)');axes[0].set_ylim(0,29)
    axes[1].set_ylabel('Reward / natural event');axes[1].set_ylim(0,.75)
    axes[0].set_title('A. Equal diagnostic evidence');axes[1].set_title('Performance and computation')
    axes[0].legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(FIG/'legacy.png',dpi=175);plt.close(fig)

    with np.load(MAIN/'core_raw.npz') as raw:
        fig,axes=plt.subplots(2,1,figsize=(11,6.5),sharex=True)
        for mode in LABEL:
            correct=np.concatenate([raw[f'{mode}__{s}__correct'] for s in range(6)])
            valid=np.concatenate([raw[f'{mode}__{s}__valid'] for s in range(6)])
            reward=np.concatenate([raw[f'{mode}__{s}__reward'] for s in range(6)])
            # 128-event windows, averages of individual accuracies, no smoothing across phases.
            acc=correct.reshape(96,128,32).sum(axis=1)/np.maximum(valid.reshape(96,128,32).sum(axis=1),1)
            re=reward.reshape(96,128,32).mean(axis=1)
            for ax,yy in [(axes[0],acc),(axes[1],re)]:
                ax.plot(np.arange(96)*128+64,yy.mean(axis=1),color=COLORS[mode],label=mode,
                        alpha=.85,linewidth=1.8 if mode=='online' else 1.3)
        for ax in axes:
            for boundary in range(2048,12288,2048):ax.axvline(boundary,color='#aab5b8',lw=.7,ls=':')
            ax.grid(axis='y',alpha=.15)
        axes[0].axhline(.5,color='#999',lw=.6,ls='--');axes[0].set_ylim(.05,1)
        axes[0].set_ylabel('Accuracy among valid answers');axes[1].set_ylabel('Reward / event')
        axes[0].set_title('B. Continuous experience, learned cue decoding')
        axes[0].legend(ncol=5,loc='upper center',fontsize=8,frameon=False)
        axes[1].set_xticks(np.arange(6)*2048+1024,['A','B','A return','A reversed','A restored','B return'])
        axes[1].set_xlabel('Six phases; 2,048 events each; no individual reset')
        fig.tight_layout();fig.savefig(FIG/'learning.png',dpi=175);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),sharey=True)
    for ax,ctx,title in [(axes[0],0,'Old context A after learning B'),(axes[1],1,'Current context B')]:
        modes=['online','repeat','replay','frozen','no_memory']
        vals=[B['evaluations'][f'{m}__1__eval{ctx}']['accuracy'] for m in modes]
        y=np.array([v['mean'] for v in vals])*100
        ax.bar(np.arange(5),y,color=[COLORS[m] for m in modes],width=.6)
        ax.errorbar(np.arange(5),y,yerr=[y-np.array([v['low'] for v in vals])*100,
                    np.array([v['high'] for v in vals])*100-y],fmt='none',color='#243d42',capsize=3)
        ax.axhline(50,color='#a45d52',ls='--',lw=1,label='Chance')
        ax.axhline(60,color='#658479',ls=':',lw=1,label='Known encoding: expected 60%')
        ax.set_xticks(np.arange(5),['online','repeat','replay','frozen','no memory'],rotation=22)
        ax.set_ylim(0,68);ax.set_title(title);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Controlled basic-channel accuracy (%)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'retention.png',dpi=175);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,3.8))
    keys=['retained_A_after_B','new_B_after_B','reversed_A']
    vals=[S['comparisons'][k] for k in keys];y=np.array([v['mean'] for v in vals])*100
    ax.errorbar(y,np.arange(3),xerr=[y-np.array([v['low'] for v in vals])*100,
                np.array([v['high'] for v in vals])*100-y],fmt='o',color='#087f8c',capsize=5,markersize=7)
    ax.axvline(0,color='#829293',lw=1);ax.set_yticks(np.arange(3),['Retained A after B','New B after B','Reversed A'])
    ax.invert_yaxis();ax.set_xlabel('Reservoir minus recent ring replay (percentage points; paired 95% interval)')
    ax.set_title('Supplement: same 256-event capacity, new independent cohort')
    ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(FIG/'supplement.png',dpi=175);plt.close(fig)


class Report:
    def __init__(self):self.h=[];self.md=[]
    def heading(self,text):self.h.append('<h2>'+html.escape(text)+'</h2>');self.md.append('## '+text+'\n')
    def p(self,text):self.h.append('<p>'+html.escape(text)+'</p>');self.md.append(text+'\n')
    def table(self,heads,rows):
        self.h.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>')
        self.md.extend(['| '+' | '.join(heads)+' |','| '+' | '.join(['---']*len(heads))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]+[''])
    def figure(self,name,caption):
        data=base64.b64encode((FIG/name).read_bytes()).decode()
        self.h.append('<figure><img src="data:image/png;base64,'+data+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
        self.md.append(f'![{caption}](figures/{name})\n')
    def code(self,text):self.h.append('<pre>'+html.escape(text)+'</pre>');self.md.append('```bash\n'+text+'\n```\n')
    def link(self,title,url):
        self.h.append('<p><a href="'+html.escape(url,quote=True)+'">'+html.escape(title)+'</a></p>')
        self.md.append(f'[{title}]({url})\n')


def audit():
    primary={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in M['source_sha256'].items()}
    secondary={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in SM['source_sha256'].items()}
    out={'primary_recorded_files_match':all(primary.values()),'primary_per_file':primary,
         'supplement_recorded_files_match':all(secondary.values()),'supplement_per_file':secondary,
         'primary_original_set_flag':M['source_unchanged'],
         'explanation':'Initial global set equality became false because importing the legacy source generated an inputs/__pycache__ file. All originally recorded protocol/source/input hashes still match. Original manifest preserved.'}
    (ROOT/'results/audit.json').write_text(json.dumps(out,indent=2)+'\n')
    assert all(primary.values()) and all(secondary.values())
    return out


def main():
    plots();au=audit();r=Report()
    r.heading('已实现什么，尚未解决什么')
    r.p('本轮已经实际运行一个小型混合持续学习核心：神经网络自行接收含噪线索、跨空白时刻保留信息、给出答案，并根据事后正确反馈学习；工具选择与维护由学习控制器完成。原来由程序直接把线索当作答案的环节已从新核心移除。')
    r.p('核心能力得到支持，旧技能保留没有达标。它能够学会当前情境，并在规则反转后重新学习；学习另一情境却会明显损害旧情境表现。补充的长历史采样改善有限，不能宣布解决持续学习。')
    r.table(['组成','实际实现'],[
        ['感知网络','17 维输入，24 维 tanh 循环隐状态，二元读出；1,033 个可训练参数'],
        ['动作控制','2 情境 × 25 次数状态 × 5 动作 = 250 个 Q 值；实际反馈更新 + 4 次 Dyna 备份'],
        ['后果模型','从观察到的状态转移和奖励学习；影响动作估值；不读取真实编码规则'],
        ['记忆与连续性','循环隐状态、神经参数、Q/模型、最多 256 条经验、上一反馈和内部随机状态均可保存'],
        ['训练范围','主测试每方法 32 个个体 × 12,288 事件；补充使用新的 32 个种子'],
        ['仍由设计者规定','状态接口、情境、工具、维护规则、奖励、学习目标、课程；没有自创目标或语言能力']])
    r.heading('A：为什么旧任务没有解除维护依赖？')
    r.p('64 对旧历史个体，比较自主探索与 8 次统一关键探测。起点来自旧检查点加共同阶段真实 Q 更新；不同于 WUYU_02 原干预的起点。环境中的免费支持只在零次数使用时能直接观察到。旧版宏动作代答仅用于这一诊断。')
    r.p('预定主对比：统一探测给普通 Q 学习带来的额外“维护下降”，为 '+ci(A['comparisons']['probe_effect_on_reduction'])+'。区间包含零，因此本次不支持“只要补足这些关键证据，普通 Q 就会明显解除依赖”。这也不能排除其他探测剂量、时序或探索策略。')
    rows=[]
    for m in ['q','replay','dyna']:
        before=A['conditions'][f'{m}__0__1__support_phase'];after=A['conditions'][f'{m}__1__1__support_phase']
        rows.append([m,pct(before['maintenance']['mean']),pct(after['maintenance']['mean']),
                     pp(before['maintenance']['mean']-after['maintenance']['mean'])+' 个百分点',
                     f"{after['reward']['mean']:.3f}",'1' if m=='q' else '1 + 8'])
    r.table(['方法（统一探测）','仍需维护','免费支持','维护下降','支持期回报','每真实事件备份'],rows)
    r.figure('legacy.png','同一证据条件，末 256 步结果。误差线以 64 个历史对重采样；模型/回放方法使用更多计算。')
    r.p('回放和 Dyna 的效果提示经验利用与信息传播值得继续研究。Dyna 也改善了仍需维护时的表现，因此不能把它相对 Q 的全部优势都归为识别免费支持。模型以完整共同历史形成统计，回放初始缓存为最近 512 条；模型压缩、采样分布、奖励估计及有效信息保留并非完全相同，不能据此声称模型方法本质优越。')
    rows=[]
    for m in ['q','replay','dyna']:
        sup=A['conditions'][f'{m}__1__1__support_phase'];wd=A['conditions'][f'{m}__1__1__withdrawal']
        change=A['conditions'][f'{m}__1__1__choice_change']
        rows.append([m,pct(change['mean']),f"{sup['reward']['mean']:.3f}",f"{wd['reward']['mean']:.3f}",pct(wd['maintenance']['mean'])])
    r.table(['方法','探测后其他状态选择改变','支持期回报','撤回后回报','撤回后维护率'],rows)
    r.p('其他状态是过去见过、在探测阶段没有直接环境交互的 24 个状态。无反馈选择变化仅说明内部更新传到了其他选择，不保证选择更好，更不是新任务泛化或因果理解的充分证据。后续自然互动包含再学习；撤回后的回报是恢复描述，不是瞬间推理能力。')
    r.heading('B：感知学习核心真的在使用记忆吗？')
    r.p('每个事件依次呈现情境/工具信息、含噪线索、空白延迟、作答。作答时不再呈现线索；无记忆对照在每个微时刻把隐状态清零。有效答案正确 +1、错误 −1；维护与无效使用 −0.20。由于任务只有两个答案，个体能够从自己的答案及正确反馈推回标签。这是事后监督，不是预先泄漏，也不是完全无监督。')
    r.p('验证集仅用于从 .005/.02/.05 选一个全局学习率，选中 '+str(SEL['selected_rate'])+'；主测试没有再调参。经历顺序为 A → B → A 返回 → A 规则反转 → A 恢复 → B 返回，跨阶段不重置个体。网络梯度只覆盖当前事件的 4 个微时刻，不能声称无限长时序学习。')
    r.figure('learning.png','自然行动轨迹：每点为 128 事件窗口的个体指标均值；工具选择会影响准确率，因此另做固定基本通道的感知评估。')
    r.table(['检查','结果','判定'],[
        ['初始 A 末段有效答案准确率',ci(B['stages']['online__0']['tail']['accuracy'],100,'%'),'通过 B1'],
        ['受控 A：在线 − 无记忆',ci(B['comparisons']['memory_contribution_A']),'通过 B2'],
        ['规则反转后受控 A：在线 − 冻结',ci(B['comparisons']['reversal_online_minus_frozen']),'通过 B3'],
        ['学 B 后旧 A：近期回放 − 重复当前',ci(B['comparisons']['replay_minus_repeat_retained_A_after_B']),'未通过 B4：区间下界未超过 3 个百分点'],
        ['检查点恢复','同一未来输入下行动、概率、隐状态、参数、缓存逐项相同','通过 B5']])
    r.p('自然行动中在线组初始 A 末段准确率约 88.1%，包含其选用高质量工具的影响。受控评估统一使用正确率 .60 的基本通道，已知编码时的期望最佳准确率为 60%。这两种准确率不应直接混用。冻结组未经元学习训练，它的失败不能排除所有固定权重网络的状态适应能力。')
    r.heading('主要缺口：学习新情境后，旧技能明显受损')
    r.figure('retention.png','固定感知条件，阶段 B 完成后的 A/B 评估。评估在个体副本中进行，不把结果写回持续个体。')
    r.p('在线组受控 A 准确率从 '+pct(B['evaluations']['online__0__eval0']['accuracy']['mean'])+' 降至 '+pct(B['evaluations']['online__1__eval0']['accuracy']['mean'])+'；变化为 '+ci(B['comparisons']['online_A_retention_change_after_B'])+'。能够重新学会 A，与同时保留 A、B 两套规则，是不同要求。')
    r.p('近期回放比重复当前经历的旧 A 表现高约 1.97 个百分点，但新 B 低约 2.76 个百分点。原工程判据未通过。两者都做每事件两次梯度计算，但维护等无新标签事件中，回放仍可使用旧标签，重复当前则是零有效梯度；实际非零学习次数不同，不能把小幅差异仅归因于记忆内容。')
    r.heading('补充：保存更长历史，是否就足够？')
    r.p('主实验后发现：B 阶段结束时，近期环形缓存中 A 经历占比为 0。于是先写 SUPPLEMENT_01.md，再在新的 32 个种子上比较相同 256 条容量的近期回放与 reservoir sampling。后者以流式均匀抽样保留全部历史中的样本，不读取情境来选择样本。补充没有改写主实验 B4。')
    r.figure('supplement.png','补充比较使用新的一组独立个体，所有差值在同一组内配对。不是拿新样本与旧主测试直接相减。')
    r.table(['补充比较：长历史 − 近期回放','差值','预设结果'],[
        ['学 B 后的旧 A',ci(S['comparisons']['retained_A_after_B']),'S1 未达标：区间下界 <3 个百分点'],
        ['学 B 后的新 B',ci(S['comparisons']['new_B_after_B']),'S2 在预设 −2 个百分点界内'],
        ['反转阶段的当前 A',ci(S['comparisons']['reversed_A']),'S3 在预设 −2 个百分点界内']])
    r.p('阶段 B 后，长历史缓存中 A 的平均占比为 '+pct(S['stages']['reservoir__1']['old_A_cache_fraction']['mean'])+'，但受控旧 A 的准确率仍只有 '+pct(S['stages']['reservoir__1']['eval0']['accuracy']['mean'])+'，近期缓存为 '+pct(S['stages']['ring__1']['eval0']['accuracy']['mean'])+'。保存旧样本有帮助的迹象，但未可靠达到预设改善幅度，也没有恢复好旧技能。')
    r.p('这支持进一步区分表示干扰、情境识别、回放时的状态匹配以及记忆调用问题；当前实验尚未把这些原因分开。长历史样本也可能包含过时规则，本轮未见超过预设界限的明确代价，不代表在其他变化频率下没有代价。')
    r.heading('计算量、样本与可复现性')
    rows=[]
    for mode in LABEL:
        c=B['costs'][mode]
        rows.append([LABEL[mode],str(c['events_per_individual']),f"{c['network_updates']['mean']:.0f}",
                     '61,440',f"{c['seconds']:.1f} 秒"])
    r.table(['主方法','每个体真实事件','平均非零神经更新','每个体控制备份','本环境批次耗时'],rows)
    r.p(f"主运行（含 A、选参、B、评估和数据保存）约 {M['seconds']:.1f} 秒；补充约 {SM['seconds']:.1f} 秒。均为当前云端 CPU 环境，Python 3.12.14 / NumPy 2.3.5。耗时不是跨硬件性能基准，也不包含设计、编程、验证与报告制作。独立样本单位为 A 的 64 个历史对、B 的 32 个个体、补充新的 32 个个体；不同方法和阶段是配对分支，不当成更多独立个体。")
    r.p('区间均为固定种子、2,000 次个体/历史对 bootstrap；多项区间是描述性分析，未做整体多重比较校正。主协议与补充假说均有运行前 SHA-256。协议是本地预先固定方案，不是第三方预注册。')
    r.p('主清单中的 source_unchanged 聚合字段为 false：旧环境模块首次导入生成了输入目录下的 Python 编译缓存，扩大了被追踪的文件集合。逐个复核主清单原有的协议、源码与输入，全部 SHA-256 匹配；补充清单记录的文件也全部匹配。保留原字段与审计说明，未改写原始运行记录。')
    r.heading('哪些事情还不能声称完成')
    r.p('没有训练 0.1B～0.5B 模型，没有实现自然语言、社会互动、开放式结构生成、自主定义学习目标，也没有证明数字生命、意识或内在关切。维护对象依然是预设外部工具，尚未构成维护学习系统自身组织的过程。六阶段实验只能检验有限的适应与遗忘，不能证明长期可塑性问题已经解决。')
    r.p('循环网络与表格控制器通过真实线索和反馈连接，属于混合原型；控制器仍依赖设计好的情境/次数状态表示。存储的旧隐状态可能随网络变化而失配。真实奖励是外部规定的二元正确性目标。以上都是当前边界，不应用“自主成长”措辞覆盖。')
    r.heading('由结果决定的下一步')
    r.p('优先研究“有历史样本，却不能在正确情境恢复旧能力”的原因。下一轮应固定感知与行动经验，先对比：原循环表示、明确情境条件化的表示、同容量的情境检索/门控；再分别干预旧样本、旧隐状态与上下文线索。必须同时检查旧规则保留、新规则学习与过时规则淘汰，不能只提高一个保留分数。')
    r.p('可组合能力与初级社会互动继续保留在路线中；当前最值得先补的是让多个已学规则在同一个个体内可区分、可调用、可修正。无需因此把“普通学习能否解释”重新设成生命门槛，也无需现在就扩大到零点几 B。')
    r.heading('如何继续运行交付的训练状态')
    r.code('python -B -m unittest discover -s tests -v\npython -B demo.py --context A --steps 256\npython -B demo.py --checkpoint results/supplement/checkpoint_reservoir.npz --context B --steps 256')
    r.p('完整包包括源代码、原始逐事件数据、真实训练检查点、协议、补充说明、图表和校验值。演示使用新的随机事件流，默认不覆盖检查点。模拟随程序停止而停止，没有自动部署的长期运行服务。')
    r.heading('研究依据与借鉴边界')
    r.link('Gershman, Markman & Otto：经验回放、模型与习惯控制的协作','https://gershmanlab.com/pubs/GershmanMarkmanOtto14.pdf')
    r.link('Dohare 等：连续学习中可塑性损失的实验研究','https://www.nature.com/articles/s41586-024-07711-7')
    r.link('Montévil 与 Mossio：约束闭合的组织理论','https://montevil.org/publications/articles/2015-mm-organisation-closure-constraints/')
    r.p('这些来源提供研究动机；本代码是独立简化实现，不宣称论文复现。组织理论尚未在本轮转化为数字生命判据。')
    css='''*{box-sizing:border-box}body{margin:0;background:#edf2f2;color:#17343c;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}header{background:#123e48;color:#fff;padding:52px max(6vw,22px) 42px}header small{letter-spacing:.18em;color:#9ed6d2}h1{font-size:clamp(27px,5vw,43px);line-height:1.3;max-width:950px;margin:15px auto}header p,header small{display:block;max-width:950px;margin:10px auto}main{max-width:1030px;margin:auto;background:#fff;padding:22px 42px 70px}h2{font-size:23px;line-height:1.5;margin:44px 0 16px;border-left:4px solid #087f8c;padding-left:13px}p{margin:15px 0}table{border-collapse:collapse;min-width:580px;width:100%;font-size:14px}th{background:#e7f2f1;text-align:left}td,th{padding:12px 14px;border-bottom:1px solid #d8e2e3;vertical-align:top}.table{overflow-x:auto;margin:24px 0}figure{margin:26px 0;background:#f8faf9;border:1px solid #e1e9e8;padding:12px}img{display:block;width:100%;height:auto}figcaption{font-size:13px;line-height:1.65;color:#526b70;margin:8px 5px}pre{background:#153b44;color:#e7f5f2;padding:20px;overflow-x:auto;font-size:13px;line-height:1.9;border-radius:6px}a{color:#087f8c;overflow-wrap:anywhere}footer{color:#5d7478;text-align:center;font-size:13px;padding:26px}@media(max-width:620px){main{padding:12px 19px 45px}header{padding-top:30px;padding-bottom:28px}h2{font-size:20px}body{font-size:15px}figure{padding:4px;margin-left:-8px;margin-right:-8px}}'''
    body='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_03 实验报告</title><style>'+css+'</style></head><body><header><small>WUYU / EXPERIMENT 03 · 2026-10-10</small><h1>感知学习已接通<br>跨情境保留仍是缺口</h1><p>实际代码 · 真实训练状态 · 原始数据 · 保留未达标结果</p></header><main>'+''.join(r.h)+'</main><footer>WUYU_03 · CPU 机制实验 · 不构成生命或意识证明</footer></body></html>'
    (ROOT/'REPORT.html').write_text(body)
    (ROOT/'REPORT.md').write_text('# WUYU_03：感知学习已接通，跨情境保留仍是缺口\n\n'+'\n'.join(r.md))
    validation='''# 验证与审计

主实验前六项检查全部通过：四微时刻 BPTT 的中心差分梯度核验；环境线索与回答分离；无记忆对照不能读先前线索；标签仅从回答后的正确反馈推回；后果模型只写入真实见过的键；完整状态保存恢复后相同未来输入逐项一致。

补充前新增两项并连同前六项全部通过：reservoir 容量和过去样本约束；超过容量后抽样随机状态的保存恢复连续性。日志保存在 results/main/validation.log 与 results/supplement_validation.log。

主清单 source_unchanged=false 是文件集合检查的误报：导入 inputs/legacy_world_source.py 后生成了 inputs/__pycache__/legacy_world_source.cpython-312.pyc，该输入目录下的新缓存被宽泛的集合比较收录。主清单里原先记录的所有协议、源码、输入均与现在的 SHA-256 匹配。没有改动其源代码后重用原结果，也没有更改原清单字段。逐文件复核结果见 results/audit.json。缓存作为审计材料保留，不是训练数据。

补充代码和报告生成器是在主实验后新增的文件；主文件未修改。补充清单中记录的文件也逐项匹配。补充结果保持独立，不改写主 B4。未进行按主测试结果调整学习率、初始化或阈值的重跑。

原始检查点含网络、控制器、模型、缓存、持续状态和内部随机状态；复现未来还要求提供相同未来外生事件。demo.py 提供新的事件流，故演示指标不当作总体性能证据。
'''
    (ROOT/'VALIDATION.md').write_text(validation)
    print(json.dumps({'report_html_bytes':len(body.encode()),'audit_original_files_match':au['primary_recorded_files_match'],
                      'supplement_files_match':au['supplement_recorded_files_match']},indent=2))


if __name__=='__main__':main()
