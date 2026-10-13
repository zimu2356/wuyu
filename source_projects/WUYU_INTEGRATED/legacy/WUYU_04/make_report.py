"""Render a mobile-readable, self-contained research report from saved results."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
D=json.loads((R/'diagnostic.json').read_text());H=json.loads((R/'hidden.json').read_text())
M=json.loads((R/'manifest.json').read_text());S=json.loads((R/'selection.json').read_text())
raw=np.load(R/'hidden_raw.npz',allow_pickle=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#fbfcff','axes.facecolor':'#fbfcff'})


def pct(x):return f'{100*x:.1f}%'
def ci(x):return f"{pct(x['mean'])} [{pct(x['low'])}, {pct(x['high'])}]"
def pp(x):return f"{100*x['mean']:+.2f} 个百分点 [{100*x['low']:+.2f}, {100*x['high']:+.2f}]"
def numci(x):return f"{x['mean']:+.5f} [{x['low']:+.5f}, {x['high']:+.5f}]"


names={'joint':'原网交错训练','sequential':'原网顺序训练','dense':'每微步补情境','gated':'两个独立 RNN',
       'repeat':'重复当前样本','stale':'回放旧隐藏状态','burnin':'回放＋重算前序状态','count':'情境计数表',
       'single':'单模块持续学习','oracle':'已知真情境上界','passive':'隐藏情境·被动','active':'隐藏情境·主动'}
keys=['joint','sequential','dense','gated','repeat','stale','burnin','count']
labels=['Interleaved RNN','Sequential RNN','Dense context','Two gated RNNs','Repeat current',
        'Replay / stored state','Replay / 8-event burn-in','Context count table']
fig,ax=plt.subplots(figsize=(10.3,6.3),layout='constrained')
yy=np.arange(len(keys))
for j,color,label in [(0,'#2563eb','Old rule A'),(1,'#16a085','New rule B')]:
    stats=[D['methods'][k]['final'][j] for k in keys]
    vals=np.array([s['mean'] for s in stats]);err=np.array([[s['mean']-s['low'] for s in stats],[s['high']-s['mean'] for s in stats]])
    ax.barh(yy+(j-.5)*.34,vals,height=.31,color=color,label=label,xerr=err,error_kw={'lw':1,'capsize':2})
ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Frozen-weight evaluation accuracy (32 individuals; 95% bootstrap CI)')
ax.axvline(.9,ls=':',color='#687488',lw=1);ax.set_title('A / Same decoding problem, different retention mechanisms',loc='left',pad=16)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False);fig.savefig(F/'diagnostic.png',dpi=170);plt.close(fig)

colors={'single':'#9b678c','oracle':'#65718a','passive':'#2563eb','active':'#0b9e85'}
fig,(ax,bx)=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1.2]},layout='constrained')
for mode,label in [('single','Single module'),('oracle','Oracle context'),('passive','Passive inference'),('active','Active inference')]:
    a=raw[mode+'__correct'].reshape(72,64,32).mean(axis=(1,2))
    ax.plot(np.arange(72)*64+32,a,color=colors[mode],lw=1.8,label=label,ls='--' if mode=='oracle' else '-')
for p in range(1,9):
    ax.axvline(p*512,color='#dbe1eb',lw=.8);bx.axvline(p*512,color='#dbe1eb',lw=.8)
for axis in (ax,bx):axis.axvspan(7*512,8*512,color='#f6e5cf',alpha=.7)
ceiling=np.array([.9]*7+[.6,.9]);ax.stairs(ceiling,np.arange(10)*512,color='#888',ls=':',label='Expected ceiling')
q=raw['active__query'].reshape(72,64,32).mean(axis=(1,2))
bx.fill_between(np.arange(72)*64+32,q,color=colors['active'],alpha=.25)
bx.plot(np.arange(72)*64+32,q,color=colors['active'],lw=1.5)
ax.set_ylim(.3,1);bx.set_ylim(0,.65);ax.set_ylabel('Accuracy / 64-event block');bx.set_ylabel('Probe fraction')
ax.legend(loc='lower left',ncol=2,frameon=False);ax.set_title('B / Hidden rule changes, a novel rule, and degraded observations',loc='left',pad=16)
bx.set_xticks(np.arange(9)*512+256,['A','B','A return','C new','A return','B return','C return','A noisy','A restored'])
bx.set_xlim(0,4608);bx.set_xlabel('Hidden phase: 512 ordinary events per segment; apprenticeship omitted')
fig.savefig(F/'hidden_learning.png',dpi=170);plt.close(fig)


def table(headers,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def mtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def img(name,caption):
    data=base64.b64encode((F/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

rowsA=[[names[k],ci(D['methods'][k]['final'][0]),ci(D['methods'][k]['final'][1])] for k in keys]
rowsB=[[names[k],ci(H['methods'][k]['old_return_first32']),ci(H['methods'][k]['new_rule_last128']),
        f"{H['methods'][k]['net_reward']['mean']:.5f}"] for k in ('single','oracle','passive','active')]
costA=[]
for k in keys[:-1]:
    c=D['methods'][k]['cost'];costA.append([names[k],c['parameters'],f"{c['numeric_bytes_per_individual']/1024:.1f} KiB",
        c['forward_micro_batches'],c['backward_micro_batches'],f"{c['seconds']:.2f} s"])
costB=[]
for k in ('single','oracle','passive','active'):
    c=H['methods'][k]['cost'];costB.append([names[k],c['parameters'],f"{np.mean(c['probes_per_individual']):.2f}",
        f"{np.mean(c['update_calls_per_individual']):.2f}",c['agent_numeric_bytes_per_individual'],f"{c['seconds']:.2f} s"])
gatelabels={
 'A1_joint':'联合训练同时学好 A/B（两项下界 >80%）',
 'A2_gated':'双模块保留 A/B（两项下界 >80%）',
 'A3_burnin':'重算回放改善旧 A 且不明显损害 B',
 'B1_passive_old_return':'被动：旧情境返回初期下界 >80%',
 'B2_passive_new_rule':'被动：新规则末期下界 >80% 且旧权重不变',
 'B1_active_old_return':'主动：旧情境返回初期下界 >80%',
 'B2_active_new_rule':'主动：新规则末期下界 >80% 且旧权重不变',
 'B3_active_switch':'主动：切换初期改善下界 >2 个百分点',
 'B3_active_net':'主动：全程净收益改善下界 >0',
 'exact_runtime_restore':'中途保存后精确复现未来事件与完整状态'}
gaterows=[[gatelabels[k],'通过' if v else '未通过'] for k,v in {**D['gates'],**H['gates']}.items()]
active=H['methods']['active'];passive=H['methods']['passive']
mean_probe=np.mean(active['cost']['probes_per_individual'])
comparison=H['comparisons'];burn=D['comparisons']['burnin_minus_stale'][0]

sections=[];md=[]
def add(title,body,text):
    sections.append(f'<section><h2>{title}</h2>{body}</section>');md.append('## '+title+'\n\n'+text)

add('这次真正推进了什么',
    '<p><strong>已经做出可以保有旧规则、根据证据选择规则、再学习一种新规则的有限核心。</strong>同时找到了旧 RNN 回放流程中一个值得继续采用的改动：用当前参数重新走过样本之前的短段经历，改善回放状态的一致性。</p>'
    '<p>这项进展属于未来持续个体所需的一个机制：经历新事情后，旧能力仍能被重新调用。它还没有构成你想要的完整“物欲”，也没有提供意识、人格、生命或成为人的证据。</p>'
    '<div class="callout">关键负结果：主动探查切换得更快，但全程付出的成本超过了新增收益；观测可靠率下降时，它仍会混淆“规则变了”和“信息变差了”。</div>',
    '已完成有限规则保留、证据选择与新规则学习核心，以及原 RNN 回放诊断。主动探查更快但本轮净收益更差；可靠率退化仍造成误判。没有完整数字生命、意识或人格的证据。')

add('实验分成两层，不能混在一起读',
    '<p><b>A 层</b>逐字复用 WUYU_03 的 1033 参数 RNN，将动作固定、线索可靠率设为 0.9，隔离表示和学习问题。<b>B 层</b>是一个独立的简化规则核心：8 种线索、二元答案、两个学过后冻结的模块，加一个可塑的新规则槽。B 层没有整合旧版的身体与工具控制。</p>'
    '<p>A 使用 8 个验证个体和 32 个正式个体；B 使用另 32 个正式个体。相同个体跨方法共享普通事件与噪声。学习率在验证集上选出后固定；原网和双模块均选中 0.01，三档验证准确率各自相同，依协议取最小值。B 学习率事前固定为 0.2。</p>'
    '<p>区间是以个体为单位进行 4000 次配对 bootstrap 的 95% 百分位区间，不把时间步当作独立样本；多个比较未做多重校正。</p>',
    'A：原 RNN、可靠率0.9、固定动作。B：独立三模块简化核心。A 验证8个体/正式32个体，B另32个体。区间为个体级配对 bootstrap 4000次，未做多重比较校正。两个层次的成绩不能混为同一完整系统。')

add('旧核心：重算回放确实有帮助',
    table(['方法','学完 B 后，旧 A','学完 B 后，新 B'],rowsA)+img('diagnostic.png','同一批 32 个体；误差线是 95% 区间。0.9 是此任务的期望准确率上限，有限样本均值可能略高。')+
    f'<p>旧隐藏状态回放保留 A 的准确率为 {ci(D["methods"]["stale"]["final"][0])}；用当前参数重算最多 8 个前序事件后为 {ci(D["methods"]["burnin"]["final"][0])}。配对差为 <strong>{pp(burn)}</strong>，B 的准确率差为零。这次达到事前规定的收益门槛。</p>'
    '<p>这是有限 burn-in 方案的实证收益。它同时改变了回放初始状态的时效性和有限历史近似，不能据此断言“所有遗忘都只由陈旧状态引起”。回放训练也多用了梯度计算，已用“重复当前样本”作额外计算对照。</p>'
    '<p>把情境输入重复送入每一个微步，仍没有挽救顺序原网；它不是简单的“再提醒一次就行”。双模块把两套规则分开学习，接近噪声上限，但参数和执行工作量也加倍。</p>'
    '<p><b>联合训练门槛没有通过。</b>原网交错训练的 A/B 均值约 81%，但两项区间下界都低于 80%。严格逐次交错还可能让网络利用时序规律，而评估是分情境连续输入，存在分布差异。因此不能把这次失败解释为容量不够。同样 1033 参数的重算回放条件已经在无参数更新的 A/B 测试中达到约 90%，没有证据要求现在先扩大模型。</p>'
    '<p>最终单次反事实探针中，仅交换情境而保留线索和当前隐藏状态，顺序原网答案翻转率为 0；重算回放为 62.5%；双模块为 100%。单次探针与连续 256 步评估不同：前者不允许状态经连续输入重新稳定，不能据其不足 100% 就断言规则未被存储。</p>',
    mtable(['方法','最终旧A','最终新B'],rowsA)+f'\n\n重算回放对旧A的配对改善：{pp(burn)}，对新B差为零。联合训练门槛失败；不能据此认定容量不足。情境输入重复无效；双模块有效但参数翻倍。单次情境翻转率：原网0、重算回放62.5%、双模块100%。重算8个前序事件是有限近似，不是完整历史恢复；本结果未在旧身体/工具环境中重测。')

add('隐藏情境：先学过，再凭证据找回来',
    '<p>学徒期 A/B 各 1024 次，允许提供情境标签；两个模块的答案均从作答后的反馈学得。进入隐藏期后，代理不再收到规则编号，依次经历 A → B → A → C → A → B → C → A可靠率退化 → A恢复，每段 512 次。C 在学徒期从未出现。</p>'
    '<p>被动核心从普通反馈更新三种情境的信念；主动核心在信念较不确定时，选择模块意见最不一致的线索，付费请求一次带独立噪声的标定样本。该样本可能与当前线索相同，但不直接提供当前事件的目标，也不提供情境编号。</p>'+
    table(['方法','旧 A/B 返回的前32步','首次新 C 的末128步','全隐藏期净收益/步'],rowsB)+
    '<p>“旧返回”是阶段 2、4、5（从0起算）的平均，不能解释为每一次返回都超过 80%。主动条件首次 C 前32步只有 59.9%，末128步才达到 89.6%；它需要获得证据并学习，并非一到新世界就知道规则。</p>'
    '<p>被动的旧返回区间下界是 80.05%，仅略过门槛；主动是 83.79%。所有三模块条件的 A/B 权重最终都与学徒期后逐位相同。这来自明确的冻结机制，不是自行发现的保护策略。冻结权重检查证明存储保持，隐藏期行为才检验是否能访问这些记忆。</p>'+img('hidden_learning.png','曲线为每64事件的32个体平均，未显示区间；表格提供关键区间。棕色区域把标签可靠率从0.9降到0.6，模拟观测关系退化。'),
    mtable(['方法','旧返回前32步','新C末128步','净收益/步'],rowsB)+'\n\n学徒期真实情境可见，隐藏期不提供情境标签。C为新规则。主动首次C前32步59.9%，末128步89.6%。旧A/B权重完全不变是冻结设计的结果；隐藏行为另行测量访问能力。被动旧返回区间下界80.05%只是勉强过门槛。')

add('主动探查：快一点，不一定划算',
    f'<p>六个规则切换阶段前 32 步，主动相对被动提高 <strong>{pp(comparison["active_minus_passive_switch_first32"])}</strong>。但每个体平均多请求了 <strong>{mean_probe:.2f}</strong> 次标定，占 4608 个普通事件的 {pct(active["probe_fraction"]["mean"])}。</p>'
    f'<p>每次标定事前定价 0.10；答对奖励 +1、答错 −1。主动相对被动的全程净收益差为 <strong>{numci(comparison["active_minus_passive_net"])}</strong>，整个区间都低于零。不能把准确率改善直接称为自主探查策略已经更优。</p>'
    '<p>可靠率退化段，主动条件在约 47.4% 的事件中继续探查，但准确率只有 54.2%；真实情境路由上界为 59.6%。这里的“传感退化”仅通过增大标签噪声模拟，没有独立物理传感器，也没有因果故障定位实验。</p>'
    '<p>全部阶段结束后，无更新枚举八种线索：主动核心旧 A/B 映射仍为 100%，可塑 C 模块为 97.3%；被动 C 为 100%。因此“新规则学会过”仍不等于“之后始终完整保留”。本轮没有逐阶段 C 权重检查点，不能进一步定位是哪一段导致了最终偏差。</p>',
    f'切换前32步配对改善 {pp(comparison["active_minus_passive_switch_first32"])}；平均额外标定 {mean_probe:.2f} 次/个体。净收益差 {numci(comparison["active_minus_passive_net"])}，负结果。可靠率退化时主动探查47.4%、准确率54.2%，oracle59.6%。最终主动C映射97.3%、被动100%；无法据本轮数据定位可塑C发生偏差的确切阶段。')

add('花了什么资源',
    '<p>A 层下表的微步数是对整批 32 个体执行的网络前向/反向微步；每个体经历相同次数。双模块实现对两个模块都执行运算，其中一个梯度为零。数字数组内存含权重、状态、缓冲；不含 Python 对象、临时张量或输出日志。</p>'+table(['A 条件','参数/个体','数字数组/个体','前向微步','反向微步','该批运行'],costA)+
    '<p>为了只改变回放状态算法，旧状态和重算条件都分配了相同的前序历史缓存；旧状态版本其实可以删掉未使用的历史，降到约 0.19 MiB/个体。因此表中相等的内存是本次对照实现，不代表两者最低内存相同。重算条件的前向微步是旧状态回放的 3.67 倍。</p>'+table(['B 条件','权重/个体','额外标定/个体','更新调用/个体','代理数字字节/个体','该批运行'],costB)+
    '<p>B 每个体另有相同的 4608 次普通事件，学徒期 2048 次未算入交互列。更新调用包含权重为零的责任更新，不等于非零参数变化。代理数组包含一份冻结权重审计副本；世界数组另为36字节/个体。主动实现还在某个体请求探查时对整批计算概率，因此不能仅按请求次数推断计算量。</p>'
    f'<p>整个正式流程（含验证集、评估、保存和统计）记录耗时 {M["seconds"]:.2f} 秒，Python {M["python"]}、NumPy {M["numpy"]}，CPU 运算。这里的时间属于本次运行环境，不能当成手机或其他机器的速度承诺。</p>',
    mtable(['A条件','参数','数字数组','前向微步','反向微步','时间'],costA)+'\n\n'+mtable(['B条件','权重','额外标定','更新调用','数字字节','时间'],costB)+f'\n\n整个实验记录 {M["seconds"]:.2f} 秒。旧状态回放为控制变量也保存前序历史，最低必要内存可更小；重算前向工作量为其3.67倍。数组字节不是整个进程内存。')

add('事前门槛与独立核验',
    table(['事前门槛','结果'],gaterows)+
    '<p>6 项机制测试通过。又从原始数组独立重算了主要均值，确认四种 B 方法的普通线索与目标完全配对、成本公式一致、后验归一化，以及冻结权重精确保持。</p>'
    '<p>C 学习中途保存后，重新加载出的 128 步输出与原始运行记录逐项相同；另一次保存恢复测试比较了后续输出、全部数值状态和世界两条随机流。演示程序已实际从第3712事件运行至第3840事件。</p>'
    '<p>实验启动时冻结的每个源文件哈希都未改变；旧 RNN 复制件与原文件字节相同。报告、demo 和独立审计脚本是结果生成后新增的交付工具。标准输出文件只保留了开头两行，未把缺失输出伪造补齐；完整结果以 NPZ、JSON 和校验文件为准。</p>',
    mtable(['事前门槛','结果'],gaterows)+'\n\n6项机制测试通过；原始数组独立重算通过；中途恢复128步与原始记录逐项一致，完整数值状态及环境RNG恢复测试通过；初始冻结文件哈希未变。stdout捕获不完整，原始NPZ/JSON完整。')

add('现在还不能声称什么',
    '<p>不能说已经形成通用模型基底、完整物欲个体或数字生命。这里既没有语言与真实多模态感知，也没有长期身体、自维护动机、社会学习、能力变化的自我模型，运行也是有限仿真。</p>'
    '<p>三模块划分、两个冻结槽、一个可塑槽、真实标签学徒期、二元反馈反推标签、固定切换先验、探查触发策略，都是设计者提供的结构。代理学了具体答案和在线情境信念，但没有自行发现任意数量的新情境，也没有学会如何定价探查。第四套不同规则将超出这轮的预留设计，尚未测试。</p>'
    '<p>A 的新回放机制尚未接回 WUYU_03 的完整工具世界；B 使用不同、更加透明的规则网络。因此这一版提供的是两个可继续整合的机制结果，不是声称所有旧功能已经在一个新个体里共同运行。</p>',
    '未证明通用基底、数字生命、意识或人格；未整合旧工具世界。已知情境学徒期、冻结/可塑槽、固定hazard与探查策略都是工程先验。第四新规则、任务数量发现、自我模型、身体与动机闭环均未检验。')

add('据这些结果，下一步应该怎样走',
    '<p><strong>优先建立“观测可靠性”与“规则身份”的分离估计，再把通过的记忆机制接回持续环境。</strong>当前失败已经给出了具体方向，无须先把模型放大。</p>'
    '<ol><li>保留原始旧系统，单独接入当前参数的前序状态重算，在同一个身体/工具环境、相同交互预算下验证收益能否迁移。</li><li>在 B 核心中分别表示规则、观测可靠率和自身执行能力；让世界分别改变它们，并提供可区分原因的诊断动作。仅观察一次错误时三者可能不可辨，必须设计能产生不同证据的干预。</li><li>比较探查的预计价值与真实成本，让系统在继续试探、等待、换传感通道、修正模型之间作选择；不能只把“更不确定”翻译为“继续花钱探查”。</li><li>之后再取消预留第三槽，检验新增、合并、冻结与淘汰记忆；用未见过的第四、第五规则和组合任务验证，避免凭固定三槽成绩宣称开放式成长。</li></ol>'
    '<p>你希望的“物欲”最终需要这些机制与持续的身体、环境、动机和经历连在一起。本轮已经把“知道旧规则在哪里、何时该用它”的一部分变成了可运行、可复现的实验。</p>',
    '下一步优先：1. 将重算回放接回旧工具世界；2. 分开估计规则身份、观测可靠性、自身执行能力，并设计可区分原因的诊断动作；3. 按预期价值与成本选择探查；4. 再取消预留槽，测试第四/第五规则与记忆合并淘汰。不要凭固定三槽成绩宣称开放式成长。')

sources=[('R2D2 作者页：循环网络回放中的表示漂移与状态陈旧','https://willdabney.com/publication/r2d2/'),
         ('Masse 等：context-dependent gating 与稳定化（2018）','https://arxiv.org/html/1802.01569v2'),
         ('Heald 等：情境推断与技能记忆表达（2021）','https://www.nature.com/articles/s41586-021-04129-3')]
add('可复现文件与相关研究',
    '<p>压缩包包含协议、完整源码、6项测试、参数选择、逐步原始数组、全状态检查点、图表、报告及 SHA256 清单。用 <code>python3 demo.py --steps 128</code> 可继续运行真实保存状态；<code>README.md</code> 给出完整复现命令。默认演示环境有终点，不是自动后台常驻程序。</p>'
    '<ul>'+''.join(f'<li><a href="{url}">{name}</a></li>' for name,url in sources)+'</ul>'
    '<p>文献用于机制启发，不是这些方法的严格复现。R2D2 只据作者页摘要；本轮结果由交付代码与数据支持。</p>',
    '源码、测试、原始数据、检查点、图表与SHA256清单均在压缩包。\n\n'+'\n'.join(f'- [{name}]({url})' for name,url in sources)+'\n\n这些是机制启发，不是原论文复现；R2D2依据作者摘要。')

style='''*{box-sizing:border-box}body{margin:0;background:#edf1f7;color:#1d2c42;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.85}main{max-width:1000px;margin:0 auto;padding:32px 30px 64px;background:#fff}header{border-bottom:3px solid #185bc2;padding-bottom:28px;margin-bottom:26px}.eyebrow{color:#356ba6;font-weight:700;letter-spacing:.12em;font-size:13px}h1{font-size:clamp(28px,5vw,42px);line-height:1.25;margin:15px 0}h2{font-size:23px;line-height:1.4;margin:0 0 18px;color:#163d70}section{padding:22px 0;border-bottom:1px solid #e3eaf3}p{margin:12px 0}strong{color:#164e97}.sub{color:#60728a}.callout{background:#fff3df;border-left:4px solid #cf922a;padding:15px 18px;margin:20px 0}.tablewrap{overflow-x:auto;margin:20px 0;border:1px solid #dbe4f0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}th,td{text-align:left;padding:12px 13px;border-bottom:1px solid #e5eaf2;vertical-align:top}th{background:#eaf1fb;color:#244d7b}tr:last-child td{border:0}td:first-child{font-weight:600;min-width:130px}figure{margin:24px 0}img{display:block;max-width:100%;height:auto;border:1px solid #e2e8f2;border-radius:8px}figcaption{font-size:13px;color:#66778d;margin-top:9px}li{margin:10px 0}a{color:#1b63bb;overflow-wrap:anywhere}code{background:#eef3fa;padding:2px 5px;border-radius:4px;font-size:13px;overflow-wrap:anywhere}.tag{display:inline-block;border-radius:20px;padding:4px 11px;margin:4px 4px 0 0;background:#edf3fc;color:#335c8e;font-size:12px}footer{padding:20px 0;color:#6c7a8d;font-size:12px}@media(max-width:600px){main{padding:24px 17px 45px}body{font-size:15px}h2{font-size:21px}table{font-size:12px}th,td{padding:9px}section{padding:20px 0}ol{padding-left:24px}}@media print{main{max-width:none;padding:0}section{break-inside:auto}.tablewrap,figure{break-inside:avoid}body{background:white}a{color:inherit}}'''
head='<header><div class="eyebrow">WUYU · EXPERIMENT 04 · 2026-10-10</div><h1>记得旧规则，<br>也要知道何时使用它。</h1><p class="sub">规则共存 / 隐藏情境 / 主动取证 / 资源成本</p><span class="tag">已执行</span><span class="tag">原始数据可复现</span><span class="tag">包含负结果</span></header>'
report='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_04 — 规则保留与证据切换</title><style>'+style+'</style></head><body><main>'+head+''.join(sections)+'<footer>本报告所有实验数值由 results/*.json 与原始数组生成。没有使用大模型生成行为样本。</footer></main></body></html>'
(ROOT/'REPORT.html').write_text(report,encoding='utf-8')
(ROOT/'REPORT.md').write_text('# WUYU_04 — 规则保留与证据切换\n\n实验报告 · 2026-10-10\n\n'+'\n\n'.join(md)+'\n',encoding='utf-8')
print('report and 2 figures generated')
