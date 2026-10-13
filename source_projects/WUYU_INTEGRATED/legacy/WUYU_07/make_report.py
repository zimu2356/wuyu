import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,base64,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES={'full':'混合控制器','no_skill':'同知识，无技能执行器','frozen':'混合体，冻结个体学习','fixed_value':'混合体，固定需要权重',
       'plain_rnn':'普通循环策略参照','neural_student':'直接神经控制','neural_student_frozen':'神经控制，冻结个体学习'}
EN={'full':'Hybrid','no_skill':'Hybrid / no skill executor','frozen':'Hybrid / frozen learning','fixed_value':'Hybrid / fixed need weights',
    'plain_rnn':'Recurrent reference','neural_student':'Neural direct control','neural_student_frozen':'Neural / frozen learning'}


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def percent(s):return f"{s['mean']*100:.1f}% [{s['ci'][0]*100:.1f}, {s['ci'][1]*100:.1f}]"
def image(name,caption):return '<figure><img src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'" alt="'+html.escape(caption)+'"><figcaption>'+caption+'</figcaption></figure>'


if __name__=='__main__':
    s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());p=s['probes'];diff=s['paired_differences']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f4f7f6','axes.facecolor':'#f4f7f6','savefig.facecolor':'#f4f7f6'})
    modes=list(NAMES);vals=np.array([s['main'][m]['survival']['mean'] for m in modes]);ci=np.array([s['main'][m]['survival']['ci'] for m in modes])
    fig,ax=plt.subplots(figsize=(10,5.2));y=np.arange(len(modes));colors=['#318675' if m=='neural_student' else '#7a9997' for m in modes]
    ax.barh(y,vals,color=colors);ax.errorbar(vals,y,xerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),fmt='none',color='#213c3c',capsize=3)
    ax.set_yticks(y,[EN[m] for m in modes]);ax.invert_yaxis();ax.set_xlim(0,1.15);ax.set_xlabel('Fraction of 1,536-event life active')
    for j,v in enumerate(vals):ax.text(max(v,ci[j,1])+.018,j,f'{v:.1%}',va='center')
    ax.set_title('Formal continuous lives: 3 initializations x 32 worlds',loc='left',fontweight='bold',pad=14);fig.tight_layout();fig.savefig(FIG/'lives.png',dpi=160);plt.close(fig)
    labels=['Independent operation','New dependencies','Need revaluation','Changed operation']
    before=[p[x]['mean'] for x in ['independent_birth_success','composite_birth_success','revalue_fixed','change_before_success']]
    after=[p[x]['mean'] for x in ['independent_mature_success','composite_mature_success','revalue_adaptive','change_after_success']]
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(4)
    ax.bar(x-.18,before,.36,label='Before / fixed valuation',color='#adbab7');ax.bar(x+.18,after,.36,label='After / adaptive valuation',color='#318675')
    ax.set_xticks(x,labels);ax.set_ylim(0,1.22);ax.set_ylabel('Probe success / correct goal fraction');ax.legend(frameon=False,loc='upper left',ncol=2)
    for i in range(4):
        ax.text(i-.18,before[i]+.025,f'{before[i]:.1%}',ha='center',fontsize=9);ax.text(i+.18,after[i]+.025,f'{after[i]:.1%}',ha='center',fontsize=9)
    ax.set_title('Hybrid mechanism probes; parameters frozen during assessment',loc='left',fontweight='bold',pad=14);fig.tight_layout();fig.savefig(FIG/'mechanisms.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.2));labels=['Experienced','At birth','Same hidden state / memory reset']
    keys=['student_mature_success','student_birth_success','student_memory_reset_success'];values=[p[k]['mean'] for k in keys]
    ax.barh(range(3),values,color=['#318675','#a8b7b4','#bd916b']);ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,1.14)
    for i,v in enumerate(values):ax.text(v+.02,i,f'{v:.1%}',va='center')
    ax.set_xlabel('First body-resource delivery matches current need within 40 steps')
    ax.set_title('Neural controller: no device-goal instruction',loc='left',fontweight='bold',pad=14);fig.tight_layout();fig.savefig(FIG/'neural_probes.png',dpi=160);plt.close(fig)
    with np.load(OUT/'raw_neural_student_0.npz',allow_pickle=False) as z:
        raw={k:z[k] for k in ['body','skills','position','action','product','phase']}
    t=np.arange(1,1537);fig,axs=plt.subplots(2,1,figsize=(10,5.2),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    axs[0].plot(t,raw['body'][:,0,0],label='Energy',lw=1.1,color='#318675');axs[0].plot(t,raw['body'][:,0,1],label='Integrity',lw=1.1,color='#bd916b')
    axs[0].set_ylim(0,1.06);axs[0].legend(frameon=False,ncol=2);axs[0].set_title('One recorded neural life: initialization 0, world 740000',loc='left',fontweight='bold')
    axs[1].plot(t,raw['skills'][:,0],color='#365967');axs[1].set_ylabel('Valid motor options');axs[1].set_xlabel('Event');axs[1].set_ylim(-.2,6.4)
    for ax in axs:
        for boundary in [384,768,1152]:ax.axvline(boundary,color='#6e827d',ls=':',lw=1)
    fig.tight_layout();fig.savefig(FIG/'recorded_life.png',dpi=160);plt.close(fig)
    diagnostic=json.loads((OUT/'student_diagnostic.json').read_text())
    daudit=json.loads((OUT/'diagnostic_audit.json').read_text())
    diagnostic_rows=[]
    for phase in [0,1]:
        for severity in ['moderate','large']:
            diagnostic_rows.append(['独立设备' if phase==0 else '新依赖','温和缺口' if severity=='moderate' else '较大缺口',*[percent(diagnostic[f'{mode}_phase{phase}_{severity}']) for mode in ['neural','neural_greedy','planner_same_memory']]])
    rows=[]
    for m in modes:
        row=s['main'][m];rows.append([NAMES[m],percent(row['survival']),f"{row['body_area']['mean']:.3f}",'/'.join(f'{v:.1%}' for v in row['survival']['by_initialization'])])
    gates=[[k,g['name'],'通过' if g['passed'] else '未通过'] for k,g in s['gates'].items()]
    evidence=[['独立操作',percent(p['independent_birth_success']),percent(p['independent_mature_success'])],
        ['新依赖组合',percent(p['composite_birth_success']),percent(p['composite_mature_success'])],
        ['当前需要对应的终极目标',percent(p['revalue_fixed']),percent(p['revalue_adaptive'])],
        ['改变后的设备操作',percent(p['change_before_success']),percent(p['change_after_success'])],
        ['未改变操作的保留',percent(p['retention_before']),percent(p['retention_after'])]]
    budgets=[]
    for family in ['rnn','student','student_final']:
        for i in range(3):
            r=json.loads((OUT/f'training_{family}_{i}.json').read_text());best=r['selected']
            budgets.append([{'rnn':'循环参照','student':'学生初轮','student_final':'学生自身轨迹修正'}[family],i,best['file'],f"{best['survival']:.1%}",f"{r['seconds']:.1f} 秒"])
    data=[dict(t=i+1,body=np.round(raw['body'][i,0],4).tolist(),skills=int(raw['skills'][i,0]),position=int(raw['position'][i,0]),action=int(raw['action'][i,0]),product=int(raw['product'][i,0]),phase=int(raw['phase'][i,0])) for i in range(1536)]
    passed=sum(g['passed'] for g in s['gates'].values());neural=s['main']['neural_student']['survival'];frozen=s['main']['neural_student_frozen']['survival']
    css='''*{box-sizing:border-box}body{margin:0;background:#f0f4f2;color:#1f3633;font:16px/1.8 system-ui,-apple-system,"Microsoft Yahei",sans-serif}.wrap{max-width:1050px;margin:auto;padding:40px 22px 70px}header{padding:25px 0 28px;border-bottom:2px solid #318675}.kicker{font-size:13px;letter-spacing:.16em;color:#4f746b}h1{font-size:clamp(30px,6vw,51px);line-height:1.22;max-width:840px;margin:18px 0}h2{font-size:24px;line-height:1.45;margin-top:0}.lead{font-size:19px;color:#496761}section{padding:28px;background:white;border:1px solid #d8e3dd;border-radius:16px;margin:24px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:25px}.card{padding:16px;background:#e4eee8;border-radius:10px}.card strong{display:block;font-size:27px}.card span{font-size:13px}.note{padding:16px 20px;border-left:4px solid #318675;background:#edf4f0;margin:18px 0}.warn{border-left-color:#a5754d;background:#f6efe6}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.65}td,th{padding:12px 10px;border-bottom:1px solid #dee7e1;text-align:left;vertical-align:top}th{background:#edf3ef;white-space:nowrap}figure{margin:23px 0 10px}img{width:100%;height:auto;display:block}figcaption,.small{font-size:13px;color:#59736a}pre{white-space:pre;overflow:auto;background:#173b34;color:#e6eee9;padding:20px;border-radius:10px;font-size:13px}.replay{padding:20px;border-radius:12px;background:#e9f0ec}.replay input{width:100%;accent-color:#318675}.meter{height:12px;background:#cfddd4;border-radius:8px;overflow:hidden;margin:7px 0 12px}.meter span{display:block;height:100%;background:#318675}a{color:#276b60}footer{color:#59736a;font-size:13px;margin-top:30px}@media(max-width:600px){.wrap{padding:18px 12px 45px}section{padding:21px 15px}.cards{gap:7px}.card{padding:11px}.card strong{font-size:23px}td,th{min-width:100px;padding:10px 8px}.lead{font-size:17px}}'''
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_07 连续发育实验</title><style>{css}</style><body><main class="wrap">
<header><div class="kicker">WUYU · EXPERIMENT 07 · 2026-10-10</div><h1>经历能否留下<br>可以再次使用的能力？</h1><p class="lead">直接神经控制能维持较长生活，但面对新需要与新依赖，仍未学稳如何组织已有知识。</p><div class="cards"><div class="card"><strong>{neural['mean']:.1%}</strong><span>神经版本正式存活时长占比</span></div><div class="card"><strong>3 × 32</strong><span>每个条件的正式个体</span></div><div class="card"><strong>{p['student_mature_success']['mean']:.1%}</strong><span>神经版本的新需要／依赖测试成功率</span></div></div></header>
<section><h2>已经做出的内容</h2><p>六位置世界中，每台设备需要两次正确按钮准备，再取出能源、维修或中间物。个体通过自己的操作学习按钮规律与产物；后续生活加入中间物依赖、需要变化和局部规则改变，身体和记忆跨阶段保留。</p><p>两个实现分别接受检验：混合体使用显式的通用依赖规划器；神经学生经过教师轨迹与自身轨迹训练，正式运行只由循环网络输出动作。二者的关联模型和技能执行参数由实际反馈更新。</p><div class="note">直接神经控制的存活时长占比为 {percent(neural)}；关闭其个体关联/技能学习后为 {percent(frozen)}。这支持本轮神经控制依赖在线获得的知识；它不等于共享动作网络已实现终身权重学习。</div><p>神经学生共享网络分配 4,680 参数，其中 4,647 参与动作路径，33 个价值参数在模仿训练中未使用；另有每个个体 168 个可塑参数。关联头是 one-hot softmax 记忆，有明确容量与坐标先验。</p></section>
<section><h2>正式连续生活</h2>{table(['系统','存活时长占比及 95% 条件区间','身体资源面积 0–2','三个初始化'],rows)}{image('lives.png','正式生命长 1,536 事件。存活时长占比不是无限生存概率；三个初始化共享 32 个配对世界。')}<p>普通循环参照的网络为 1,520 参数，使用 actor-critic 训练；神经学生使用更多参数、显式关联记忆和教师监督。二者不是严格等容量或同先验的比较，不能将全部差异归因于技能。</p></section>
<section><h2>经历、组合、重新估值</h2>{table(['受控检查','经历前／固定需要／改变前','经历后／当前需要／练习后'],evidence)}{image('mechanisms.png','这些是混合控制器的机制检查；独立操作、组合和保留测试由评估者给出目标设备，参数冻结。不能把这些成绩冒充自主神经策略的成绩。')}<p>新组合测试使用第 384 事件前的经历，切换到需要中间物的世界；原有参数不能在测量窗口更新。改变测试在前后两次固定参数测量之间，允许最多 16 步亲身练习。未改变操作的保留受益于各设备参数分离，尚未解决共享大网络的遗忘。</p><p>语义预测的全按钮平均 Brier 均方误差从 {p['semantic_birth_brier']['mean']:.4f} 降到 {p['semantic_mature_brier']['mean']:.4f}。只使用预测损失学习该通道，策略收益梯度没有写入它。</p>{table(['编号','原定研发判据','结果'],gates)}<p>{passed}/5 是有限混合体的分项研发判据，不是生命、意识或总体智能评分。多个比较未校正，不把它们合成一个总体显著性结论。</p><div class="note warn">独立技能执行器的额外贡献：full 与同知识的 no_skill 截断完成成本差为 {diff['skill_cost_saved']['mean']:.3f} 步，95% 条件区间 [{diff['skill_cost_saved']['ci'][0]:.3f}, {diff['skill_cost_saved']['ci'][1]:.3f}]。机制存在和被调用，不足以证明它不可替代。</div></section>
<section><h2>神经个体能否根据经历满足需要</h2>{image('neural_probes.png','第 384 事件检查点的旁路：相同身体、位置和设备初态，在新依赖条件下自主行动；无目标设备提示，正常在线学习。')}<p>成熟神经个体的成功率为 {percent(p['student_mature_success'])}，出生个体为 {percent(p['student_birth_success'])}。保持同一循环隐状态、但清空关联记忆后的成功率为 {percent(p['student_memory_reset_success'])}。后者有输入分布改变，属于依赖诊断，不能独立当作理解证明。</p><p>这里的成功是 40 步内第一次获得的身体资源对应当前较大的缺口。资源与需要的基础含义来自设计，尚未证明价值标准自主形成。</p></section>
<section><h2>追加诊断：知识能否被控制器有效使用</h2><p>看到神经自主测试失败后，固定增加一个探索性 2×2 诊断；没有再训练、改动模型或重选模型。把同一学生第 384 事件的初始关联与技能记忆，分别交给原神经策略、最高概率动作策略以及通用规划控制器。</p>{table(['设备关系','需要变化','神经采样','神经最高概率','规划器读取同一初始记忆'],diagnostic_rows)}<p>规划控制器在这些条件中均达到 100%；取消神经动作采样没有消除失败。这提示问题包含控制策略如何利用已有知识，不能仅解释为知识完全没有学到，也不能仅归因于随机探索。</p><p>两条控制路径行动后仍正常学习，后续经历可能分叉；规划器没有学生的循环隐状态且具有手写结构先验，因此结果不能精确隔离某一个神经部件。此诊断复用了正式世界，是失败后的探索性定位，不是另一份独立确认。</p><p>追加的 {daudit['traces']:,} 条轨迹、{daudit['steps']:,} 步行动均已重放并重算成功标签。</p></section><section><h2>查看一条真实录制的生命</h2>{image('recorded_life.png','固定展示初始化 0、世界 740000；竖线是环境变化位置。未按成功与否筛选该个体。')}<div class="replay"><label for="tick">拖动查看记录：<strong id="time"></strong></label><input id="tick" type="range" min="0" max="1535" value="383"><div id="event"></div><div>能源 <span id="energy"></span><div class="meter"><span id="ebar"></span></div></div><div>完整度 <span id="integrity"></span><div class="meter"><span id="ibar" style="background:#bd916b"></span></div></div><div id="knowledge"></div></div><p class="small">页面展示已有记录，滑动不会训练模型，也不会让个体在后台继续生活。</p></section>
<section><h2>训练调整与数据隔离</h2><p>初轮教师轨迹模仿的动作准确率较高，闭环生活仍不稳定；因此在正式集封存期间，增加固定预算的自身轨迹训练，让教师根据学生当前已经学到的知识提供动作标签。全部旧候选保留；用相同独立验证集选点，正式 740000–740031 世界从未用于选模。</p>{table(['阶段','初始化','选中模型','验证存活时长','阶段运行时间'],budgets)}<p>每个循环参照训练 768 × 32 × 64 = 1,572,864 个计划事件；每个学生初轮和自身轨迹修正各 512 × 32 × 32 = 524,288 个计划事件。包含失能后的占位槽；修正日志另报有效样本数。分项运行时间不等于整个制作耗时。</p><p>统计先对三个固定初始化平均，再按 32 个世界配对重采样 4,000 次；区间不覆盖任意训练初始化。旁路来自同一批正式个体，不能视为额外独立重复。</p></section>
<section><h2>核验与限制</h2><p>独立物理账本复算 {audit['event_slots']:,} 个正式事件槽，指标从原始记录重算；检查 {audit['skill_origins']:,} 次技能调用的历史来源。六个真实检查点各恢复 32 个世界、384 个事件，后续动作、身体与下一检查点全部字段一致。{audit['npz_files']} 个 NPZ 通过 CRC；{audit['probe_traces']:,} 条旁路、{audit['probe_steps']:,} 个动作的物理结果与成功标签核验通过。</p><p>13 项核心检查包括神经梯度、反馈时序、隐藏信息边界、身体成本、死亡吸收、技能来源、参数冻结、完整恢复和学生部署时无教师/规划调用。正式开始后的核心和协议哈希保持一致；RNN 开始训练后修改了 {', '.join(audit['changed_since_rnn_training']) or '无文件'}，用途和兼容性记录于评价细节，未隐瞒这些修改。</p><div class="note warn">已经形成的是有限关联与操作能力的可复查实验。六设备、两阶段技能模板、身体价值、记忆分配以及教师规划方式都具有人工先验。学生的共享动作权重在正式生命中固定；技能表示和概念没有开放式增长，尚不能把本版本称为完整物欲基底、数字生命或人。</div></section>
<section><h2>继续运行</h2><pre>cd WUYU_07
python -m pip install -r requirements.txt
python demo.py --steps 48
python demo.py --steps 64 --save continued_07.npz
python demo.py --checkpoint continued_07.npz --steps 64</pre><p>默认恢复实际神经个体的第 384 事件检查点。轻量包适合先运行；完整包包含全部正式记录、候选模型、旁路轨迹和审计。程序仅在执行期间运行。</p></section>
<section><h2>研究依据</h2><p><a href="https://proceedings.mlr.press/v80/barreto18a.html">Barreto 等，2018：后继特征与跨任务技能复用</a>；<a href="https://arxiv.org/abs/1907.01657">Sharma 等，2020：可预测技能发现与组合</a>；<a href="https://www.jneurosci.org/content/27/15/4019">Valentin 等，2007：结果价值改变与目标导向行为实验</a>。本代码并未复现这些论文；它们提供实验问题与设计参考。</p></section><footer>WUYU_06 原始代码和数据未修改。所有成绩取自随附实际运行，保留失败与未通过项。</footer></main>
<script>const life={json.dumps(data,separators=(',',':'))};const acts=['左移','右移','按钮 0','按钮 1','按钮 2','取出','等待'];const products=['能源','维修','中间物 0','中间物 1','无'];function show(){{const d=life[+document.getElementById('tick').value];document.getElementById('time').textContent=d.t+' / 1536';document.getElementById('event').textContent='动作：'+acts[d.action]+' · 位置 '+d.position+' · 交付：'+products[d.product];document.getElementById('energy').textContent=d.body[0].toFixed(3);document.getElementById('integrity').textContent=d.body[1].toFixed(3);document.getElementById('ebar').style.width=100*d.body[0]+'%';document.getElementById('ibar').style.width=100*d.body[1]+'%';document.getElementById('knowledge').textContent='有效操作技能：'+d.skills+' / 6 · '+(Math.min(...d.body)>.001?'存活':'失能');}}document.getElementById('tick').addEventListener('input',show);show();</script></body></html>'''
    path=ROOT.parent/'WUYU_07_REPORT.html';path.write_text(body);(ROOT/'REPORT.html').write_text(body)
    print(json.dumps(dict(report=str(path),bytes=path.stat().st_size,figures=4)))
