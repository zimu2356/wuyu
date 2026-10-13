import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,base64,html
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures'
CELLS=['flat_regular','flat_varied','graph_regular','graph_varied']
CN={'flat_regular':'循环 · 常规训练','flat_varied':'循环 · 需要变化训练','graph_regular':'关系 · 常规训练','graph_varied':'关系 · 需要变化训练','graph_varied_frozen':'关系 · 关闭个体学习'}
CONDS=['independent','dependency','long_chain','changed'];CC=['独立设备','新依赖','更长链条','关系改变']
def p1(v):return str(Decimal(str(v*100)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
def pct(s):return f"{p1(s['mean'])}% [{p1(s['ci'][0])}, {p1(s['ci'][1])}]"
def num(s):return f"{s['mean']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
def table(headers,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def image(name):return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'">'

if __name__=='__main__':
    FIG.mkdir(exist_ok=True);s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());cfg=json.loads((ROOT/'CONFIG.json').read_text())
    q=lambda cell,c,metric='first_success',history='common',mode='none':s['probes'][f'{cell}/{history}/{c}/{mode}'][metric]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fc','axes.facecolor':'#ffffff'})
    data=np.array([[q(cell,c)['mean'] for c in CONDS] for cell in CELLS])
    fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_xticks(range(4),['Independent','Dependency','Long chain','Changed link']);ax.set_yticks(range(4),['Flat / regular','Flat / varied','Graph / regular','Graph / varied'])
    for i in range(4):
        for j in range(4):ax.text(j,i,p1(data[i,j])+'%',ha='center',va='center',color='white' if data[i,j]>.65 else '#123047',fontsize=13,fontweight='bold')
    ax.set_title('First body resource matches current need · held-out worlds',pad=15);fig.colorbar(im,ax=ax,label='Success fraction');fig.tight_layout();fig.savefig(FIG/'comparison.png',dpi=160);plt.close(fig)
    keys=CELLS+['graph_varied_frozen'];vals=np.array([s['lives'][c]['survival']['mean'] for c in keys]);cis=np.array([s['lives'][c]['survival']['ci'] for c in keys])
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(np.arange(5),vals*100,color=['#b8c6db','#7b9cbd','#9dc3bd','#247d77','#d7a366'])
    ax.errorbar(vals*100,np.arange(5),xerr=np.maximum(0,np.stack([vals-cis[:,0],cis[:,1]-vals]))*100,fmt='none',color='#243247',capsize=4)
    ax.set_yticks(np.arange(5),['Flat / regular','Flat / varied','Graph / regular','Graph / varied','Graph / learning off']);ax.invert_yaxis();ax.set_xlim(0,109);ax.set_xlabel('Fraction of 1,536 events alive (%)');ax.set_title('Autonomous life with six external resource-loss events')
    for i,v in enumerate(vals):ax.text(min(103,cis[i,1]*100+1.4),i,p1(v)+'%',va='center')
    fig.tight_layout();fig.savefig(FIG/'lives.png',dpi=160);plt.close(fig)
    modes=['none','one_round','zero_round','cut_edges','planner'];xx=np.arange(len(modes));fig,ax=plt.subplots(figsize=(9,4))
    for j,(c,color) in enumerate([('dependency','#287c98'),('long_chain','#ba763e')]):
        ax.bar(xx+(j-.5)*.35,[q('graph_varied',c,mode=m)['mean']*100 for m in modes],width=.34,color=color,label='Dependency' if j==0 else 'Long chain')
    ax.set_xticks(xx,['3 rounds','1 round','0 rounds','No messages','Planner']);ax.set_ylim(0,108);ax.set_ylabel('First-needed-resource success (%)');ax.set_title('Same-history controller interventions');ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'mechanisms.png',dpi=160);plt.close(fig)
    with np.load(OUT/'life_graph_varied_0.npz') as z:
        body=z['body'][:,0];action=z['action'][:,0];goal=z['goal'][:,0];target=z['target'][:,0];product=z['product'][:,0];alive=z['alive'][:,0];pos=z['position'][:,0]
    fig,ax=plt.subplots(figsize=(9,3.4));ax.plot(np.arange(1,1537),body[:,0],label='Energy',color='#247c9c');ax.plot(np.arange(1,1537),body[:,1],label='Integrity',color='#b16d3b')
    for t in [384,768,1152]:ax.axvline(t,color='#a0aabc',ls='--',lw=.8)
    for t in [480,640,960,1120,1280,1440]:ax.axvline(t,color='#d88686',alpha=.4,lw=.8)
    ax.set_ylim(0,1.04);ax.set_xlabel('Physical event');ax.set_title('Recorded life · initialization 0 · world 20,000,000');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(FIG/'recorded_life.png',dpi=160);plt.close(fig)
    rows=[[CN[cell]]+[pct(q(cell,c)) for c in CONDS] for cell in CELLS]
    comparison=table(['控制器']+CC,rows)
    outcomes=table(['关系条件','首次对应需要','40 步内最终取得','未取得身体资源','截断首次成本','需要资源净变化'],[[cc,pct(q('graph_varied',c)),pct(q('graph_varied',c,'any_success')),pct(q('graph_varied',c,'no_body')),num(q('graph_varied',c,'first_cost')),num(q('graph_varied',c,'body_gain'))] for c,cc in zip(CONDS,CC)])
    effectrows=[]
    for label,cn in [('coverage_flat','循环结构：改变训练覆盖'),('coverage_graph','关系结构：改变训练覆盖'),('architecture_regular','常规训练：改变结构'),('architecture_varied','需要变化训练：改变结构'),('interaction','两因素交互')]:
        effectrows.append([cn]+[pct(s['effects'][f'{label}/{c}']) for c in CONDS])
    effects=table(['配对成功率差（百分点）']+CC,effectrows)
    lifetable=table(['连续自主生活','存活时长占比','身体资源面积 0–2','五个初始化'],[[CN[c],pct(s['lives'][c]['survival']),num(s['lives'][c]['body_area']),'/'.join(f'{x*100:.1f}%' for x in s['lives'][c]['survival']['by_initialization'])] for c in keys])
    own=table(['控制器','自主经历后的独立设备','自主经历后的新依赖'],[[CN[c],pct(q(c,'independent',history='own')),pct(q(c,'dependency',history='own'))] for c in CELLS])
    need=table(['控制器','原始需要输入','身体输入固定','同一个体两种需要都成功'],[[CN[c],pct(q(c,'independent')),pct(q(c,'independent',mode='body_clamp')),pct(q(c,'independent','pair_success'))] for c in CELLS])
    mechan=table(['关系网络干预','普通依赖','更长链条'],[[label,pct(q('graph_varied','dependency',mode=m)),pct(q('graph_varied','long_chain',mode=m))] for m,label in [('none','完整三轮'),('one_round','只保留一轮'),('zero_round','取消内部轮次'),('cut_edges','移除依赖消息'),('planner','同知识显式规划器')]])
    trainrows=[]
    for c in CELLS:
        for i in range(5):
            t=json.loads((OUT/f'training_{c}_{i}.json').read_text());b=t['selected'];trainrows.append([CN[c],i,b['update'],f"{b['score']*100:.1f}%",t['active_examples'],f"{t['seconds']:.1f} 秒"])
    train=table(['组别','初始化','选中更新','验证成功率','有效训练事件','进程墙钟时间'],trainrows)
    diagnosis=json.loads((OUT/'failure_diagnosis.json').read_text())
    supplementary=('<section><h2>更长链条的失败发生在哪里</h2><p>看到首个正式初始化的失败后，固定增加这项探索性轨迹分析；没有重新训练、选模或改变主指标。它复用同一批正式世界。</p>'
        +f"<p>候选版本共 {diagnosis['total_branches']} 个长链分支，其中 {diagnosis['failed_branches']} 个主指标失败；{diagnosis['failed_with_blocked_take']} 个失败分支出现过“设备已准备、但缺少前置物，仍执行取出”的动作。失败分支的这种动作中位数为 {diagnosis['median_blocked_take_among_failures']:.1f} 次。这里描述可观察事件，不据此断言唯一的神经内部原因。</p>")
    if diagnosis['adequate_branches']:
        supplementary+=f"<p>在所有六台设备的产物、正确操作与当前前置条件均可由已有记忆确定的保守子组中，共 {diagnosis['adequate_branches']} 个分支、涉及 {diagnosis['adequate_unique_worlds']} 个独立世界。神经控制成功率 {diagnosis['adequate_neural_success']*100:.1f}%，同知识规划器 {diagnosis['adequate_planner_success']*100:.1f}%。子组按观察者的物理真值筛选，属于有条件的探索分析；真值没有输入行动网络。</p>"
    ex=diagnosis['example']
    if ex:
        kinds=['能源','修复资源','中间物 0','中间物 1']
        reached=[f"第 {t+1} 步在设备 {ex['planner_positions'][t]} 取得{kinds[k]}" for t,k in enumerate(ex['planner_products']) if k<4][:3]
        supplementary+=f"<p>按预定顺序取出的首个案例：初始化 {ex['initialization']}、世界 {ex['world']}。神经控制没有取得身体资源，发生 {ex['blocked_take_actions']} 次缺少材料的取出尝试；同知识规划器的前三次产物为：{'；'.join(reached)}。对应完整动作序列随原始记录保留。</p>"
    supplementary+='<p class="small">完整分类规则、逐分支结果与第一个符合固定条件的失败案例，保留在 SUPPLEMENT_DIAGNOSTIC.md、diagnose_failures.py 与 results/failure_diagnosis.json。</p></section>'
    gd=q('graph_varied','dependency')['mean'];gl=q('graph_varied','long_chain')['mean'];gi=q('graph_varied','independent')['mean'];sv=s['lives']['graph_varied']['survival']['mean']
    cov=s['effects']['coverage_graph/dependency'];arch=s['effects']['architecture_varied/dependency']
    conclusion=f'关系结构＋需要变化训练，在独立设备／新依赖／更长链条中的首次需要匹配成功率为 {p1(gi)}%／{p1(gd)}%／{p1(gl)}%。自主连续生活的存活时长占比为 {p1(sv)}%。这些是不同实验，不合成为“生命评分”。'
    ledger={'body':body.round(5).tolist(),'action':action.tolist(),'goal':goal.tolist(),'target':target.tolist(),'product':product.tolist(),'alive':alive.astype(int).tolist(),'position':pos.tolist()}
    text=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_08 · 需要能否真正改变行动？</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f3f5f9;color:#18273b;font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.8}}main{{max-width:1060px;margin:auto;padding:28px 22px 64px}}header{{padding:30px 0 16px}}.eyebrow{{letter-spacing:.15em;color:#487879;font-size:12px;font-weight:700}}h1{{font-size:clamp(29px,6vw,49px);line-height:1.25;max-width:750px;margin:18px 0}}h2{{font-size:23px;margin-top:0}}h3{{font-size:17px}}p{{margin:12px 0}}.sub{{font-size:18px;color:#526477;max-width:810px}}section{{background:white;border:1px solid #e1e6ed;border-radius:16px;padding:26px;margin:22px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#e9f0f3;border-radius:12px;padding:18px}}.big{{font-size:33px;font-weight:750;color:#1b6668}}.small{{font-size:13px;color:#647384}}.table-wrap{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:11px 12px;border-bottom:1px solid #e2e7ee;text-align:left;vertical-align:top;min-width:120px}}th{{background:#edf2f7;font-weight:650}}td:first-child,th:first-child{{min-width:170px}}img{{display:block;width:100%;height:auto;margin:18px 0}}.note{{border-left:4px solid #bb8954;background:#fbf5eb;padding:12px 16px;margin:18px 0}}code,pre{{font-family:ui-monospace,monospace}}pre{{overflow:auto;background:#172638;color:#e8f1f7;padding:18px;border-radius:10px;line-height:1.65}}a{{color:#176b91}}input[type=range]{{width:100%;accent-color:#237d80}}.readout{{background:#edf3f6;padding:15px;border-radius:10px;font-variant-numeric:tabular-nums}}footer{{color:#6b7788;font-size:13px}}@media(max-width:600px){{main{{padding:16px 12px 40px}}section{{padding:19px 15px}}.cards{{gap:7px}}.card{{padding:12px 9px}}.big{{font-size:25px}}.card span{{font-size:12px}}}}
    </style></head><body><main><header><div class="eyebrow">WUYU · EXPERIMENT 08 · 2026-10-10</div><h1>需要能否<br>真正改变行动？</h1><p class="sub">四组对照、五个训练初始化。检验当前身体需要能否通过亲身获得的知识，改变神经个体的选择与行动。</p></header>
    <div class="cards"><div class="card"><div class="big">{p1(gi)}%</div><span>候选版本 · 独立设备</span></div><div class="card"><div class="big">{p1(gd)}%</div><span>候选版本 · 新依赖</span></div><div class="card"><div class="big">{p1(gl)}%</div><span>候选版本 · 更长链条</span></div></div>
    <section><h2>本轮实际做了什么</h2><p>{conclusion}</p><p>两种控制器都直接输出左移、右移、三种按钮、取出和等待七种原始动作。正式运行不调用教师、递归规划或手写导航。新关系网络以共享参数进行三轮依赖关系处理；目标和子目标训练标签来自读取学生自身经历记忆的教师。</p><p>循环网络 11,325 参数，关系网络 11,345 参数，相差约 0.18%；但循环状态分别为 30 和 240 个标量，关系网络计算更多。比较涉及对象分解、关系结构与内部轮次，不能只归因于“多想几轮”。每个个体仍有 168 个在线关联／技能参数。</p><div class="note">这是需要控制与关系组合的有限实验。基础身体价值、输入表示、依赖边构造和教师规划先验由设计提供；共享策略权重在正式生活中固定。尚未实现开放式概念、自主价值起源或显式未来身体轨迹推演。</div></section>
    <section><h2>四组对照：共同经历后的需要匹配</h2><p>所有模型观察相同的真实教师探索经历，通过实际动作反馈建立记忆。条件切换后进行六步物理巡查，逐站观察当前前置条件；随后复制同一状态，只交换身体缺口。参数冻结，自主执行 40 步。成功要求第一次取得的身体资源对应当前较大缺口。</p>{image('comparison.png')}{comparison}<p class="small">每项 5 个固定训练初始化 × 32 个世界 × 2 个需要方向。括号为配对世界重采样的 95% 条件区间。初始化先平均，分支不当作独立世界；区间不覆盖所有可能训练初始化。</p><p>常规与变化训练都包含普通依赖关系。变化组额外经历身体缺口，因此此因素主要检验需要覆盖；所有训练均不包含中间物之间的新增依赖。更长链条测试要求先取得 token 0，再制作 token 1，最后获得身体资源。</p>{effects}<p class="small">差值按百分点显示；负值表示对应改变降低成绩。多个研发对照未作家族错误率校正，逐项报告，不合并显著性。</p></section>
    <section><h2>选对需要、完成链条与身体收益</h2><p>以下展开关系＋变化训练的全部结果。第一次选错仍计为主指标失败；“最终取得”另报后续恢复，避免只报最有利定义。</p>{outcomes}<p>代价为取得第一次正确身体资源的动作数，主指标失败截断为 40。资源净变化是 40 步后当前缺口资源与起点之差，不是独立的主观价值测量。</p>{need}<p>固定身体输入的干预保留其余可见输入与历史，可能改变输入分布；它与配对需要实验共同用于定位依赖，不能单独证明理解。双需要成功要求同一个体的两个分支都选对。</p></section>
    <section><h2>关系运算是否真正参与控制</h2>{image('mechanisms.png')}{mechan}<p>这些分支来自同一份初始经历记忆，塑性参数冻结，只有控制器的运算或输入改变。显式规划器具有手写结构先验，不持有神经隐藏状态，属于诊断参照。内部轮次干预也会改变中间状态分布。</p><p>如果移除某个部件没有降低成绩，应保留这一否定结果；完整架构胜出并不足以证明每一轮计算都必要。该网络目前处理已学依赖关系，没有显式预测多步身体轨迹。</p></section>
    <section><h2>自主连续生活</h2><p>这部分从出生开始由网络自行行动，个体记忆在线学习，完全不提供教师动作。四个原有物理阶段之间身体与历史持续保留；另有六次固定外部资源损失，观察需要在实际生活中改变时能否应对。</p>{image('lives.png')}{lifetable}<p>关闭候选版本的个体关联／技能学习，存活时长占比差为 {pct(s['online_learning_effect'])}。这项干预改变学习及其可见知识状态；不等于共享网络实现了终身权重发育。</p><p>从各自自主生活第 384 事件取出旁路，检验自己的经历能否支持同样的需要任务：</p>{own}<p class="small">已失能的父个体不会被复活。共同经历成绩与自主经历成绩分别报告，教师引导的知识获得不能冒充自主学习。</p></section>
    <section><h2>查看一条实际记录</h2>{image('recorded_life.png')}<p class="small">固定展示初始化 0、世界 20,000,000，未按结果筛选。虚线为物理阶段切换，淡红线为外部资源损失。</p><label for="event">事件位置</label><input id="event" type="range" min="1" max="1536" value="384"><div class="readout" id="readout"></div><p class="small">滑动只查看已录制数据，不会训练模型，也不会使个体在后台继续生活。</p></section>
    <section><h2>训练、选择与验证边界</h2><p>每个模型固定 2,048 × 16 × 24 = 786,432 个计划事件；前半使用教师轨迹，随后逐步转到自己的轨迹，最后四分之一完全由学生行动。所有监督均依据学生当前已获得的知识。训练含失能后的占位事件，有效事件另列。各组保留三个候选，用独立验证集选择；正式集打开后不再训练或重选。</p>{train}<p class="small">进程时间受并行执行影响，不作为纯架构速度比较。工程 64 次更新的试跑失败和 1,024 次更新的预算试跑均保留；它们使用独立工程世界。</p><p>独立物理账本复算 {audit['life_event_slots']:,} 个正式生命事件槽、{audit['probe_event_slots']:,} 个分支事件槽和 {audit['common_history_event_slots']:,} 个共同经历事件槽。分支指标从轨迹重算，检查共同初始事实一致与冻结参数未变。四组各一个实际训练检查点恢复 32 个个体、连续执行 384 步，全部后续状态与保存记录一致。</p><p>9 项核心检查包括循环／多轮梯度、隐藏信息边界、反馈时序、成本与死亡、前置条件、冻结分支、恢复、相同信息接口和部署无教师调用。另核验 {audit['npz_crc_files']} 个 NPZ 的 CRC，训练及正式阶段封存的源文件哈希保持一致。</p></section>
    <section><h2>继续运行与研究方向</h2><pre>python -m pip install -r requirements.txt
python demo.py --steps 48
python demo.py --steps 64 --save continued_08.npz
python demo.py --checkpoint continued_08.npz --steps 64</pre><p>默认恢复本报告展示组的真实第 384 事件检查点。下一步应根据四组差异和更长链条失败的位置，继续研究学习到的计算如何泛化，以及共享控制能力如何在个人生命中改变。</p><p>本轮仍只有六个已知位置、有限资源类别与两阶段设备操作。更长链条是有限结构外推，未检验任意对象数量、任意新概念或开放世界。行为改进不等于生命、意识或人格已经成立。</p><p class="small">研究参考：<a href="https://arxiv.org/abs/1602.02867">Value Iteration Networks</a>；<a href="https://proceedings.mlr.press/v162/velickovic22a.html">CLRS Algorithmic Reasoning Benchmark</a>；<a href="https://arxiv.org/abs/2301.04104">DreamerV3</a>。本实验未复现这些论文，不将其成绩归于本代码。</p></section>
    <footer>源码、实际模型、负面结果和核验记录随包提供。WUYU_07 原始内容未修改。</footer></main><script>const data={json.dumps(ledger,separators=(',',':'))};const actions=['左移','右移','按钮 0','按钮 1','按钮 2','取出','等待'];const input=document.getElementById('event');function render(){{const i=Number(input.value)-1;document.getElementById('readout').textContent=`事件 ${{i+1}} ｜ 能源 ${{data.body[i][0].toFixed(3)}} ｜ 完整度 ${{data.body[i][1].toFixed(3)}} ｜ 动作：${{actions[data.action[i]]}} ｜ 位置 ${{data.position[i]}} ｜ 网络目标 ${{data.goal[i]}} / 子目标 ${{data.target[i]}}`;}}input.addEventListener('input',render);render();</script></body></html>'''
    text=text.replace('<section><h2>自主连续生活</h2>',supplementary+'<section><h2>自主连续生活</h2>')
    text=text.replace('关系网络计算更多。','关系网络计算更多。身体消耗只计模拟动作成本，内部计算没有额外扣除身体资源。')
    text=text.replace('每个个体仍有 168 个在线关联／技能参数。','上述总量包含辅助终极目标读出；直接参与原始动作路径的参数分别是 11,139 与 11,304。终极目标读出用于训练和诊断，动作解码器使用子目标注意力，不能把辅助读出的准确率当作它直接控制了行动。每个个体仍有 168 个在线关联／技能参数。')
    (ROOT/'REPORT.html').write_text(text,encoding='utf-8');(ROOT.parent/'WUYU_08_REPORT.html').write_text(text,encoding='utf-8')
    print(json.dumps(dict(report_bytes=len(text.encode()),figures=4)))
