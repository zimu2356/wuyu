from pathlib import Path
import json,base64,html,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES={'full':'完整核心','plain_rnn':'普通循环代理','rule':'规则控制器','frozen_fast':'冻结个体预测更新','no_replay':'关闭回放效果','no_goal':'移除经历目标','no_forecast':'屏蔽预测输入'}
EN={'full':'Full core','plain_rnn':'Plain RNN','rule':'Rule controller','frozen_fast':'Frozen fast model','no_replay':'No replay effect','no_goal':'No experience goal','no_forecast':'No forecast input'}
COLORS={'full':'#286b60','plain_rnn':'#77878d','rule':'#c1bbaa','frozen_fast':'#759caa','no_replay':'#a99a75','no_goal':'#bba893','no_forecast':'#bc7352'}

def pct(x):return f'{100*x:.2f}%'
def ci(v,percent=False):
    f=pct if percent else lambda x:f'{x:.4f}'
    return f"{f(v['mean'])} [{f(v['ci95'][0])}, {f(v['ci95'][1])}]"
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def image(name,caption):
    data=base64.b64encode((FIG/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


if __name__=='__main__':
    summary=json.loads((OUT/'summary.json').read_text());stress=json.loads((OUT/'stress_summary.json').read_text());probes=json.loads((OUT/'probes.json').read_text());diagnostic=json.loads((OUT/'w05_diagnostic.json').read_text());audit=json.loads((OUT/'audit.json').read_text());training=json.loads((OUT/'training_summary.json').read_text())
    methods=list(NAMES);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f8f4','axes.facecolor':'#f7f8f4','savefig.facecolor':'#f7f8f4'})
    fig,ax=plt.subplots(figsize=(9,4.5));y=np.arange(len(methods));values=np.array([summary['methods'][m]['survival']['mean'] for m in methods]);bounds=np.array([summary['methods'][m]['survival']['ci95'] for m in methods])
    ax.barh(y,values,color=[COLORS[m] for m in methods],height=.62);ax.errorbar(values,y,xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='none',ecolor='#243a39',capsize=3)
    ax.set_yticks(y,[EN[m] for m in methods]);ax.invert_yaxis();ax.set_xlim(0,1.13);ax.set_xlabel('Fraction of the 2,048-event life still active')
    for j,v in enumerate(values):ax.text(max(v,bounds[j,1])+.017,j,f'{v*100:.2f}%',va='center',fontsize=10)
    ax.set_title('Main test: survival reaches a ceiling',loc='left',fontweight='bold',pad=16);fig.tight_layout();fig.savefig(FIG/'main.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,kind,title in zip(axs,['paid_observation','maintenance_drain'],['Uninformative free cues','Extra maintenance drain']):
        x=stress[kind]['summary'];v=np.array([x['methods'][m]['survival']['mean'] for m in methods]);b=np.array([x['methods'][m]['survival']['ci95'] for m in methods]);ax.barh(y,v,color=[COLORS[m] for m in methods],height=.62)
        ax.errorbar(v,y,xerr=np.stack([v-b[:,0],b[:,1]-v]),fmt='none',ecolor='#243a39',capsize=3);ax.set_xlim(0,1.1);ax.set_yticks(y,[EN[m] for m in methods]);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Active-life fraction, 1,024 events')
    axs[0].invert_yaxis();fig.suptitle('Exploratory stress tests — no retraining',x=.02,ha='left',fontweight='bold');fig.tight_layout();fig.savefig(FIG/'stress.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharex=True)
    for ax,origin in zip(axs,['fixed','learned']):
        group=diagnostic['groups'][origin+'_change_yoked']['contrasts'];keys=['LF-FF','FL-FF','LL-FF'];vals=[group[k]['reward_128'] for k in keys]
        for j,(k,v) in enumerate(zip(keys,vals)):
            ax.errorbar(v['mean'],j,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',color=['#286b60','#bc7352','#3d5358'][j],capsize=4)
        ax.axvline(0,color='#9baba6',lw=1);ax.set_yticks(range(3),['Learned fresh only','Learned replay only','Both learned']);ax.invert_yaxis();ax.set_title(origin.capitalize()+' developmental origin',loc='left',fontweight='bold');ax.set_xlabel('First 128-event reward difference vs fixed/fixed')
    fig.suptitle('WUYU_05: identical starting state and external evidence',x=.02,ha='left',fontweight='bold');fig.tight_layout();fig.savefig(FIG/'w05.png',dpi=180);plt.close(fig)
    z=np.load(OUT/'raw_full_0.npz',allow_pickle=False);fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1.4,1,1]});t=np.arange(256)
    axs[0].plot(t,z['body'][:256,0,0],label='Energy',color='#286b60');axs[0].plot(t,z['body'][:256,0,1],label='Integrity',color='#bc7352');axs[0].set_ylim(0,1.05);axs[0].legend(ncol=2,frameon=False,loc='lower right');axs[0].set_title('One recorded life — model 0, world 980000',loc='left',fontweight='bold')
    axs[1].step(t,z['action'][:256,0],where='mid',color='#526d7b',lw=.9);axs[1].set_yticks(range(7),['left','right','interact','inspect','replay','rest','reconsider']);axs[1].tick_params(axis='y',labelsize=8)
    goalpos=z['goal'][:256,0,:4].argmax(1);goalpos[np.all(z['goal'][:256,0]==0,axis=1)]=-1
    axs[2].step(t,goalpos,where='mid',color='#8c7753');axs[2].set_yticks([-1,0,1,2,3],['none','site 0','site 1','site 2','site 3']);axs[2].set_ylabel('Goal location');axs[2].set_xlabel('Recorded event');fig.tight_layout();fig.savefig(FIG/'life.png',dpi=180);plt.close(fig)
    main_rows=[[NAMES[m],ci(summary['methods'][m]['survival'],True),f"{summary['methods'][m]['body_area']['mean']:.4f}"] for m in methods]
    gates=[('G1','超过普通循环代理',summary['gates']['G1']),('G2','个体预测更新增加存活收益',summary['gates']['G2']),('G3','经历目标增加存活收益',summary['gates']['G3_performance']),('G4','预测输入有因果作用且增加存活收益',summary['gates']['G4_performance'] and all(p['forecast_action_tv']>1e-6 for p in probes)),('G5','付成本回放增加存活收益',summary['gates']['G5'])]
    selected=[[r['mode'],r['initialization'],r['selected']['update'],pct(r['selected']['survival']),f"{r['seconds']:.2f}s"] for r in training['runs']]
    probe_rows=[[p['initialization'],f"{p['body_action_tv']:.4f}",f"{p['goal_action_tv']:.4f}",f"{p['forecast_action_tv']:.4f}",f"{p['history_action_tv']:.4f}"] for p in probes]
    diag_rows=[]
    for origin in ['fixed','learned']:
        for coupling in ['yoked','closed']:
            g=diagnostic['groups'][f'{origin}_change_{coupling}']['contrasts'];diag_rows.append([origin,coupling,ci(g['LF-FF']['reward_128']),ci(g['FL-FF']['reward_128']),ci(g['LL-FF']['reward_128'])])
    parts=[]
    parts.append('<section><h2>这次完成了什么</h2><p>一个 3,386 参数的共享循环神经核心，直接选择七种动作；身体后果预测、价值估计和经历目标选择共同参与训练。个体在运行中保留隐状态、有限记忆、目标与 693 个快速预测参数。目标候选来自实际交互获得的结果，资源不足会阻止相应环境操作。</p><p>完整核心的三个训练版本，在各自 32 个正式世界中都完成了 2,048 事件。普通循环代理同样全部完成，因此不能把维持运行的成功归功于新增目标、回放或快速学习。</p><div class="callout">主结论：连续运行闭环已接通；预测输入对这个已训练核心有可测作用。自主目标和回放的必要性尚未得到主指标支持。</div></section>')
    parts.append('<section><h2>主测试：运行成功，评价出现上限</h2>'+table(['方法','存活时长占比，95%条件区间','身体资源面积（0–2）'],main_rows)+image('main.png','存活时长占比表示测试时长内仍能行动的比例，不是无限期存活概率。多个方法达到 100%，显示这个指标在当前环境中缺少区分力。')+'<p>屏蔽预测输入后，完整核心相对该干预的存活时长差为 '+ci(summary['contrasts']['no_forecast']['survival'],True)+'。这支持预测输入在该核心中的作用；干预同时改变了策略输入分布，且普通循环代理同样达到上限，不能据此断言所有有效策略都需要这个预测模块。</p><p>完整核心相对普通循环代理的身体资源面积增加 '+ci(summary['contrasts']['plain_rnn']['body_area'])+'，是辅助结果，不能替代未通过的主门槛。</p></section>')
    parts.append('<section><h2>按事前判据判定</h2>'+table(['编号','主张','结果'],[[g,t,'通过' if v else '未通过'] for g,t,v in gates])+'<p>五项中只有 G4 满足本轮条件。没有将身体资源微小改善或行为变化替换成主指标通过。G3/G4 的行为影响条件由独立副本探针检查。</p></section>')
    parts.append('<section><h2>状态、目标和历史真的影响动作吗</h2>'+table(['训练版本','改变身体状态 TV','交换已经历目标 TV','屏蔽预测 TV','不同历史、同当前输入 TV'],probe_rows)+'<p>TV 是动作概率分布的总变差，范围 0–1。每次干预都在副本上进行，不写回正式生命。身体与目标确实可以改变动作，但这不等于改变合理，也不等于出现主观需要。历史探针控制当前位置、身体和观察，保留不同的内部经历状态；它只支持历史依赖，不能证明人格、偏好成长或迁移收益。</p><p>统一比较七种候选动作的后果预测时，保留快速参数的均方误差均值为 '+f"{np.mean([p['all_action_prediction_mse'] for p in probes]):.5f}"+'；清零快速参数后为 '+f"{np.mean([p['zero_fast_prediction_mse'] for p in probes]):.5f}"+'。这是受控预测诊断；主测试中快速更新的额外存活收益仍未通过。</p>'+image('life.png','真实原始记录中的一个个体。目标位置来自它曾获得资源的经历。目标变化和活动轨迹本身不作为自主性证据。')+'</section>')
    stress_rows=[]
    for m in methods:stress_rows.append([NAMES[m],pct(stress['paid_observation']['summary']['methods'][m]['survival']['mean']),pct(stress['maintenance_drain']['summary']['methods'][m]['survival']['mean'])])
    parts.append('<section><h2>补充压力测试仍未解除主要上限</h2><p>看到主测试上限后，先固定 SUPPLEMENT_01.md，再用新世界做一次性测试；没有重新训练或按结果调节压力。免费观察失去区分力后，付费 inspect 仍可获得可靠标记；另一项测试在首阶段后增加每步维护消耗。</p>'+table(['方法','免费标记失效','额外维护消耗'],stress_rows)+image('stress.png','探索性补充，不能改写原主门槛。两种压力下，完整核心和普通循环代理仍都达到存活时长上限。')+'<p>这进一步限制了结论：当前四位置、有限资源世界允许较简单的循环策略维持运行；它尚不足以检验需要长期目标或经验整合的发展。这里没有继续提高压力直到某种架构获胜。</p></section>')
    parts.append('<section><h2>WUYU_05 的原因诊断：回放调节也在拖慢适应</h2><p>从 fixed 与 learned 两种已形成的完整前缀分别复制个体，只替换新学习率和回放率。FF 为两项固定，LF 仅新学习率可学习，FL 仅回放率可学习，LL 两项可学习。yoked 条件固定外部动作、线索与标签；答案造成的内部反馈仍会不同。closed 为自主动作。</p>'+table(['前缀来源','证据条件','LF − FF','FL − FF','LL − FF'],diag_rows)+image('w05.png','反转后前 128 事件的每步奖励差。使用旧 WUYU_05 测试种子做探索性定位，不是新的独立确认试验。')+'<p>相同外部证据下，FL − FF 在两种前缀来源中均为负，区间均低于零；LF 的负面影响依赖前缀来源。这支持“回放率调节具有负因果贡献”，而非只怪新学习太慢。它尚未区分哪些旧记忆需要保留、哪些已不适用。</p></section>')
    parts.append('<section><h2>实现边界</h2>'+table(['项目','实际实现与限制'],[
      ['全局神经参数','完整核心 3,386；普通代理分配相同数组，其中目标/预测到动作的 381 个参数不参与有效控制路径。不能称有效容量完全相同。'],
      ['个体持续学习','693 个身体后果预测快速参数在线更新；全局动作权重在正式生命中固定。'],
      ['目标','8 个有界槽，按观察到的位置与主要资源增加分类；目标最长 24 事件。表示、分类和完成条件均为设计先验。'],
      ['记忆','64 条环形经验，每条最多 8 个前序输入；回放用当前网络重算前序。记忆淘汰和情境采样规则尚未学习。'],
      ['基础价值','身体资源变化 + 每活跃步 0.02 − 失能事件 1.0。价值来源仍有明确人工规定。'],
      ['资源成本','是仿真内操作成本，不是实际服务器能耗；失能后批量计算可能仍产生占位输出，但不执行环境动作或继续塑性更新。'],
      ['自主性','无逐步任务提示；具备持续状态和经历目标接口。尚无目标机制不可替代、开放式技能积累、语言或意识证据。'],
      ['规则对照','使用自己的观察和正资源结果建立简单位置记忆，无真实地图访问；不是最优控制器。其附带神经预测/目标不控制规则动作。']])+'</section>')
    parts.append('<section><h2>训练、统计和可复现性</h2>'+table(['家族','初始化','选中更新次数','验证存活时长','训练用时'],selected)+'<p>每个训练运行 512 次更新 × 32 事件 × 64 并行个体，共 1,048,576 个计划事件槽位，含失能后的占位事件；两家族共六次训练。用 16 个独立验证世界选择 128/256/384/512 四个保存点中的一个。每个版本在 32 个新正式世界评估；相同世界跨方法配对。</p><p>95% 区间以世界为单位配对重采样 4,000 次，先对三个固定训练版本平均；它们不是对任意训练随机性的总体置信区间。逐版本结果全部保留；辅助与探索比较未作多重校正。六次训练 '+f"{training['seconds']:.2f}"+' 秒，正式评估 '+f"{summary['seconds']:.2f}"+' 秒，压力测试 '+f"{stress['seconds']:.2f}"+' 秒，WUYU_05 诊断 '+f"{diagnostic['seconds']:.2f}"+' 秒；这些分项时间不等于整个工作耗时。</p><p>八项核心检查通过：有限差分核对循环与各输出头梯度、逐个体随机流、反馈前不更新权重、资源不足阻止操作、目标/回放来源严格早于当前事件、观测接口边界、冻结参数对照、完整恢复一致性。独立账本逐步复算正式记录中的位置、资源、噪声观察、奖励与目标来源；'+str(sum(v['events_checked'] for v in audit['physical_ledgers'].values()))+' 个事件槽位通过。三个真实检查点各恢复 384 事件，与原记录和下一检查点全部字段精确一致。</p><p>训练前固定的代码与协议哈希保持一致，原 WUYU_05 源码未修改。05 的 fixed/fixed 参考分支与旧实验前 512 事件逐项相同。</p></section>')
    parts.append('<section><h2>记录修复说明</h2><p>独立核验发现最初有四个 NPZ 文件不完整，缺少 ZIP 结束目录。底层截断原因尚未确定。已用相同选中模型、种子和配置重新执行对应轨迹，所有原始指标逐项一致；没有重新训练或调参。初始不完整字节、前后哈希和恢复日志保留在完整包中。修复后的正式记录通过独立物理账本和实际检查点回放核验。</p></section>')
    parts.append('<section><h2>如何继续运行</h2><p>轻量运行包包含源码、六个选中模型、一个真实生命检查点和报告。完整实验包另含全部逐步数据、训练候选和对照检查点。</p><pre>cd WUYU_06\npython -m pip install -r requirements.txt\npython demo.py --steps 48\npython demo.py --steps 64 --save continued.npz\npython -m unittest discover -s tests -v</pre><p>完整复现顺序：run_train.py → run_eval.py → run_probes.py → diagnose_wuyu05.py → run_stress.py → audit_results.py → make_report.py。已有训练日志会被复用；如需重新训练，请在另一个解压副本中清空 results 后执行。程序运行于执行它的 Python 环境，报告页面本身不会在后台继续运行个体。</p></section>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5ef;color:#203a35;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}.wrap{max-width:1060px;margin:auto;padding:40px 22px 80px}header{padding:38px 0 26px;border-bottom:2px solid #286b60}.eyebrow{font-size:13px;letter-spacing:.15em;color:#58736a}h1{font-size:clamp(30px,6vw,54px);line-height:1.18;margin:18px 0}h2{font-size:24px;line-height:1.4;margin:0 0 18px}section{background:#fff;padding:30px;border-radius:16px;margin:24px 0;border:1px solid #dde5db}p{margin:12px 0}.lead{font-size:19px;max-width:830px;color:#466157}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0 0}.card{padding:18px;border:1px solid #cad9cc;border-radius:12px}.card strong{display:block;font-size:28px;color:#286b60}.card span{font-size:13px}.callout{padding:18px 22px;background:#ecf3ee;border-left:4px solid #286b60;margin:22px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid #dce6df;vertical-align:top}th{background:#edf3ee;white-space:nowrap}td:first-child{font-weight:600}figure{margin:24px 0 10px}figure img{width:100%;height:auto;display:block;border-radius:10px}figcaption{font-size:13px;color:#687b71;margin:8px 0}pre{background:#183d34;color:#e8f1e9;border-radius:12px;padding:20px;overflow:auto;font-size:13px;line-height:1.7}footer{font-size:13px;color:#687b71;margin-top:30px}@media(max-width:600px){.wrap{padding:20px 12px 48px}section{padding:20px 16px}.cards{gap:7px}.card{padding:12px}.card strong{font-size:24px}.lead{font-size:17px}th,td{padding:10px 8px;min-width:100px}}'''
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUYU_06 连续个体实验报告</title><style>'+css+'</style><body><main class="wrap"><header><div class="eyebrow">WUYU · EXPERIMENT 06 · 2026-10-10</div><h1>让一个核心持续行动，<br>也检验它为什么行动。</h1><p class="lead">连续个体原型已运行。关键结果是：内部预测对已训练策略有作用；目标与回放的额外必要性仍未得到证明。</p><div class="cards"><div class="card"><strong>3,386</strong><span>全局神经参数</span></div><div class="card"><strong>3 × 32</strong><span>完整核心正式生命</span></div><div class="card"><strong>1 / 5</strong><span>主效益判据通过</span></div></div></header>'+''.join(parts)+'<footer>所有图表来自随附原始记录。结果针对有限仿真和本轮训练；不能据此声称已经建立数字生命或意识。</footer></main></body></html>'
    (ROOT/'REPORT.html').write_text(document,encoding='utf-8');(ROOT.parent/'WUYU_06_REPORT.html').write_text(document,encoding='utf-8')
    md='# WUYU_06：连续个体实验\n\n共享神经核心、真实仿真资源约束、经历目标、快速预测学习和可恢复连续状态已实现。主指标出现上限；五项收益门槛中仅预测输入干预条件通过。\n\n| 方法 | 存活时长占比 | 身体资源面积 |\n|---|---|---|\n'+'\n'.join('| '+' | '.join(row)+' |' for row in main_rows)+'\n\nHTML 报告包含全部解释、图表、因果探针、两项探索压力测试和 WUYU_05 同状态诊断。所有置信区间对三个固定训练版本条件化，不代表训练随机性的总体不确定性。\n\n关键限制：有限目标表示/期限与记忆管理先验；全局动作权重在正式生命中固定；未验证开放式目标形成、能力积累、人格或意识。\n'
    (ROOT/'REPORT.md').write_text(md,encoding='utf-8')
    (OUT/'final_gates.json').write_text(json.dumps({g:{'description':t,'passed':bool(v)} for g,t,v in gates},ensure_ascii=False,indent=2))
    print(json.dumps({'html_bytes':len(document.encode()),'figures':4,'gates_passed':sum(v for _,_,v in gates)}))
