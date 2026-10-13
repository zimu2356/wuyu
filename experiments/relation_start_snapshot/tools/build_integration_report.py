"""Build self-contained reports directly from the saved acceptance results."""
from pathlib import Path
import base64
import html
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/integrated_effects'
NAMES={'stationary':'固定世界','recipe_change':'局部配方变化','developmental':'完整发展环境'}

def main():
    result=json.loads((OUT/'SUMMARY.json').read_text());audit=json.loads((OUT/'AUDIT.json').read_text())
    diag=json.loads((OUT/'POSTHOC_MECHANISM_DIAGNOSTIC.json').read_text())
    assert audit['passed'] and result['runs']==768
    table=[]
    for case in NAMES:
        m=result['means'][case];d=result['paired_vs_prior'][case]['combined']['burden']
        table.append([NAMES[case],f"{m['prior']['final_alive']*32:.0f}/32",f"{m['combined']['final_alive']*32:.0f}/32",f"{m['online']['final_alive']*32:.0f}/32",
                      f"{d['mean']:+.6f}",f"[{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]"])
    def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    main_table=mdtable(['场景','原路径存活','完整组合存活','仅证据预测存活','组合−原路径身体匮乏','描述性 95% 区间'],table)
    paragraphs=[
      ('结论','源码与状态整合完成，兼容机制已经在同一个连续个体内共同运行，基础能力保持；“组合后整体更强”没有得到支持。完整组合通过本轮预设的不退化筛选，但没有新增生存优势。最终“由物理欲望形成的人”的目标仍未达成。'),
      ('本次整合究竟做了什么','覆盖物欲 01–08、此前整合工程、目标形成、自主探索、持续经历和证据复验/修订四轮研究。16 个来源包合并为 12,373 个原内容文件，291 个 Python 源文件及 JavaScript 全部保留；其中 01–08 有 115 个 Python 文件。补全早前整合缺少的 06–08 模型、检查点与记录。13 套历史运行系统加一个组合候选，共 14 个统一入口。历史重复副本保留来源，文件数不是独立机制数量。'),
      ('机制与运行逻辑','组合候选以 08 图网络和 07/08 关联/技能记忆组织动作，以 03–06 延续的私有循环后果学习及前缀回放保存经验，加入新实验的有限证据窗、动作年龄复验 E 和局部证据降权 U。全部围绕同一身体、库存、历史、时钟和私有随机流。先看见当前状态、作预测、选唯一原始动作，再接受实际反馈并更新；WAIT 也付出身体成本。存档、恢复与诊断分支不能改变主个体经历。'),
      ('不是所有旧模块都被强装','01 的学习率门控、02 的 Q 宏动作、04B 的专家路由、06 的目标池/身体付费回放仍有完整独立入口，没有伪称已在工坊共同起效。后两轮关于持续状态与证据的约束进入主循环；目标/探索实验的路径搜索和身体重置不冒充连续神经行动。目标缓冲在既有实验中未显示独立贡献，故没有默认新增一个“欲望模块”。具体对照见 docs/MECHANISM_INTEGRATION.md。'),
      ('如何验收','先保持原语义，再检查组合后行为。131 项历史测试与 22 项新测试全部通过，共 153 项。原证据修订实验 seed 71000 的五方法×六场景，共 72,000 步，每个原始事件字段与最终完整状态均精确一致。13 套历史测试没有重新训练历史模型；这一回归也不冒充全部 128 种子的重新独立实验。全部 12,373 个原文件重新哈希，无缺失、无改写。'),
      ('行为验收的边界','运行前固定 32 个种子、8 个条件、3 个场景、每条生命 1,536 步，共 768 次运行、1,179,648 步。四个已训练初始化各分配八个世界，不是完整交叉试验。全部死亡与失败保留；bootstrap 单位是种子，不把步数当样本。完整发展环境同时包含依赖变化、配方变化及身体冲击；另有固定世界和只改变配方的控制。候选额外计算单列，未匹配计算预算，也未向身体收取计算费用。'),
      ('怎样解读不退化通过','预设筛选为每个场景组合−原路径的存活比例差 95% 下界大于 −0.02，身体匮乏差 95% 上界小于 +0.02。三场景全部通过。该工程容许界不是严格等效或普遍优越证明，区间均为描述性、未作多重比较校正。身体匮乏是两项 (1−储量)² 之和，死亡吸收态记 2，越低越好；不能与后续实验以零为理想点的身体损失直接比较。'),
      ('真正有作用的是哪些机制','完整发展环境中，保留全部关联/效应/技能学习的组合存活 31/32；从出生冻结这一整组学习后为 0/32，平均取物从 348.34 降到 29.19。说明在这套环境与已训练网络下，这组在线经验机制具有实际作用；没有分开证明组内每一种机制不可缺少。固定世界和局部变化中的冻结消融也明显变差。'),
      ('新增模块没有自动带来更强能力','组合候选确实在学习：完整发展环境每个体平均证据更新 1,505.09 次、神经更新 3,009.19 次（包含 1,504.09 次回放）、局部修订 1.94 次。但在同一实际状态与同一随机抽样下，候选偏置平均只改变 2.31/1,536 次动作。只用证据预测的轻组合在该环境为 32/32 存活，完整组合与原路径均为 31/32，差别来自 seed 920004 一例，不足以宣布轻组合普遍胜出。固定混合是否削弱有效预测仍需定位，不能据此断言神经回放有害。'),
      ('修订触发不等于有效适应','事后逐轨迹诊断显示：online 与仅增加 U 的 update，在全部 96 个场景×种子配对中动作完全相同。E 与 E+U 在固定及局部配方变化的 64 对中也完全相同；完整发展环境有 2/32 对改变轨迹。U 可以真实改变模型而未跨过动作选择边界。原证据修订研究中的成功，不能直接移植为工坊中的适应增益。此分析明确属于事后诊断，没有修改预设验收口径。'),
      ('以前的四轮实验怎样影响这次判断','目标形成与自主探索实验支持给定调节准则下的学得结果选择、经历覆盖与有限预算探索；删除目标缓冲未改变行为。自主采集中的身体仍按协议每步重置。持续经历实验建立了不断档状态和有选择的记忆作用，也暴露过时经历会妨碍变化后的行动。证据修订正式研究在局部对换中给出 E+U 相对 E 的有限改善，但随机复验对照更强；这些正反结果均保留，不把“后来”自动当成“更好”。'),
      ('哪些预期已达到，哪些尚未达到','已达到：全量来源可追溯，旧实验语义和结果保持，暂停恢复可精确续接，兼容机制共同运行，历史学习能改变行为，本次有限环境中基础能力未被组合破坏。未达到：组合机制稳定带来额外收益；可靠新长链组织；生命期内图网络行动权重持续学习；自主生成新目标类型或新价值标准；社会关系、反思性自我改变、主观体验或“人”的形成。当前愿景不能用 153 项测试通过来代替。'),
      ('这次整合暴露的具体问题','身体缺口评分与理想点仍是给定的；两类预测器的表示、历史窗口与遗忘速度不同，固定混合没有可靠性依据。局部阶段差仍混入换位置看到另一设备阶段的变化，不能称为任务进展。E 只读动作年龄，覆盖距离尚未成为校准后的决策依据。图网络动作先验很强，新证据很少实际改变动作；目标、选择、学习之间仍有需要查清的连接。'),
      ('下一步的约束','整合已完成后，后续研究应从这些具体缺口决定方向。保留 prior 为稳定参照，combined 为可复核候选；不因模块齐全而提升默认地位。优先比较同一历史下预测的准确性及其改变动作的条件，再检查缺少前置物时能否保持目标并组织替代路径。本轮没有继续调权重、增加模块或扩大模型来追求一次胜利，先交付全部结果与可续接状态。'),
      ('复核入口','README.md 提供运行、恢复、分支、原实验重跑和数据核验命令。provenance/SOURCE_IMPORT_MANIFEST.json 记录每个原文件的来源和 SHA256；validation/integrated_effects/ 保存事前协议、源码冻结、768 条完整原始轨迹、全部种子指标、24 个完整存档、汇总和审计。审计核对全部 1,179,648 行及文件哈希，24 个代表存档和 36,864 步物理重演通过；没有身体重置或死亡后复活。Source 包能独立运行默认系统与全部测试；全量历史大文件在 Data 包，解压到同一父目录。')]
    text='# 物欲整合报告：代码、机制、运行逻辑与实际效果\n\n2026-10-12 · 1–8 + 后续四轮实验\n\n'
    for title,body in paragraphs:
        text+=f'## {title}\n\n{body}\n\n'
        if title=='行为验收的边界':text+=main_table+'\n\n'
    text+='## 全条件结果\n\n'
    all_tables=[]
    for case in NAMES:
        rows=[]
        for mode,r in result['means'][case].items():rows.append([mode,f"{r['final_alive']*32:.0f}/32",f"{r['burden']:.6f}",f"{r['deliveries']:.2f}",f"{r['blocked']:.2f}"])
        t=mdtable(['条件','最终存活','身体匮乏','平均取物','平均受阻取物'],rows);all_tables.append((NAMES[case],rows));text+=f'### {NAMES[case]}\n\n'+t+'\n\n'
    (ROOT/'WUYU_Integration_Report_2026-10-12.md').write_text(text)
    def htable(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.5),layout='constrained')
    modes=['prior','online','combined','combined_no_memory'];labels=['Prior','Evidence only','Combined','Memory learning off']
    axes[0].bar(labels,[result['means']['developmental'][m]['final_alive']*100 for m in modes],color=['#718096','#258a74','#335c9b','#b85b4a'])
    axes[0].set_ylim(0,112);axes[0].set_ylabel('Final survival (%)');axes[0].set_title('Developmental world / 32 seeds');axes[0].tick_params(axis='x',labelsize=8)
    for i,m in enumerate(modes):axes[0].text(i,result['means']['developmental'][m]['final_alive']*100+2,f"{result['means']['developmental'][m]['final_alive']*32:.0f}/32",ha='center',fontsize=9)
    for i,case in enumerate(NAMES):
        d=result['paired_vs_prior'][case]['combined']['burden'];axes[1].errorbar(d['mean'],i,xerr=[[d['mean']-d['ci95'][0]],[d['ci95'][1]-d['mean']]],fmt='o',color='#335c9b',capsize=4)
    axes[1].set_yticks(range(3),['Stationary','Recipe change','Developmental']);axes[1].axvline(0,color='#87919f',lw=1);axes[1].set_title('Combined minus prior / body burden');axes[1].set_xlabel('Paired mean and descriptive 95% interval')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    figure=ROOT/'docs/integration_effects.png';fig.savefig(figure,dpi=160);plt.close(fig)
    body='<header><div class="eyebrow">WUYU · 2026.10.12 · INTEGRATION 1.0</div><h1>物欲的整合，必须由实际运行来验收</h1><p class="lead">基础能力保持。机制已接通。整体增益尚未得到支持。</p></header>'
    body+='<div class="cards"><div><b>12,373</b><span>原文件完整保留</span></div><div><b>153</b><span>正确性测试通过</span></div><div><b>768</b><span>组合行为验收生命</span></div><div><b>1,179,648</b><span>实际验收事件</span></div></div>'
    for title,para in paragraphs:
        body+='<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(para)+'</p>'
        if title=='行为验收的边界':body+=htable(['场景','原路径存活','完整组合存活','证据预测存活','匮乏差','95% 区间'],table)
        if title=='新增模块没有自动带来更强能力':body+='<img alt="完整发展环境的存活比较，以及组合相对原路径的身体匮乏差异" src="data:image/png;base64,'+base64.b64encode(figure.read_bytes()).decode()+'"><p class="caption">按种子统计；32 个种子，四个固定训练初始化。区间为描述性，不是多重比较校正后的优越性结论。</p>'
        body+='</section>'
    body+='<details><summary>展开全部 8 条件 × 3 场景的结果</summary>'
    for case,rows in all_tables:body+='<h3>'+case+'</h3>'+htable(['条件','最终存活','身体匮乏','平均取物','平均受阻取物'],rows)
    body+='</details><footer>依据已保存源码、协议、原始轨迹与审计生成。历史研究的结论与新组合验收分别保留。</footer>'
    css='body{margin:0;background:#f1f3f5;color:#233044;font:16px/1.8 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1020px;margin:32px auto;background:white;padding:44px 56px;border-radius:16px}h1{font-size:34px;line-height:1.4;margin:16px 0}h2{font-size:22px;margin:32px 0 10px}h3{font-size:19px}.eyebrow{font:12px/1.5 monospace;letter-spacing:1px;color:#536783}.lead{font-size:20px;color:#35577e}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}.cards div{padding:16px;background:#edf2f7;border-radius:8px}.cards b{display:block;font-size:25px;color:#2c507b}.cards span,.caption{font-size:13px;color:#607086}p{margin:10px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 8px;border-bottom:1px solid #dae1e9;text-align:left;white-space:nowrap}th{background:#edf2f7}.scroll{overflow:auto}img{width:100%;height:auto;margin-top:16px}summary{cursor:pointer;font-weight:600}details{margin-top:30px;border-top:1px solid #dae1e9;padding-top:20px}footer{font-size:12px;color:#607086;margin-top:40px}@media(max-width:700px){main{padding:24px 18px;margin:0;border-radius:0}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}}'
    (ROOT/'WUYU_Integration_Report_2026-10-12.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物欲整合报告</title><style>'+css+'</style><main>'+body+'</main></html>')
    print(json.dumps({'reports':2,'figure':str(figure),'gate':result['all_cases_cleared']},ensure_ascii=False))
if __name__=='__main__':main()
