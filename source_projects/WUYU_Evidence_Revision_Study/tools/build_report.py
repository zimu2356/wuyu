"""Render a self-contained Chinese report from audited frozen results."""
import sys,json,html,base64,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from storage import write_json

ROOT=Path(__file__).resolve().parents[1]
CN={'stable':'始终稳定','local_swap':'局部工具对换','opportunity':'备用通道变好','noise_pulse':'暂时噪声增大','visible_shock':'可见状态冲击','recurrence':'A→B→A：首次改变'}
METHODS=['B0','BE','BU','BEU','BR']

class Report:
    def __init__(self): self.md=[];self.parts=[];self.nav=[]
    def heading(self,title):
        i='section'+str(len(self.nav));self.nav.append((i,title));self.md.append('## '+title+'\n');self.parts.append(f'<h2 id="{i}">{html.escape(title)}</h2>')
    def paragraph(self,s,kind=''):
        self.md.append(s+'\n');escaped=html.escape(s)
        escaped=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',escaped)
        escaped=re.sub(r'`(.*?)`',r'<code>\1</code>',escaped)
        self.parts.append(f'<p class="{kind}">{escaped}</p>')
    def table(self,heads,rows):
        self.md.append('| '+' | '.join(heads)+' |\n| '+' | '.join(['---']*len(heads))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
        self.parts.append('<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    def figure(self,name,caption):
        data=base64.b64encode((ROOT/'figures'/f'{name}.png').read_bytes()).decode()
        self.md.append(f'![{caption}](figures/{name}.png)\n')
        self.parts.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>')

def main():
    c=json.loads((ROOT/'formal_protocol.json').read_text());a=json.loads((ROOT/'holdout/ANALYSIS.json').read_text())
    d=json.loads((ROOT/'development_diagnostics_v2/DIAGNOSTICS_SUMMARY.json').read_text());sup=json.loads((ROOT/'validation/DESCRIPTIVE_SUPPLEMENT.json').read_text())
    proof=json.loads((ROOT/'validation/REPRODUCTION_COMPARISON.json').read_text());audit=json.loads((ROOT/'holdout/AUDIT.json').read_text())
    rows={(r['case'],r['method']):r for r in a['table']};p=a['primary'];ca=a['causal'];r=Report()
    title='WUYU 实验报告：经历的复验、修订与代价'
    r.md=['# '+title+'\n','2026-10-12 · 已完成正式留出、独立重跑与全量审计\n']
    r.heading('结论与选择')
    r.paragraph('**局部修订具有可定位的独立贡献；当前策略仍存在明显的取样延迟和噪声代价。** 本轮完成了原定五方法、六环境、四类机制诊断和正向更新干预。主检验成立，但不同条件给出的选型信号不一致。', 'lead')
    r.paragraph(f'局部工具改变后，组合方法 BEU 的平均身体损失为 {rows["local_swap","BEU"]["post_loss"]:.4f}，单独复验 BE 为 {rows["local_swap","BE"]["post_loss"]:.4f}；配对改善 {p["mean"]:.4f}，95% 区间 [{p["low"]:.4f}, {p["high"]:.4f}]，超过预定最小有用效应 0.02。按两组均值之比，损失降低约 {p["mean"]/rows["local_swap","BE"]["post_loss"]*100:.1f}%。92/128 个体改善，36/128 退化。')
    r.paragraph('稳定世界中 BE 与 BEU 的 128 对完整原始轨迹逐字节相同，稳定代价约束通过。与此同时，随机复验 BR 在局部改变后的平均损失仅 0.1581；噪声世界中 BEU 的损失从 BE 的 0.0199 上升到 0.0427。主检验支持的是一个具体机制的增量作用，通用选型仍需要权衡这些代价。','callout')
    r.table(['本轮判断','证据','实际取舍'],[
        ['修订机制有独立作用','主效应区间下界 0.0373 > 0.02；真实更新阻断/恢复成立','保留 U 的源码与证据，明确其触发条件和适用范围'],
        ['主动取样仍是主要瓶颈','BR 更快取得新样本，512 步内 123/128 确认恢复','后续必须与 BR 比较，并同时约束稳定期代价'],
        ['条件外推需要单独处理','开发诊断中，身体/疲劳外推的条件模型 MSE 明显高于均值模型','优先进行同数据预测与覆盖范围回退比较'],
        ['噪声稳健性尚不足','映射不变的噪声世界仍频繁触发 U，并出现长期失调个体','为噪声、条件外推与真实映射改变建立独立对照'],
        ['暂不扩大形成结论','理想点、动作、预测族、复验和修订规则仍是给定条件','技能、目标内容、评价标准和体验分别保留解释责任']])

    r.heading('实验规模与事先承诺')
    r.paragraph('独立抽样单位为基础种子。正式使用 128 个全新种子 71000–71127，每种子运行 5 方法 × 6 情形，共 3840 条连续生命、9,216,000 个原始动作。每条生命 2400 步，WAIT 同样推进时间。8 个 pilot 种子、32 个完整开发种子和正式种子互不重叠；没有删除失败者、未恢复者或不利情形。')
    r.paragraph('初次改变时点 τ 由独立世界随机流在 800–1200 中抽取；普通代理不知道情形、工具真值、改变时点或评估窗口。唯一主比较为局部对换情形中，τ 至 τ+511 动作后身体平方损失的均值差 Δ=L(BE)−L(BEU)。稳定约束比较全部 2400 步的 BEU−BE，容许增加为 0.002。')
    r.paragraph(f'事先规则从 64、128 中按开发配对标准差选正式 N；开发标准差 {c["calibration"]["development_primary_sd"]:.4f}，即使 N=128，预计区间半宽也为 {c["calibration"]["expected_halfwidths"]["128"]["primary"]:.4f}，高于目标 0.01，因此按资源上限执行并在留出前记录精度不足。正式区间实际半宽约 {(p["high"]-p["low"])/2:.4f}。没有事后追加样本、改窗口、改阈值或挑选算法。')
    r.paragraph('区间由 10,000 次按基础种子成组的 percentile bootstrap 得到。除主比较和稳定约束外，其余区间均为未作多重比较校正的次要或探索性描述。未预先规定跨情形权重，因此没有用一个事后平均分选“总冠军”。')
    r.table(['方法','复验 E','修订 U','新增规则'],[
        ['B0','无','无','上一轮条件模型、累计访问探索与评分'],
        ['BE','有','无','距上次使用时间 /240，封顶加成 0.04'],
        ['BU','无','有','连续、同方向的预测矛盾触发局部旧样本降权'],
        ['BEU','有','有','组合两项机制'],
        ['BR','随机','无','每步 0.07125 概率在全部可用动作中均匀复验']])
    r.paragraph('U 至少需要已有 8 条该动作样本；身体预测残差范数连续 3 次大于 0.04，且相邻方向余弦不小于 0.5，才把较旧样本权重乘 0.1，保留最近两条矛盾样本，再加入本次真实观察。触发后冷却 6 次该动作使用。所有方法每动作最多保存 24 条样本、每步一次模型拟合；U 每次触发另重建至多 24 条样本的统计量。没有额外专家、世界真值或模拟行动预算。')
    r.paragraph('BR 概率按稳定 pilot 中 BE 被年龄偏向改变选择的比例确定，未按改变后胜负选参。总动作数匹配不等于主动工具调用数匹配，下面单列活动比例；另有严格 16 次工具取样的开发诊断。')

    r.heading('六类世界的完整结果')
    r.table(['条件','B0','BE','BU','BEU','BR'],[[CN[case]]+[f'{rows[case,m]["post_loss"]:.6f}' for m in METHODS] for case in c['cases']])
    r.paragraph('表中为变化后 512 步身体平方损失，越低越好。稳定世界使用虚拟对齐时点，不发生物理变化。A→B→A 的第二次改变在 τ+600，因此首次 512 步与局部对换完全相同，这两行不能当作独立重复证据。')
    r.figure('six_conditions','图 1｜五种方法的六情形结果。每个小图纵轴不同；条形仅显示均值，种子层面的差异见图 2。')
    r.paragraph('局部对换同时翻转同一身体维度的正负工具，另一维度不变，改变前后都保留双向调节能力。机会世界从出生起就多一个备用动作，身体收缩系数从 0.005 变为 0.4，其他动作规律不变；额外动作和 24 条样本容量对该情形的所有方法一致。噪声脉冲持续 128 步，标准差 0.003→0.06→0.003。可见状态冲击是 τ 动作后一次外生身体 (+0.45,−0.35) 与疲劳置 0.95，之后条件工具映射仍旧。')
    r.paragraph('备用通道在评估器参照下确实有用：8 个 pilot 的固定单步参照中，允许使用该通道的损失 0.00282，禁用时 0.01004。但正式生命中 B0 已在平均 6.44 步重新尝试它，128/128 都发现了改变；BE 平均 2.97 步，却有更高的持续复验成本。这个机会任务验证了通道的可用性，未形成“基础版长期完全不访问”的困难条件，不能据此确认 E 是发现机会的必要机制。')
    r.paragraph('A→B→A 回归后另一个完整 512 步窗口的损失如下。五方法都没有保存可直接调用的历史情境专家；下表衡量的是现有模型重新适应的代价，不能称为识别旧情境并直接复用。')
    r.table(['回归窗口','B0','BE','BU','BEU','BR'],[['τ+600 起的 512 步']+[f'{a["recurrence_return_loss"][m]:.6f}' for m in METHODS]])

    r.heading('通过主检验以后，仍有哪些代价')
    r.figure('paired_results','图 2｜每个点对应一个独立种子。左图为主比较，中央图为随机复验参照；右图保留所有正负配对差。')
    r.paragraph(f'BEU−BR 在局部改变窗口的探索性配对差为 {sup["local_BEU_minus_BR"]["mean"]:.4f}，95% 区间 [{sup["local_BEU_minus_BR"]["low"]:.4f}, {sup["local_BEU_minus_BR"]["high"]:.4f}]；121/128 个体的 BR 损失更低。然而 BR 在稳定生命中比 BE 平均多 {sup["stable_BR_minus_BE_life"]["mean"]:.5f} 损失，128/128 都更高。这是一项速度与稳定开销的取舍。')
    noise=sup['noise_BEU_minus_BE']
    r.paragraph(f'噪声世界中 BEU−BE 的平均损失增加 {noise["mean"]:.4f}，次要 95% 区间 [{noise["low"]:.4f}, {noise["high"]:.4f}]。每条生命在 τ 后平均触发 {rows["noise_pulse","BEU"]["post_revisions"]:.2f} 次修订，尽管工具规律并未改变。个体中位损失差只有 {noise["median"]:.5f}，均值代价受到少数严重失调个体影响；这些个体全部保留。')
    r.paragraph('主指标只衡量身体偏差，因此同时给出实际疲劳和复合代价 C=身体平方损失+0.08×疲劳²+0.002×活动指示。可见冲击下，B0 的身体偏差更低，但平均疲劳更高；单一身体指标不能替代全部代价判断。')
    r.table(['条件 / 方法','身体损失','平均疲劳','复合代价','主动动作 %'],[[CN[case]+' / '+m,f'{rows[case,m]["post_loss"]:.4f}',f'{rows[case,m]["post_fatigue"]:.3f}',f'{rows[case,m]["post_composite_loss"]:.4f}',f'{100*rows[case,m]["post_active_fraction"]:.1f}'] for case in ('stable','local_swap','noise_pulse','visible_shock') for m in ('B0','BE','BEU','BR')])

    r.heading('恢复速度：取样、模型与行为要分开')
    r.table(['方法','首次复验：均值 / 删失动作数','三条新样本：均值 / 删失动作数','预测达标：均值 / 删失个体','确认行为恢复'],[
        [m,f'{a["latency"][m]["first_use_capped_mean"]:.1f} / {a["latency"][m]["first_use_censored_512"]}/256',f'{a["latency"][m]["three_samples_capped_mean"]:.1f} / {a["latency"][m]["three_samples_censored_512"]}/256',f'{a["latency"][m]["prediction_accuracy_capped_mean"]:.1f} / {a["latency"][m]["prediction_accuracy_censored_512"]}/128',f'{rows["local_swap",m]["recovered"]}/128'] for m in METHODS])
    r.paragraph('单位为改变后的原始步数，所有等待值封顶 512。前两项对每个种子的两个相关动作逐个统计，因此分母 256；独立推断单位仍为 128 个种子。预测达标要求两个相关工具在预定匹配身体/疲劳和两材料条件下 MSE 均 <0.0025，记录首次跨越，不要求之后持续成立。行为恢复要求连续两个 64 步窗平均损失均不超过 max(0.02, 变化前均值×2)，报告确认完成时刻；不达标者按 512 右删失。')
    r.paragraph(f'BEU 第一次相关模型修订的平均封顶延迟为 {a["latency"]["BEU"]["revision_capped_mean"]:.1f} 步，128/128 在窗口内触发。触发修订并不保证预测或行为已经恢复：BEU 仅 27/128 在主窗口确认行为恢复，BE 为 34/128，BR 为 123/128。BEU 降低了平均过程损失，却没有在这个严格恢复标准上超过 BE。')
    r.figure('aligned_trajectories','图 3｜围绕改变时点对齐的全体平均轨迹；显示采用向后 32 步平滑，分析仍使用原始数据。回归情形的第二条竖线为旧规律返回。')
    r.paragraph('局部改变后的长期结果也必须保留：最后 400 步，B0、BE、BEU 的平均损失已分别降至 0.00955、0.00927、0.00916。主要困难是较长的适应过程和部分条件中的失调，不能把前 512 步表现概括为整个生命都无法恢复。')
    r.paragraph('第 τ+512 步的实际状态探针中，BEU 评分第一动作的预测 MSE 为 0.00048，其余动作平均 0.02121；BR 相应为 0.000092 与 0.000015。这里“评分第一动作”不包含 BR 随机覆盖，也不是额外真实取样；两类动作的覆盖条件不同，不能把误差差异直接解释为模型优越性。所有已实际选择动作的更新前残差亦保存在原始轨迹及图表数据中。')

    r.heading('四种失败解释的开发诊断')
    r.paragraph('以下只使用 32 个开发种子，起点统一为 B0 局部改变后的 τ+128 检查点。它们承担问题定位，均不冒充新的正式留出主检验。')
    r.figure('mechanism_diagnostics','图 4｜取样、给定数据下更新、预测族和正确单步预测的四类诊断。真值参照只存在于评估副本。')
    eq=json.loads((ROOT/'figures/EQUAL_SAMPLING_TOTAL_COST.json').read_text())
    modes=['policy_ranked_tools','round_robin','oldest'];labels=['按现策略给工具排序','轮流覆盖四工具','优先最久未用工具']
    r.table(['同量取样规则','取样后 128 步损失','全部 192 步含取样代价'],[[label,f'{d["sampling"][mode]:.4f}',f'{np.mean([x["all_192_step_loss"] for x in eq if x["mode"]==mode]):.4f}'] for label,mode in zip(labels,modes)])
    r.paragraph('三组都真实执行 16 次工具动作和 48 次 WAIT，再自由行动 128 步，总计 192 步。“按现策略排序”也必须从工具中选择，因此它不是允许任意 WAIT 的完整原策略。不同选择会产生不同身体状态与经历；该比较识别取样安排的作用，不是同数据拟合比较。全预算结果同样支持覆盖的重要性，但没有把任何组解释为智能探索的普遍最优。')
    r.paragraph(f'对完全相同的 64 条按时序记录的真实转移，FIFO 的更新前预测 MSE 为 {d["same_data_mse"]["FIFO"]:.6f}，固定 64 步半衰期的时间降权为 {d["same_data_mse"]["calendar_decay"]:.6f}，局部修订为 {d["same_data_mse"]["local_revision"]:.6f}。简单时间降权在这条数据流上更好；它只是固定数据诊断，尚未作为完整自主策略验证。')
    r.table(['同一模型样本，评价输入类别','条件模型 MSE','均值模型 MSE'],[[label,f'{d["predictor_before"][group]["conditional_mse"]:.5f}',f'{d["predictor_before"][group]["mean_mse"]:.5f}'] for group,label in [('recorded_support','记录中出现过的输入'),('material_interpolation','身体/疲劳历史中位数，材料 0.5'),('body_fatigue_extrapolation','身体/疲劳外推'),('boundary','接近身体边界')]])
    r.paragraph('“出现过的输入”不保证其旧输出仍符合改变后的规律；材料插值也不能保证联合身体条件在训练支持内，其平均最近样本距离约 0.302。这里使用的覆盖距离和有效样本量都是描述性指标，未经校准为概率置信度。均值模型在外推条件上更稳，并不等于在所有已覆盖状态下都更好。')
    r.paragraph(f'在同一检查点副本保留原有单步决策和探索项，仅用评估器的无噪声单步后果替换预测，随后 128 步损失从 {d["coordination"]["learned"]:.4f} 降至 {d["coordination"]["truth"]:.4f}。这说明在这些副本中改善预测已足以带来大幅改善，暂未出现必须用更深规划才能解决的证据。该参照有世界真值特权，非公平学习基线，也不是全局最优；边界附近的无噪声截断结果不等于随机结果的期望。')

    r.heading('正向链条：一次真实新经历怎样影响选择')
    r.paragraph('对每个正式 BEU 个体，固定选择改变后第一次相关工具模型修订。复制实际动作后的同一世界状态，比较正常接受新经历、记录经历但阻断对应模型及其检测状态更新、阻断无关模型的假干预、恢复对应模型更新。主生命继续运行，干预只发生在副本；各副本随后都有 64 次真实行动预算。')
    r.table(['干预','随后 64 步平均损失'],[[{'normal':'正常更新','block_target':'阻断对应更新','block_unrelated_sham':'无关模型假干预','restore':'恢复真实更新'}[name],f'{value:.6f}'] for name,value in sup['causal_group_means'].items()])
    ce=ca['conditional_effect']
    r.paragraph(f'128/128 个体都产生了预定事件。阻断−正常的平均差为 {ce["mean"]:.4f}，次要 95% 区间 [{ce["low"]:.4f}, {ce["high"]:.4f}]；71 个体阻断后变差，22 个体变好，35 个体相同。31/128 的第一步选择立即改变。恢复分支与正常分支在全部 128 个体上完整状态和事件精确一致。')
    r.paragraph('这条证据链支持“某段实际新证据经具体模型更新参与后续行为”。事件按首次相关修订选择，不能推广为任意经历都有效。无关模型分支之所以相同，是因为本架构每步只更新被选模型；它是恢复操作的假干预检查，不能独立证明复杂心理层面的任务选择性。平均有益也不能掩盖 22 个体中的有害更新。')

    r.heading('接回物欲：哪些形成责任已经推进')
    r.table(['问题','本轮已获得','仍然欠缺'],[
        ['经历如何影响当前行为','新证据、局部权重改变、选择和反事实损失的可追溯链条','跨任务、跨材料族和更长生命的适用范围'],
        ['如何修订旧判断','主动取样与局部更新有可分离的功能效果','可靠区分噪声、条件外推、映射改变与旧规律回归'],
        ['能力如何形成','学习给定原始动作的后果','动作序列的发现、复用、组合与预算匹配收益'],
        ['新追求如何形成','尚未改变固定理想点和目标类别','新可达结果进入选择，及其评价来源'],
        ['关系、标准与体验','本轮没有相应实验','特定互动、评价标准变化、体验与归属需独立论证']])
    r.paragraph('下一步优先做一个有明确边界的预测与取样研究：在相同数据、相同样本容量下比较条件模型、简单均值和按覆盖范围选择的回退；完整自主运行必须同时与 BR 比较，检查稳定成本、噪声脉冲和恢复时间。U 的噪声代价需要专门约束。当前数据没有支持直接增加更多历史专家或更深规划作为首要修复。')
    r.paragraph('有上限的年龄加成可能无法压过过时模型的强烈负面评分，这是从源码与长取样延迟得到的待检验解释；本轮没有单独操纵加成上限，不能视为已经证实。固定数据中时间降权的优势同样需要新的自主收集检验。完成这项具体取舍后，再进入有持续外部成果、可复用动作序列和新可达结果的小工作坊，将工程修复接回能力与目标内容的形成。')

    r.heading('完整性、复现与已知边界')
    r.table(['核验项目','结果'],[
        ['机制测试','13 项通过；含单次误差不触发、局部修订、状态隔离、可达性、未知机会与快照恢复'],
        ['旧算法兼容','B0 与上一轮冻结代码 600 步动作/状态相同，预测数值容差内一致'],
        ['跨进程续接','第 1300 步起，50+50 与 100 步的全部事件及最终状态相同'],
        ['开发主轨迹','960 条生命、2,304,000 条事件、4,960 个最终模型通过审计'],
        ['正式主轨迹',f'{audit["lives"]:,} 条生命、{audit["events"]:,} 条事件、{audit["final_models"]:,} 个最终模型通过审计'],
        ['干预恢复','开发 32 对、正式 128 对状态/分支精确恢复'],
        ['独立复现',f'{proof["total_identical_files"]:,} 个原始与分析文件逐字节一致；包括全部 3,840 份正式事件文件'],
        ['源文件与协议','正式前冻结，运行期间未改变；最终加权统计量最大审计残差约 5.24×10⁻¹⁴']])
    r.paragraph('开发期间有两项明确记录的修复：可见冲击动作索引原先会污染变化前窗口，已在完整开发前更正并重做可达性检查；完整开发命令漏传中间检查点开关，诊断首次失败后用 32 条完整 B0 生命逐步重放、逐值比对并验证最终状态哈希，恢复诊断检查点，未改写原轨迹。开发重定向进度日志只保留前 3 行，完成情况由 32 个块回执、COMPLETE 和原始文件清单确认。全部失败日志、原版本源码快照和修复说明均保留。')
    r.paragraph('本轮正式数据没有发生缺尾或按哈希恢复事件。独立复现使用同一冻结算法的独立进程；额外审计独立表达逐步物理和加权正规方程，但这仍不是由另一团队独立实现整个理论。数值逐字节一致限于本运行环境，不承诺跨不同 BLAS 平台恒定。图表已逐张检查并修正一次绘图类别顺序错误；HTML 为自包含报告，完成结构与资源检查，未做浏览器截图级验证。')
    r.paragraph('报告中的正式主结果来自 `holdout/ANALYSIS.json`；全部逐种子结果在 `summaries.json`，完整状态和事件在成对 `.json.gz` / `.npy.gz`；开发诊断在 `development_diagnostics_v2/`。`README.md` 给出重跑和继续生命命令，`FORMAL_FREEZE.json`、`MANIFEST.json`、`AUDIT.json` 与复现比较文件提供校验入口。完整包保留所有不重复原始数据；独立复现的同哈希大文件在归档中省略重复副本，保留完整清单与核验结果。')
    r.paragraph('原有 01–08、Integrated、LAAP 源码定位与核验边界沿用 `references/WUYU_SOURCE_MAP_2026-10-12.md`；本轮完整源码是独立研究项目，并未宣称把所有旧阶段整合成新的总架构。理论第二版原文随参考材料保留，没有据此实验改写或扩大理论结论。')

    style='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f7;color:#17263a;font:16px/1.85 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}header{background:#132d3a;color:#fff;padding:52px max(24px,calc((100vw - 1100px)/2)) 38px}header .eyebrow{color:#93d7cc;font-size:13px;letter-spacing:.15em}h1{font-size:clamp(27px,3.2vw,43px);line-height:1.35;margin:14px 0}header p{color:#d5e3e8;margin:12px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:28px}.stat{border-top:1px solid #527482;padding-top:12px}.stat b{display:block;font-size:26px;color:#adede0}.stat span{font-size:13px;color:#d1e0e7}nav{max-width:1100px;margin:20px auto;padding:0 18px;display:flex;flex-wrap:wrap;gap:8px}nav a{color:#236b75;text-decoration:none;font-size:13px;border:1px solid #ccdce1;border-radius:20px;padding:4px 12px;background:#fff}main{max-width:1140px;margin:auto;padding:12px 20px 60px;background:white}h2{font-size:25px;line-height:1.4;border-top:1px solid #d9e4e7;padding-top:30px;margin:38px 0 20px;color:#174b55;scroll-margin:20px}p{margin:18px 0}.lead{font-size:19px}.callout{border-left:4px solid #cb8a33;background:#fff6e7;padding:18px 22px;border-radius:0 6px 6px 0}.tablewrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{text-align:left;background:#eaf2f3;color:#20515a;padding:12px 13px;white-space:nowrap}td{border-bottom:1px solid #e2e9eb;padding:12px 13px;vertical-align:top}tr:nth-child(even){background:#f8fafb}td:not(:first-child){min-width:82px}figure{margin:30px 0;padding:14px;background:#fff;border:1px solid #dce5e7;border-radius:8px}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#596d78;margin:10px 8px}code{font-size:.88em;background:#edf2f5;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}footer{max-width:1100px;margin:0 auto 36px;padding:18px;color:#61737d;font-size:13px}@media(max-width:650px){body{font-size:15px}.stats{grid-template-columns:1fr 1fr}main{padding:10px 15px}h2{font-size:22px}figure{padding:4px}td,th{padding:9px}header{padding:30px 22px}}@media print{body{background:white}header{background:white;color:#132d3a;padding:20px 0}header p,.stat span{color:#334}.stat b{color:#18565a}nav{display:none}main{padding:0}h2{break-after:avoid}figure,tr{break-inside:avoid}}'''
    nav=''.join(f'<a href="#{i}">{html.escape(title)}</a>' for i,title in r.nav)
    header=f'<header><div class="eyebrow">WUYU / EVIDENCE REVISION STUDY / 2026.10.12</div><h1>{title}</h1><p>局部贡献得到验证，稳健适应仍需处理取样、预测与代价。</p><div class="stats"><div class="stat"><b>128</b><span>独立留出种子</span></div><div class="stat"><b>5 × 6</b><span>方法与环境</span></div><div class="stat"><b>+0.0504</b><span>主比较配对改善</span></div><div class="stat"><b>7,811</b><span>文件逐字节复现</span></div></div></header>'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style></head><body>'+header+'<nav aria-label="报告目录">'+nav+'</nav><main>'+''.join(r.parts)+'</main><footer>原始动作预算与全部失败个体均保留 · 给定条件、获得的功能证据与未解决的形成责任分别记录。</footer></body></html>'
    stem='WUYU_Evidence_Revision_Report_2026-10-12'
    (ROOT/(stem+'.md')).write_text('\n'.join(r.md));(ROOT/(stem+'.html')).write_text(page)
    write_json(ROOT/'validation/REPORT_SOURCE_NUMBERS.json',dict(primary=p,primary_pass=a['overall_pass'],n=128,steps=9216000,replicated_files=proof['total_identical_files']))
    print(json.dumps(dict(html_bytes=len(page.encode()),markdown_bytes=(ROOT/(stem+'.md')).stat().st_size,sections=len(r.nav),images=4)))

if __name__=='__main__': main()
