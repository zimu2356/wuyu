"""Create a standalone, mobile-readable Chinese report from actual results."""
from pathlib import Path
import json,html,re
ROOT=Path(__file__).resolve().parents[1]


def main():
    s=json.loads((ROOT/'results/summary.json').read_text())
    d=json.loads((ROOT/'results/prediction_diagnostic.json').read_text())
    q=json.loads((ROOT/'results/dependency_diagnostic.json').read_text())
    audit=json.loads((ROOT/'validation/source_audit.json').read_text())
    m=s['means'];ci=s['paired']['replay']['deliveries']['ci95']
    text=f'''# 物欲：统一工程、真实结果与下一步

2026-10-11 · 确认范围 WUYU_01—08 · 源码整合与可复现实验

**整合工程已经可运行；新的学习机制尚未证明比原有动作系统更好。** 本次把八个版本放进统一的运行、保存、恢复和来源体系，并接通了一条真实的“观察—行动—后果—私有神经学习—回放”路径。实验结果要求我们保留这个候选机制，但不能把它升级成最终模型底座。

## 1. 本次实际完成了什么

八个版本的源文件及恢复到的历史产物保存在独立目录，并逐文件与原始压缩包核对。八个原包的校验全部一致，{sum(r.get('unchanged_members',0) for r in audit['archives'])} 个原始文件成员保持原内容；另生成排除可再生缓存的逐文件清单。源码的来源可追溯，原来的实验结论没有被新实验覆盖。

八个版本都有同一套 step/save/load 入口。01—03 原本在对象之外的世界流、时间游标等信息也进入统一存档。01—08 原测试合计 65 项通过；新增整合测试 15 项通过，共 80 项。测试包括连续运行与保存恢复结果相同、分支互不污染、个体取出与重排不改变后续随机流、预测先于反馈、回放只使用实际过去、私有预测器梯度，以及旧 08 个体迁移后的贪心动作一致性。这些是工程验证，不能当作智能水平评分。

新主闭环沿用 08 的图神经动作网络、物理世界和 07/08 的经验学习记忆。早期私有循环参数学习、重算前缀回放、行动后果预测经过明确改写接入同一个个体。新代码不包含用于导航的教师规划方法；原教师保存在旧版训练和诊断代码中。

修复了两个跨版本问题：特征编码不再顺带写入记忆；每个个体独立拥有随机状态。现在重复查看特征不会重复增加经历，一个个体离开批次也可以稳定续接自己的未来。

提供了 08 到新运行层的实际迁移工具，并交付从事件 384 继续到 448 的例子。迁移保留世界、身体、图网络、循环状态和已有记忆；新预测器从空白开始。旧批随机流改成私有流，因此迁移不是旧随机未来的逐位复制；这项变化明确写入来源信息。迁移之后的保存恢复则通过了精确验证。

## 2. 八个版本如何归位

| 来源 | 统一工程中的去向 | 尚未声称完成的部分 |
|---|---|---|
| 01 | 价值记忆与学习率实验保留为可运行档案 | 未把自适应门控默认接入新身体闭环 |
| 02 | 工具消耗、维护、Q 学习可独立运行并完整续接 | 宏动作和 Q 表不强行改称 08 的原始动作 |
| 03 | 私有循环学习原则接入新预测器；旧感知与控制实验保留 | 新旧输入及权重不兼容，未进行虚假的权重拼接 |
| 04 | 重算前缀回放接入；隐情境、主动观测保留独立世界 | 隐情境专家路由尚未在新身体世界中共同工作 |
| 05 | 独立随机状态、实际过去的回放、门控对照经验吸收 | 复杂学习率门控没有被当成必需器官 |
| 06 | 身体后果与动作的联系接入 | 经验目标池和用身体成本支付的回放仍在旧实验中 |
| 07 | 经验模型、动作技能保留；新运行类去掉教师规划 | 未宣称独立技能模块在所有任务中有增益 |
| 08 | 图网络与身体行动成为当前默认主闭环；修正读写和随机状态 | 仍不能可靠处理新增的嵌套依赖 |

完整细目见 docs/MECHANISMS.md。这里的“整合”意味着统一工程、清晰的模块边界、可复现的主闭环和保留下来的对照。它不意味着每个历史模块都已经在一个个体中共同产生了收益。

## 3. 新主闭环究竟学什么

每个事件先记录一次可观察事实。固定的图网络输出七个原始动作的概率；私有循环预测器同时预测各动作的短期身体、库存、局部阶段和取物后果。选定动作并执行后，个体才获得监督信号。它只能更新实际执行动作的后果标签，不能看见未执行动作的真实结局。

启用在线学习时，预测器更新自己的输入、循环和输出权重。启用回放时，从更早完成的实际经验中抽样，用当前权重重算最多四步前缀，再学习该过去事件。此处从零状态开始重算短前缀，是截断近似，不等于完整重演一生。当前事件最后才入库。

实验模式用声明的身体需要权重给预测后果打分，再小幅调整动作概率；没有搜索完整路径。默认图网络约 11,345 个参数，经验模型 168 个参数，新预测器 1,462 个参数，总计约 12,975 个；这个数量不包含记忆库、循环状态和优化器状态。它仍是很小的机制试验，离 0.几 B 语言模型的训练任务很远。

**目前可塑的是经验模型与私有后果预测器。图网络的动作组织权重在本次个体生命期间保持固定。** 因此不能说物欲已经学会持续改写自己的全部行动逻辑。

## 4. 正式配对实验：接通不等于有益

事先固定协议后，复用 08 的三个已训练初始化，使用 16 个新世界；每个模型—世界组合运行四个条件，每次 768 个事件。共 192 次运行、147,456 个模拟事件。条件之间共享初始模型、世界种子与私有随机流。世界分四阶段，使用原冲击规则；这是新的较短测试，不能把本表与旧版 1,536 步生命试验直接比较。

| 条件 | 存活时间比例 | 最终存活率 | 平均取物次数 | 每个体平均预测器更新数 |
|---|---:|---:|---:|---:|
| 原动作路径 prior | {m['prior']['survival']:.3%} | {m['prior']['final_alive']:.3%} | {m['prior']['deliveries']:.3f} | {m['prior']['updates']:.1f} |
| 在线预测影响动作 online | {m['online']['survival']:.3%} | {m['online']['final_alive']:.3%} | {m['online']['deliveries']:.3f} | {m['online']['updates']:.1f} |
| 在线预测加回放 replay | {m['replay']['survival']:.3%} | {m['replay']['final_alive']:.3%} | {m['replay']['deliveries']:.3f} | {m['replay']['updates']:.1f} |
| 只学习、不影响动作 shadow | {m['shadow']['survival']:.3%} | {m['shadow']['final_alive']:.3%} | {m['shadow']['deliveries']:.3f} | {m['shadow']['updates']:.1f} |

四条件的存活时间完全相同。shadow 与 prior 的全部动作一致，符合“不接入动作就不改变行为”的预期。online 与 replay 分别有 {d['changed_actions']['online']} 和 {d['changed_actions']['replay']} 个时刻的动作不同于 prior，说明学习结果确实能进入行为；但动作发生变化并没有带来本试验的生存收益。

replay 平均少取物 0.125 次，世界聚类配对 bootstrap 的 95% 区间为 [{ci[0]:.3f}, {ci[1]:.3f}] 次。绝对量很小、比较不止一个、世界只有 16 个，不能扩大成“回放普遍有害”的结论。同一个世界在三个模型上重复出现，统计先在模型间平均，再以世界为聚类单元；没有把 48 个模型—世界组合伪装成 48 个独立世界。

这里的存活率已接近上限，指标有天花板效应。因此本实验只能说“未显示收益”，不能证明在线学习没有价值。它也没有检验公平计算预算：回放额外消耗计算，当前模拟没有向身体扣除这笔计算成本。

## 5. 更严格的反证：简单基线反而更好

为了排除“只要误差下降就是聪明起来了”，追加了一个探索性诊断：在 prior/shadow 完全相同的经历上，与只按动作累计平均后果的预测器比较。这个均值预测器只使用已经发生的反馈，不偷看当前标签；它不用位置、情境或神经网络。

| 同历史预测方法 | 行动前平均均方误差，越低越好 |
|---|---:|
| 始终预测零变化 | {d['mean']['zero_mse']:.6f} |
| 新循环预测器与回放（shadow） | {d['mean']['shadow_mse']:.6f} |
| 按动作累计过去后果均值 | {d['mean']['action_mean_mse']:.6f} |

新预测器学到了一些统计规律，但仍输给非常简单的基线。当前损失把连续身体量、离散成功和局部阶段编号混在一起；移动后的局部阶段差也不等同于任务进展。预测的时间尺度、表示、损失和学习速度都还需要辨别。不能把一次失败全部归因于参数太少，也不能断定循环网络不适合这个任务。

因此默认保留 prior，online/replay 作为可重现实验条件。下一步任何更复杂的预测模型，都应先在相同输入、相同经历和相同反馈时机下胜过零预测、动作均值和局部状态—动作表。

## 6. 长链问题仍然存在

额外做了 288 次 40 步诊断：8 个新世界、3 个模型、两种身体缺口、3 种依赖条件和两种动作模式。先给各个体同一段实际发生的 384 步学徒经历，再观察它能否根据身体需要取得正确资源。该经历由旧教师离线产生；所以此诊断衡量“有这段经验之后的行为”，不是证明从出生就能自主学习。

| 世界条件 | prior 首个身体产物符合需要 | replay 首个身体产物符合需要 |
|---|---:|---:|
| 无依赖 | {q['summary']['prior']['independent']['first_success']:.2%} | {q['summary']['replay']['independent']['first_success']:.2%} |
| 一层前置需求 | {q['summary']['prior']['dependency']['first_success']:.2%} | {q['summary']['replay']['dependency']['first_success']:.2%} |
| 两层前置需求 | {q['summary']['prior']['long_chain']['first_success']:.2%} | {q['summary']['replay']['long_chain']['first_success']:.2%} |

每个格子为 48 个模型—世界—身体分支，不是 48 个独立世界。长链中，prior 平均约 {q['summary']['prior']['long_chain']['blocked']:.2f}/40 步在设备已准备好、却缺少前置物时继续取物。回放没有改善成功率。这与“会操作一个设备，但遇到前置条件时不会可靠地组织接下来的行动”相符。

关键混杂必须保留：旧训练世界允许身体资源依赖令牌，却没有让令牌设备本身依赖另一令牌。长链测试同时改变了依赖深度和设备角色。现在还不能判定是结构记忆不足，还是从未见过这种角色组合。立即加栈、加注意力层或放大模型，都可能绕开了真正原因。

## 7. 我建议的下一步：先辨别失败来源

**第一优先：保持现有图网络，做角色覆盖受控实验。** 做两套具有相同观察信息、训练步数和反馈预算的课程。一套保留原角色分配；另一套在一层依赖中轮换“什么产物会需要什么前置物”，让令牌设备也曾成为依赖的使用者。测试才组合出两层依赖。这样可以问清楚：只改变训练覆盖，现有结构能不能完成新组合？需要单独记录前置物缺失识别、离开受阻设备、找到供应者、获得前置物、返回并完成消费五个环节。

若角色覆盖就显著改善组合任务，首先修课程和数据生成；若在事实已知、角色已覆盖时仍无法完成，才引入可学习的子目标工作记忆。候选内容是当前需要、未满足前置条件、返回目标、已失败尝试。网络决定何时写入、读取和清除；环境只反馈动作结果。把正确供应者写死进导航器，不能证明它学会了组织行动。

**第二优先：把“预测事实”和“选择动作”各自做扎实。** 先在固定实际轨迹上检验下一步预测，使用状态—动作表、线性模型、小循环网络作对照；连续变化、是否成功、是否受阻分开计分。再加两步、四步预测，区分不确定性与确定的错误。只有预测真正优于简单基线，才让其推动动作；只学习、打乱预测、冻结学习、无回放、相同计算预算回放都要能对照。

随后才研究真正的行动可塑性：让一个小的私有动作适配器或价值学习器，从自己的实际后果中改变行为选择。观察熟悉规律突然变动后能否恢复、原规律回来后是否保留能力，以及一次失败是否使它改变策略。当前只给固定动作先验加短期后果偏置，远不足以承担这个目标。

**第三优先：让隐情境和主动观测进入同一个身体世界。** 04 的内容值得保留，但要解决一个明确问题：同一可见场景为何出现不同结果？身份不应由外部阶段标签直接告诉个体。额外观察应有明确成本；比较主动观测、被动等待、随机探测与相同预算。若廉价的误差统计已经解决问题，就不强加多个专家人格或情绪模块。

## 8. 更长的物欲路线

| 阶段 | 要形成的能力 | 推进前要看到的证据 |
|---|---|---|
| 持续个体底座 | 状态、记忆、身体、随机流与来源可持续续接 | 本次已有有限世界的工程验证；还需长时运行、异常恢复与资源预算 |
| 世界与行动学习 | 根据实际反馈学会后果，改变原始行动 | 固定历史上超过简单预测基线；干预学习确实影响适应表现 |
| 多步需要满足 | 围绕缺口组织前置物、返回目标和失败修正 | 覆盖相同角色后仍能组合新依赖；对延迟、干扰和断供有可解释反应 |
| 持续可塑性 | 旧经验保留、新规律可学、损坏可恢复 | 长期规则反复变动；同时衡量遗忘与学习新规律的速度 |
| 自身能力与社会交互 | 学会自己的限制、他者的行为和互动后果 | 行动代价、资源共享、协调与冲突来自环境；不能只靠文字自述 |
| 小模型底座与语言 | 让同一状态和学习机制进入 0.几 B 模型 | 扩大后仍保留可塑性和因果行为；语言能够报告事实而不代替实际经历 |

这条路线里的“身体”最终需要比两条数值更丰富：动作能力、感知质量、记忆可用性、计算预算与外部资源都可以成为必须维持的条件。但现在应先逐个证明它们有必要，不能为了像生命而一次装上很多命名模块。

持续存在也不等于持续输出文字。更有价值的是同一份个体状态能够承接行动后果，经历会改变未来，而不是每次请求都重新演一个角色。当前工程提供了实验用的状态连续性；它不提供永远运行的服务器，也没有证明人的主观体验或人格地位。

## 9. 什么时候值得做 0.几 B 的新模型底座

当小环境中已经出现稳定的组合行动、持续适应和经过对照的记忆效应时，扩模型才有明确目标。那时先固定个体状态协议：哪些信息必须跨事件保留，哪些权重在生命中可变，哪里需要回放，怎样限制学习损坏已有能力。之后可用较大的离线教师帮助学习语言与动作先验，但正式运行的欲求、历史和身体状态必须能通过真实状态路径影响行为。

目前不建议把现有代码直接包装成一个新语言模型预训练项目。先训练 0.几 B，再用提示词要求它遵循物欲逻辑，很可能得到会描述这种逻辑的模型；是否真的按此逻辑持续运行仍要重新验证。当前这组实验已经给出了一个具体教训：模块接进去了、预测误差下降了、行动也变化了，仍然没有带来需要的能力。

我的判断是：继续执行研究，但把下一轮资源集中到“角色覆盖与组合失败”和“可塑学习是否超过简单基线”两个可证伪问题。它们决定应当改数据、改表示、改学习规则，还是确实需要新的状态结构。现在已具备做这些实验的统一工程。

## 10. 外部研究提供什么帮助

CLRS 将多种经典算法放进共同的算法学习基准，适合借鉴其把数据、任务和测量方式固定下来的思路。它不会替物欲证明存在智能或生命。[1]

Recursive Algorithmic Reasoning 在图网络中加入栈，并改进中间算法轨迹表示，在 DFS 的更大图泛化实验中取得改善。这为“可学习的返回状态”提供候选机制；论文同时改变了表示等因素，不能据它认定我们的长链失败必然缺一个栈。[2]

Maintaining Plasticity in Deep Continual Learning 区分了忘记旧知识与失去学习新知识的能力，并测试了包括 continual backpropagation 在内的方法。它提醒下一轮长期实验同时测这两种失败；论文结果并不是物欲已经实现持续可塑性的证据。[3]

参考文献（原始论文，2026-10-11 核对）：

- [1] Veličković et al., The CLRS Algorithmic Reasoning Benchmark, ICML 2022. https://proceedings.mlr.press/v162/velickovic22a.html
- [2] Jürß et al., Recursive Algorithmic Reasoning, 2023. https://arxiv.org/abs/2307.00337
- [3] Dohare et al., Maintaining Plasticity in Deep Continual Learning, 2024 修订版. https://arxiv.org/abs/2306.13812

## 11. 如何读取交付物

先看本报告，再看 README 和 docs/MECHANISMS.md。运行包包含全部源码、默认历史存档、选定模型以及新实验的原始数据。完整包还保留恢复到的历史实验大文件；不包含可重新生成的 Python 缓存。

新实验的摘要在 results/summary.json，简单预测对照在 results/prediction_diagnostic.json，长链诊断在 results/dependency_diagnostic.json。完整原始轨迹和代表个体状态也在 results；协议与执行前源码指纹分别在 PROTOCOL.md 和 results/frozen_before_formal.json。可以沿着结论回到数据，也可以从同一存档重新分支。

默认运行不代表最终形态。保留稳定参考、暴露失败原因，再用证据决定下一步改动，是这次整合真正要建立的能力。
'''
    (ROOT/'REPORT.md').write_text(text)
    def inline(x):
        x=html.escape(x)
        x=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',x)
        x=re.sub(r'https://[^\s<]+',lambda m:f'<a href="{m[0]}">{m[0]}</a>',x)
        return x
    output=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                table.append([c.strip() for c in lines[i].strip('|').split('|')]);i+=1
            output.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in table[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in table[2:])+'</tbody></table></div>');continue
        if line.startswith('# '):output.append('<h1>'+inline(line[2:])+'</h1>')
        elif line.startswith('## '):
            heading=line[3:];slug='part-'+heading.split('.')[0];output.append(f'<h2 id="{slug}">'+inline(heading)+'</h2>')
        elif line.startswith('- '):output.append('<p class="reference">'+inline(line[2:])+'</p>')
        else:output.append('<p>'+inline(line)+'</p>')
        i+=1
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef2f3;color:#15262c;font:17px/1.9 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:960px;margin:0 auto;padding:42px 36px 90px;background:#fff}h1{font-size:clamp(28px,5vw,43px);line-height:1.4;letter-spacing:-.5px;margin:18px 0 12px}h2{font-size:25px;line-height:1.5;margin:48px 0 20px;padding-top:18px;border-top:2px solid #dce7e7;color:#0d6061}p{margin:18px 0}strong{font-weight:750}a{color:#09686c;overflow-wrap:anywhere}.eyebrow{font-size:12px;letter-spacing:2px;color:#217c7c;font-weight:700}.nav{display:flex;flex-wrap:wrap;gap:8px;margin:25px 0 35px}.nav a{text-decoration:none;padding:6px 14px;background:#edf5f3;border-radius:20px;font-size:14px}.table-scroll{overflow:auto;margin:22px 0;border:1px solid #dce6e6;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px;min-width:570px;text-align:left}td,th{padding:13px 15px;border-bottom:1px solid #e4ecec;vertical-align:top}th{background:#edf5f3;color:#115959}tr:last-child td{border:0}.reference{font-size:14px;color:#52646c}.footer{font-size:13px;color:#52646c;margin-top:55px;padding-top:18px;border-top:1px solid #ddd}@media(max-width:600px){main{padding:24px 19px 65px}body{font-size:16px;line-height:1.9}h2{font-size:22px;margin-top:38px}td,th{padding:10px}table{font-size:13px}}@media print{body{background:#fff}main{max-width:none;padding:0}.nav{display:none}h2{break-after:avoid}.table-scroll{overflow:visible}table{min-width:0;font-size:11px}}'''
    nav='<nav class="nav">'+''.join(f'<a href="#part-{n}">{label}</a>' for n,label in [(1,'整合内容'),(4,'真实结果'),(5,'反证'),(6,'长链问题'),(7,'下一步'),(9,'小模型底座')])+'</nav>'
    doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>物欲：统一工程与下一步</title><style>'+css+'</style></head><body><main><div class="eyebrow">WUYU / INTEGRATION REVIEW</div>'+output[0]+nav+''.join(output[1:])+'<div class="footer">独立 HTML · 无外部脚本或字体 · 可在手机浏览器阅读 · 数据与源码一同交付</div></main></body></html>'
    (ROOT/'REPORT.html').write_text(doc)
    print(json.dumps(dict(markdown_chars=len(text),html_bytes=len(doc.encode()),tables=doc.count('<table>')),ensure_ascii=False))


if __name__=='__main__':main()
