"""Build a self-contained Chinese report from the completed, audited experiment."""
import base64
from html import escape
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def html_inline(s):
    s=escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)

def render_html(md):
    lines=md.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('!['):
            alt,url=re.match(r'!\[(.*?)\]\((.*?)\)',line).groups()
            path=Path(url.removeprefix('sandbox:'));raw=base64.b64encode(path.read_bytes()).decode()
            out.append(f'<figure><img alt="{escape(alt)}" src="data:image/png;base64,{raw}"><figcaption>{escape(alt)}</figcaption></figure>');i+=1;continue
        if line.startswith('```'):
            buf=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
            out.append('<pre><code>'+escape('\n'.join(buf))+'</code></pre>');i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+',c) for c in cells):table.append(cells)
                i+=1
            out.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html_inline(c)+'</th>' for c in table[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html_inline(c)+'</td>' for c in row)+'</tr>' for row in table[1:])+'</tbody></table></div>');continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+html_inline(line[n:].strip())+f'</h{n}>');i+=1;continue
        if line.startswith('- '):
            items=[]
            while i<len(lines) and lines[i].startswith('- '):items.append('<li>'+html_inline(lines[i][2:])+'</li>');i+=1
            out.append('<ul>'+''.join(items)+'</ul>');continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ','```','![')):parts.append(lines[i]);i+=1
        out.append('<p>'+html_inline(' '.join(parts))+'</p>')
    css='''body{margin:0;background:#eef2f4;color:#243442;font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1040px;margin:32px auto;background:white;padding:42px 52px;border-radius:12px}h1{font-size:31px;line-height:1.35;color:#173f46;margin-top:0}h2{font-size:23px;border-top:1px solid #dce4e8;padding-top:24px;margin-top:36px}h3{font-size:19px}p{margin:15px 0}strong{color:#174c54}code{font:13px/1.65 ui-monospace,monospace;background:#edf2f4;padding:2px 4px;border-radius:3px}pre{background:#edf2f4;padding:18px;overflow:auto}pre code{padding:0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 13px;text-align:left;border-bottom:1px solid #dce4e8}th{background:#e9f1f2;color:#173f46}.table{overflow:auto}figure{margin:28px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#65747e}li{margin:8px 0}@media(max-width:700px){main{padding:24px 18px;margin:0;border-radius:0}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}figure,table{break-inside:avoid}}'''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物欲完整体实验报告</title><style>'+css+'</style><main>'+''.join(out)+'</main></html>'

def main():
    run=ROOT/'experiments/whole_holdout';a=json.loads((run/'analysis.json').read_text())
    audit=json.loads((run/'audit.json').read_text());assert audit['passed'] and audit['lives']==1024
    verify=json.loads((ROOT/'validation/FINAL_VERIFIED_TESTS.json').read_text());assert verify['passed']
    av=a['averages'];r=json.loads((run/'mechanism_readout.json').read_text());p=json.loads((run/'protocol.json').read_text())
    rows=[]
    names={'A':'修正语义后的完整基线','B':'动作学习，新增状态每步清空','C':'动作学习＋一般 GRU 状态','D':'动作学习＋保持／写入／清除状态'}
    for mode in 'ABCD':
        x=av['challenge'][mode];y=av['stable'][mode]
        rows.append(f"| {mode}：{names[mode]} | {x['challenge_burden']:.6f} | {x['survivors']}/{x['lives']} | {y['total_burden']:.6f} | {y['survivors']}/{y['lives']} |")
    table='\n'.join(rows)
    md=f'''# 物欲完整体：动作学习与持续状态实验

2026-10-12｜版本 1.1.0｜已完成实现、冻结评价和全量物理审计

**工程接通完成，预期的持续行动能力尚未得到证实。** 新动作学习在熟悉环境中改善了身体维持；在依赖、通路变化、需要中断与收益消失的组合挑战中，持续状态候选没有显示可靠的整体优势，也没有超过一般循环状态。

因此保留修正后的 A 为默认完整体。B/C/D 已经位于同一主干中，可以明确开启、续跑和干预；它们作为研究候选保留，当前不提升为已验证的新能力。

## 已经做成了什么

- 修正跨设备移动被误记为阶段进展的问题，神经预测、真实回放与证据模型全部采用同设备语义。
- 给已有整合体接入真实轨迹上的动作学习、跨步归因和有限工作状态，保留旧身体、技能记忆、图网络及 E/U。
- 在同一实现中建立 A/B/C/D 四个条件。B/C/D 各分配 22,120 个新增参数，C/D 使用相同优化设置；没有教师动作、隐藏路线或子目标奖励。
- 将新增状态、权重、Adam、未完成的跨步缓存和随机流写进完整存档，提供明确的旧版本迁移。
- 完成 1,024 条正式生命、2,621,440 次主生命转移，以及 64 个检查点的 256 条诊断分支。另保留两轮共 128 条开发生命与 64 条语义回归生命。

不是把一个独立实验做完再拼回主体：新机制从第一次试跑就通过完整体的同一个动作出口与真实身体反馈共同运行。

## 正式结果

64 个从未用于开发的世界，各交叉两个固定既有模型初始化；每种条件、每类环境 128 条生命，每条 2,560 步。首要指标是 t=1024..2559 固定挑战窗口的平均身体匮乏，死亡计为损失 2，越低越好。

| 条件 | 挑战身体损失 | 挑战最终存活 | 稳定环境全生命损失 | 稳定环境存活 |
|---|---:|---:|---:|---:|
{table}

![完整体四条件的身体损失、配对差异与存活结果](sandbox:{ROOT}/figures/whole_results.png)

预先指定的首要比较 D−A 为 **−0.008680，95% 世界配对区间 [−0.111065,+0.088098]**。区间跨过零，点估计也未达到预设的 0.01 最小有用下降。不能据此宣布收益成立，也不能用不显著来证明完全没有作用。

D 比 A 多存活 8/128，差为 6.25 个百分点；世界配对区间为 [−1.56,+14.06] 个百分点。因此“35 比 27 多”只是观察到的方向，尚不能给出稳定改善的判断。

C 的挑战损失平均最低。D−C 为 +0.042266，描述性 95% 区间 [0.003431,0.084748]，方向上不支持 D 优于普通 GRU。该次级比较未经多重校正，不单独宣传为确认性发现。B−A、C−A 的挑战损失区间也跨过零。

稳定环境中，B/C/D 全部 128/128 存活，身体损失分别较 A 下降约 31.8%、33.3%、33.5%。D−A 的绝对差为 −0.004376，区间 [−0.005107,−0.003620]。新增循环状态相对 B 的额外改善区间跨零，所以这里更支持动作可塑性带来的局部调节收益，不能把收益归功于特定“持续目标”结构。

四组均通过本轮稳定能力保护阈值。这个结论限定于已测稳定环境和两个固定既有初始化，不等于所有旧任务上的普遍等效。

## 机制是否真的在起作用

动作学习确实能影响活着的个体。挑战环境中，B/C/D 相对同状态组合基线、使用同一次随机抽样，平均改变 {r['aggregate']['challenge']['B']['live_action_changes']:.2f}、{r['aggregate']['challenge']['C']['live_action_changes']:.2f}、{r['aggregate']['challenge']['D']['live_action_changes']:.2f} 次动作；稳定环境中约为 115–126 次。这证明新增通道被使用，不证明这些变化有益。

在八个预先指定世界、两个初始化、两个时点上，C 和 D 各有 32 个状态检查点，其中各 27 个仍存活。屏蔽新状态读取后，C 在 4 个、D 在 2 个检查点出现共同存活期间的动作变化；D 的受影响动作总计 27 次。真实匹配历史状态替换没有造成动作变化。清除后立即精确恢复的 64 个分支全部回到原轨迹，父生命均未被分支改变。

屏蔽没有造成一致的身体损失上升；D 分支的平均损失反而下降约 0.00219。当前诊断不支持“这个状态对持续追求是必要的”。历史供体也未被证明与当前追求语义无关，扰动范数不完全相同，因此本轮没有完成选择性目标内容的因果证明。

D 的平均直接保留系数约 {r['aggregate']['challenge']['D']['mean_direct_carry']:.3f}，C 约 {r['aggregate']['challenge']['C']['mean_direct_carry']:.3f}。这说明实现了不同的状态动力学；由于候选内容还有循环路径，该数值不能直接当作总记忆寿命。给隐变量取名“工作状态”不等于它已经承载目标。

## 主要失败在哪里

![全体挑战生命的连续身体损失轨迹；死亡始终纳入](sandbox:{ROOT}/figures/whole_timeline.png)

受阻后的重复行为依然明显。只计算存活期间，D 每条挑战生命平均仍有 {r['aggregate']['challenge']['D']['live_route_denials']:.2f} 次受阻移动请求和 {r['aggregate']['challenge']['D']['live_blocked_takes']:.2f} 次缺少前置物的取物请求。持续学习没有把这些失败稳定地转化为有效改路。

代码检查指出一个具体的表示缺口：旧图输入仍使用固定环形距离；两类后果预测的目标包含身体、库存、设备阶段和产物，却没有明确预测移动是否到达、通路是否仍可达。实际位置变化留在原始经历中，但没有形成可更新的空间关系供动作层稳定使用。这个结构事实使“旧空间先验与新障碍不一致”成为下一步应优先检验的解释；本轮没有通过单独消融证明它是全部失败的唯一原因。

前置物角色覆盖也未完全消除问题。正式挑战中，确实完成两类依赖令牌经历的个体为 A 110/128、B 112/128、C 110/128、D 109/128。课程保证机会，不能保证每个自主个体都取得该经历；报告没有把未经历或早死者从分母中剔除。

## 语义修正、迁移和验证

独立语义回归使用八个新世界、两个初始化、稳定与原生发展环境，原版和修正版合计 64 条生命。稳定对照双方 16/16 存活，平均损失 0.018582 → 0.017100；原生发展双方 16/16，0.018302 → 0.017980。原标签分别记录了 685 和 1,184 个跨设备的伪进展时刻。这个回归支持修复语义问题及本轮旧能力保持，不是新的广泛优越性研究。

真实旧检查点迁移示例保留了原身体、库存、旧图状态、技能和动作 RNG，重标 32 个移动证据样本。原神经预测器已累计 767 次更新，因目标语义改变而明确重初始化，再用 64 条真实保留回放重建，从四步前缀近似恢复神经活跃状态。不能把它称为完全无损迁移。

历史 131 项测试与当前主干 31 项测试通过，共 162 项；其中包括梯度有限差分、零修正恢复基线、真实标签一致性、死亡吸收态、跨步缓存存档续跑、诊断父状态不变及迁移检查。正式 1,024 条生命逐步重放全部 2,621,440 个物理转移，观察、目标、身体账本、死亡和原始轨迹校验值全部通过。

审计捕捉到两份不完整轨迹文件，已按完全相同的冻结配置和种子重建；新文件与原运行记录的 SHA256、所有科学汇总字段及最终状态均完全吻合。保留损坏字节与修复记录，没有重新抽种子、替换成绩或增加统计样本。

原始过程标记可能包含死亡后的动作请求。报告统一使用明确的 `live_*` 审计重算计数，冻结的原始 summary 保留。身体损失和生存的主分析无需更改。

## 解释边界和版本决定

- 首要统计单位为 64 个世界，两个既有初始化在世界内先平均；时间步和诊断分支没有充当独立样本。模型初始化的结论仅覆盖本次选定的 0、1。
- 每个正式生命的动作适配器重新零初始化，最多 80 次跨步更新。这是短生命预算内在线适应的检验，不是充分预训练后的能力上限。
- 当前样本量不能精确排除很小的效应。正式方案、样本量、最小有用效应和停止规则已冻结，没有依据正式结果继续调参或延长到显著。
- 本轮固定了事件顺序和四个窗口，随机化世界布局、配方及关系组合；尚未检验任意事件顺序、更长依赖或开放世界。
- 对“合理恢复／合理放弃”还没有独立的反事实最优判准。身体结果、返回完成和低收益操作是观测指标，不能自行认证目标形成。
- 身体维持目标是给定的评价结构，本轮没有证明新欲望、新价值或人格形成。
- 本轮在线过程不提供教师动作；继承的 07/08 策略保留既有教师训练史，不能将整个系统描述成从零自发形成。

**决定：保留语义修正和完整存档；保留动作学习的稳定环境收益作为证据；A 继续为默认，B/C/D 保持可运行的内部候选。D 的特殊状态组织不获得优先地位。**

下一轮优先在这个完整体内把实际移动结果写回环境关系表示，检验动作层能否据此改路，并保留 C 作为强对照。先验证“失败经历 → 可达关系更新 → 多步行动改变 → 身体收益”这条实际闭环，再扩大持续追求的功能解释。该下一轮尚未执行。

## 复现入口

代码与全部原始源码见 Source 包；本轮全部原始实验记录见 Data 分包。默认入口和继续同一个体的命令见 `README.md`，机制细节见 `docs/WHOLE_MECHANISMS.md`，新旧状态转换见 `whole_migration.py`，正式协议与主分析在 `experiments/whole_holdout/`。

冻结协议摘要：`{p['protocol_digest']}`。
'''
    (ROOT/'WUYU_Whole_Report_2026-10-12.md').write_text(md,encoding='utf-8')
    (ROOT/'WUYU_Whole_Report_2026-10-12.html').write_text(render_html(md),encoding='utf-8')
    handoff=f'''# 物欲完整体 1.1 接续说明

已完成用户“做”的授权：在统一主干实施语义修正、真实轨迹动作学习、B/C/D 新状态控制，完成正式整体现实轨迹实验与归档。默认 `Session.create()` 和 CLI `run` 为 whole/A/native，D 未提升为默认。

## 实际结果

正式 64 世界 × 2 固定初始化 × A/B/C/D × challenge/stable，1,024 条生命、2,621,440 转移。首要 D−A = −0.0086795434，世界聚类 95% CI [−0.111064935,+0.088097809]，未建立优势。挑战 A/B/C/D 生存 27/30/33/35（各 128），损失 0.855784/0.861955/0.804839/0.847105。稳定全部 128/128，动作学习平均降低身体损失约 32–34%。D 未超过普通 GRU；次级 D−C +0.042266，描述性 CI [0.003431,0.084748]，未经多重校正。

## 机制证据和限制

同主干真实反馈、无教师/隐藏路线/子目标奖励。B/C/D 同分配 22,120 参数；A 保留原图网络及旧学习。状态屏蔽 D 仅 2/32 检查点改变存活期动作，历史替换无变化，64 个恢复分支精确。未证明目标内容或状态必要性。课程机会不等于实际覆盖；D 两类依赖令牌实际经历为 109/128。

## 完成核验

131 历史测试＋31 当前主干测试通过。正式全量物理重放通过。两份不完整轨迹以原种子和冻结配置重建，并与原 SHA256 及最终状态完全吻合，详见 validation/trace_reconstruction/recovery.json。原始过程请求包含死后时钟标记，实际行为用 mechanism_readout.json 的 live_* 字段，主损失/生存不受影响。

## 文件和重跑

README.md 有实际入口、迁移、运行与重现实验命令。全部新代码在 wuyu_unified/；旧 source_projects/ 源码不修改。新主干文件 whole.py、action_learning.py、whole_world.py、whole_migration.py。A/B/C/D 都是 WholeAgent 的配置，不是四个独立简化代理。

正式来源锁和协议：experiments/whole_holdout/protocol.json；摘要 `{p['protocol_digest']}`。结果 analysis.json；身体过程原始 traces/；干预 diagnostics/；完整存档 checkpoints/；全量审计 audit.json。开发两个版本及源码快照保留，不能拿开发变化当正式确认结果。

历史完整大数据仍在上一轮七个 Data 包中，当前 Source 包包含全部源码、运行所需模型和旧整合验收材料。本轮 Data 分包恢复全部新增实验原始记录。报告为 WUYU_Whole_Report_2026-10-12.md/html。

## 下一步建议，尚未执行

先补“实际移动经验 → 可更新的可达关系 → 动作利用”的闭环。当前编码仍使用固定环形距离，两类预测器没有移动到达目标，原始位置经历没有形成可更新通路表示。这是明确结构缺口，但尚未单独证明为全部失败原因。继续以完整体为研究单位，保留 A/B/C 强对照和负结果，先检验该闭环再谈更强持续追求功能。不要自动再加目标模块，不宣称新欲望、价值或人格已形成。
'''
    (ROOT/'HANDOFF.md').write_text(handoff,encoding='utf-8')
    (ROOT/'WUYU_Whole_HANDOFF_2026-10-12.md').write_text(handoff,encoding='utf-8')
    print('Report and handoff created from complete audited data.')

if __name__=='__main__':main()
