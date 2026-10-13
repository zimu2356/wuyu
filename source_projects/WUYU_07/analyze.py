import json
from pathlib import Path
import numpy as np
from wuyu07.state import read_tree
ROOT=Path(__file__).resolve().parent


def world_values(values):
    x=np.asarray(values,float).mean(axis=0)
    return x.mean(axis=tuple(range(1,x.ndim))) if x.ndim>1 else x


def stats(values):
    values=np.asarray(values,float);x=world_values(values);rng=np.random.default_rng(70007)
    boot=x[rng.integers(0,len(x),(4000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci=np.quantile(boot,[.025,.975]).tolist(),
        by_initialization=values.mean(axis=tuple(range(1,values.ndim))).tolist())


if __name__=='__main__':
    out=ROOT/'results';runs=json.loads((out/'main_runs.json').read_text());main={};modes=list(dict.fromkeys(r['mode'] for r in runs))
    for mode in modes:
        subset=[r for r in runs if r['mode']==mode]
        main[mode]={k:stats([r['metrics'][k] for r in subset]) for k in subset[0]['metrics']}
    arrays=read_tree(out/'probes_raw.npz')['arrays'];probes={k:stats(v) for k,v in arrays.items()}
    pairs={
        'experience_success':('independent_mature_success','independent_birth_success'),
        'experience_cost_saved':('independent_birth_cost','independent_mature_cost'),
        'composition_success':('composite_mature_success','composite_birth_success'),
        'composition_cost_saved':('composite_birth_cost','composite_mature_cost'),
        'revaluation':('revalue_adaptive','revalue_fixed'),
        'change_learning':('change_after_success','change_before_success'),
        'retention':('retention_after','retention_before'),
        'skill_success':('independent_mature_success','independent_no_skill_success'),
        'skill_cost_saved':('independent_no_skill_cost','independent_mature_cost'),
        'semantic_error_reduction':('semantic_birth_brier','semantic_mature_brier'),
        'student_experience':('student_mature_success','student_birth_success'),
        'student_memory_dependence':('student_mature_success','student_memory_reset_success')}
    diff={k:stats(arrays[a]-arrays[b]) for k,(a,b) in pairs.items()}
    values={mode:np.array([r['metrics']['survival'] for r in runs if r['mode']==mode]) for mode in modes}
    diff['student_online_learning_survival']=stats(values['neural_student']-values['neural_student_frozen'])
    gates={
        'G1':dict(name='经历形成操作能力',passed=diff['experience_success']['ci'][0]>0 and probes['independent_mature_success']['mean']>=.8),
        'G2':dict(name='既有经历用于新依赖组合',passed=diff['composition_success']['ci'][0]>0 and probes['composite_mature_success']['mean']>=.7),
        'G3':dict(name='按当前身体需要重新估值',passed=diff['revaluation']['ci'][0]>0 and probes['revalue_adaptive']['mean']>=.8),
        'G4':dict(name='局部重新学习并保留未变操作',passed=diff['change_learning']['ci'][0]>0 and diff['retention']['mean']>=-.1),
        'G5':dict(name='独立技能执行器提供额外收益',passed=diff['skill_success']['mean']>=0 and diff['skill_cost_saved']['ci'][0]>0)}
    result=dict(main=main,probes=probes,paired_differences=diff,gates=gates,
        caution='条件于三个固定初始化的世界配对区间；多项比较未校正。混合体受提示旁路与学生自主测试分别解释。')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
