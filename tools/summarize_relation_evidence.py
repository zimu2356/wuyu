"""Post-run descriptive evidence, separate from the frozen primary analysis.

Requires complete, audited cohorts. No exclusion or model selection is applied.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES=('relation_holdout_main','relation_holdout_scenarios','relation_holdout_long')

def read_cohort(name):
    directory=ROOT/'experiments'/name
    p=json.loads((directory/'protocol.json').read_text())
    a=json.loads((directory/'analysis.json').read_text())
    audit=json.loads((directory/'audit.json').read_text())
    done=json.loads((directory/'RUN_COMPLETE.json').read_text())
    assert audit['passed'] and audit['lives']==done['lives']==a['complete_lives']
    rows=[json.loads(f.read_text()) for f in sorted((directory/'summaries').glob('*.json'))]
    assert len(rows)==done['lives']
    return directory,p,a,audit,rows

def paired(rows,seeds,case,key,left='R10',right='R00'):
    def vals(mode):
        return np.array([np.mean([r[key] for r in rows if r['seed']==s and r['mode']==mode and r['case']==case]) for s in seeds])
    delta=vals(left)-vals(right)
    draws=np.random.default_rng(987622).integers(0,len(seeds),(10000,len(seeds)))
    return dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
        n_worlds=len(seeds),world_sd=float(delta.std(ddof=1)),paired_differences=delta.tolist())

def contrast(a,case,name,key):
    return next(x for x in a['contrasts'] if (x['case'],x['contrast'],x['metric'])==(case,name,key))

def main():
    result=dict(scope='Supplemental descriptive analysis after protocol freeze; primary test remains exactly predeclared.',cohorts={})
    timelines={}
    diagnostic_integrity=[]
    for name in NAMES:
        directory,p,a,audit,rows=read_cohort(name)
        groups={}
        for case in p['cases']:
            groups[case]={}
            for mode in p['modes']:
                chosen=[r for r in rows if r['case']==case and r['mode']==mode]
                lat=[r['first_reopened_crossing_latency'] for r in chosen if r['first_reopened_crossing_latency'] is not None]
                moving=sum(r['movement_attempts'] for r in chosen)
                observed=[r for r in chosen if r['contradiction_observed']]
                groups[case][mode]=dict(lives=len(chosen),
                    role_coverage=sum(r['both_dependent_token_roles_experienced'] for r in chosen),
                    death_before_route_change=sum(r['death_before_route_change'] for r in chosen),
                    death_strictly_before_route_event=sum(r['death_tick'] is not None and r['death_tick']+1<r['change_tick'] for r in chosen),
                    death_at_route_event=sum(r['death_tick'] is not None and r['death_tick']+1==r['change_tick'] for r in chosen),
                    death_after_route_event=sum(r['death_tick'] is not None and r['death_tick']+1>r['change_tick'] for r in chosen),
                    contradiction_lives=len(observed),movement_attempts_total=moving,
                    structural_denials_total=sum(r['live_structural_denials'] for r in chosen),
                    structural_denials_per_100_moves=100*sum(r['live_structural_denials'] for r in chosen)/max(moving,1),
                    post_contradiction_denials_mean_all=float(np.mean([r['post_contradiction_denials'] for r in chosen])),
                    post_contradiction_denials_mean_observed=float(np.mean([r['post_contradiction_denials'] for r in observed])) if observed else None,
                    live_at_reopening=sum(r['recovery_live_steps']>0 for r in chosen) if case=='r_reopen' else None,
                    any_reopened_crossing=sum(r['reopened_crossings']>0 for r in chosen) if case=='r_reopen' else None,
                    median_first_reopened_crossing_latency_conditional=float(np.median(lat)) if lat else None,
                    crossed_latency_n=len(lat) if case=='r_reopen' else None,alive_at_2560=sum(r['alive_at_2560'] for r in chosen),
                    prefix_challenge_burden=float(np.mean([r['prefix_challenge_burden'] for r in chosen])),
                    extended_burden=float(np.mean([r['extended_burden'] for r in chosen])) if p['horizon']>2560 else None)
                burden=np.zeros(p['horizon']);alive=np.zeros(p['horizon'])
                for r in chosen:
                    label=f"{r['seed']}_i{r['initialization']}_{case}_{mode}"
                    with np.load(directory/'traces'/(label+'.npz')) as z:
                        t=z['trace'];burden+=t['burden'];alive+=t['alive']
                timelines[name+'__'+case+'__'+mode+'__burden']=burden/len(chosen)
                timelines[name+'__'+case+'__'+mode+'__alive']=alive/len(chosen)
        # Verify diagnostic restoration directly from saved arrays, separately
        # from the runner assertions and the all-life physical audit.
        norm_differences=[];other_missing=0;diag_steps=0;other_is_reverse=0
        for f in sorted((directory/'diagnostics').glob('*_meta.json')):
            d=json.loads(f.read_text());label=f.name.removesuffix('_meta.json')
            with np.load(directory/'diagnostics'/(label+'_intact.npz')) as z: intact=z['trace']
            with np.load(directory/'diagnostics'/(label+'_restored.npz')) as z: restored=z['trace']
            assert np.array_equal(intact,restored)
            assert d['exact_restore'] and d['parent_unchanged']
            diag_steps+=4*len(intact)
            other_missing+=d['matched_other_pair'] is None
            if d['matched_other_pair'] is not None:
                pos,act=d['pair'];site,action=d['matched_other_pair']
                other_is_reverse+=site==(pos+(-1 if act==0 else 1))%6 and action==1-act
            norm_differences.append(abs(d['branches']['related_old']['perturbation_norm']-d['branches']['matched_other']['perturbation_norm']))
        diagnostic_integrity.append(dict(cohort=name,checkpoints=len(norm_differences),branches=4*len(norm_differences),
            branch_steps=diag_steps,all_saved_restore_arrays_exact=True,other_edge_missing=other_missing,
            other_is_reverse_direction_of_same_physical_edge=other_is_reverse,
            max_perturbation_norm_difference=max(norm_differences,default=0.),
            limitation='Other edge is not proven behaviorally irrelevant; live-at-reopening and latency are conditional descriptions, not the primary estimand.'))
        result['cohorts'][name]=dict(protocol_digest=p['protocol_digest'],lives=len(rows),
            physical_steps=audit['physical_steps'],groups=groups)
        if name=='relation_holdout_main':
            primary=contrast(a,'r_combined','R10-R00','challenge_burden')
            guard_loss=contrast(a,'r_stable','R10-R00','total_burden')
            guard_surv=contrast(a,'r_stable','R10-R00','final_alive')
            result['primary']=dict(**primary,minimum_useful_improvement=p['minimum_useful_improvement'],
                point_estimate_pass=primary['delta']<=-p['minimum_useful_improvement'],ci_pass=primary['ci95'][1]<0,
                stable_burden_guardrail=guard_loss,stable_survival_guardrail=guard_surv,
                guardrails_pass=guard_loss['ci95'][1]<=.02 and guard_surv['ci95'][0]>=-.02)
            result['primary']['all_pass']=result['primary']['point_estimate_pass'] and result['primary']['ci_pass'] and result['primary']['guardrails_pass']
        if name=='relation_holdout_long':
            result['long_window_contrasts']={key:paired(rows,p['seeds'],'r_combined',key) for key in
                ('prefix_challenge_burden','extended_burden','alive_at_2560','final_alive')}
    result['diagnostic_integrity']=diagnostic_integrity
    result['formal_lives']=sum(v['lives'] for v in result['cohorts'].values())
    result['formal_physical_steps']=sum(v['physical_steps'] for v in result['cohorts'].values())
    result['diagnostic_checkpoints']=sum(v['checkpoints'] for v in diagnostic_integrity)
    result['diagnostic_branch_steps']=sum(v['branch_steps'] for v in diagnostic_integrity)
    result['diagnostic_other_missing']=sum(v['other_edge_missing'] for v in diagnostic_integrity)
    result['diagnostic_other_same_edge_reverse']=sum(v['other_is_reverse_direction_of_same_physical_edge'] for v in diagnostic_integrity)
    out=ROOT/'experiments/relation_validation'
    (out/'evidence_readout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'mean_timelines.npz',**timelines)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cohorts',)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
