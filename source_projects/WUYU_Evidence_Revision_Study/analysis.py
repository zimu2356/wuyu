"""Frozen endpoints. Seed-level paired bootstrap; other contrasts are descriptive."""
import argparse, json, math
from pathlib import Path
import numpy as np
from storage import write_json, read_gz_json, read_events

def interval(values,c,offset=0):
    a=np.array(values,dtype=float);rng=np.random.default_rng(c['bootstrap_seed']+offset)
    means=a[rng.integers(0,len(a),size=(c['bootstrap_resamples'],len(a)))].mean(axis=1)
    return dict(n=len(a),mean=float(a.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),
                positive=int(np.sum(a>0)),negative=int(np.sum(a<0)),zero=int(np.sum(a==0)),
                median=float(np.median(a)),min=float(a.min()),max=float(a.max()))

def analyze(folder,c):
    folder=Path(folder);rows=json.loads((folder/'summaries.json').read_text())
    lookup={(r['seed'],r['case'],r['method']):r for r in rows};seeds=sorted({r['seed'] for r in rows})
    assert len(rows)==len(seeds)*len(c['cases'])*len(c['methods'])
    fatigue={}
    for r in rows:
        e=read_events(folder/f'{r["seed"]}_{r["case"]}_{r["method"]}.npy.gz');post=e[r['tau']:r['tau']+c['post_window']]
        fatigue[r['seed'],r['case'],r['method']]=dict(post_fatigue=float(np.mean(post['after'][:,2])),
             post_composite_loss=float(np.mean(np.sum(post['after'][:,:2]**2,axis=1)+c['fatigue_penalty']*post['after'][:,2]**2+c['action_cost']*(post['action']>0))))
    contrasts={};table=[]
    for case in c['cases']:
        for method in c['methods']:
            rr=[lookup[s,case,method] for s in seeds]
            table.append(dict(case=case,method=method,n=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in
                ('post_loss','life_loss','pre_loss','late_loss','active_fraction','post_active_fraction','retest_fraction','activity_cost',
                 'pre_revisions','post_revisions','unaffected_post_revisions')},
                 recovered=sum(r['recovery']['recovered'] for r in rr),recovery_restricted_mean=float(np.mean([r['recovery']['delay'] for r in rr])),
                 **{k:float(np.mean([fatigue[s,case,method][k] for s in seeds])) for k in ('post_fatigue','post_composite_loss')}))
            contrasts[f'{case}:BE-{method}']=interval([lookup[s,case,'BE']['post_loss']-lookup[s,case,method]['post_loss'] for s in seeds],c)
    primary=contrasts['local_swap:BE-BEU']
    stable=interval([lookup[s,'stable','BEU']['life_loss']-lookup[s,'stable','BE']['life_loss'] for s in seeds],c,1)
    comparisons={}
    for method in c['methods']:
        comparisons[method]=interval([lookup[s,'local_swap','B0']['post_loss']-lookup[s,'local_swap',method]['post_loss'] for s in seeds],c,2)
    causal_rows=[lookup[s,'local_swap','BEU'] for s in seeds];present=[r for r in causal_rows if r['causal_event'] is not None]
    causal=dict(eligible=len(present),total=len(seeds),missing_seeds=[r['seed'] for r in causal_rows if r['causal_event'] is None],
                first_action_differs=sum(r.get('causal_first_action_differs',False) for r in present),
                conditional_effect=interval([r['causal_block_minus_normal'] for r in present],c,3) if present else None,
                intent_to_diagnose_effect=interval([r.get('causal_block_minus_normal',0.) for r in causal_rows],c,3),
                caveat='Conditional on first actual changed-tool revision; zero for no eligible event in all-seed diagnostic. Not a randomly selected life event.')
    latency={};opportunity={};recurrence={};probe_rows=[]
    for method in c['methods']:
        rr=[lookup[s,'local_swap',method] for s in seeds]
        latency[method]=dict(first_use_censored_512=sum(v is None or v>=512 for r in rr for v in r['first_retest'].values()),
            first_use_capped_mean=float(np.mean([min(512,v) if v is not None else 512 for r in rr for v in r['first_retest'].values()])),
            three_samples_censored_512=sum(v is None or v>=512 for r in rr for v in r['three_new_samples'].values()),
            three_samples_capped_mean=float(np.mean([min(512,v) if v is not None else 512 for r in rr for v in r['three_new_samples'].values()])),
            revision_censored_512=sum(r['first_revision'] is None or r['first_revision']>=512 for r in rr),
            revision_capped_mean=float(np.mean([min(512,r['first_revision']) if r['first_revision'] is not None else 512 for r in rr])),
            prediction_accuracy_censored_512=sum(r['first_prediction_accuracy'] is None or r['first_prediction_accuracy']>=512 for r in rr),
            prediction_accuracy_capped_mean=float(np.mean([min(512,r['first_prediction_accuracy']) if r['first_prediction_accuracy'] is not None else 512 for r in rr])))
        op=[lookup[s,'opportunity',method] for s in seeds]
        opportunity[method]=dict(revisited=sum(r['first_spare_use'] is not None for r in op),
             first_use_capped_mean=float(np.mean([r['first_spare_use'] if r['first_spare_use'] is not None else 512 for r in op])),
             mean_uses=float(np.mean([r['post_spare_uses'] for r in op])))
        recurrence[method]=float(np.mean([lookup[s,'recurrence',method]['return_loss'] for s in seeds]))
    for case in c['cases']:
        for method in c['methods']:
            for s in seeds:
                meta=read_gz_json(folder/f'{s}_{case}_{method}.json.gz');tau=meta['summary']['tau']
                for cp in meta['checkpoints']:
                    for r in cp['rows']: probe_rows.append(dict(seed=s,case=case,method=method,relative_tick=cp['tick']-tau,**r))
    probe_summary=[]
    # Actual and matched states, selected and unselected actions kept in separate cells.
    for case in c['cases']:
        for method in c['methods']:
            for time in (0,128,512):
                for state in ('actual','matched_axis0','matched_axis1'):
                    for selected in (True,False):
                        rr=[r for r in probe_rows if r['case']==case and r['method']==method and r['relative_tick']==time and r['state']==state and r['selected']==selected]
                        probe_summary.append(dict(case=case,method=method,time=time,state=state,selected=selected,rows=len(rr),
                            **{k:float(np.mean([r[k] for r in rr])) for k in ('mse','mean_mse','age','mass','ess','coverage','old_mass_fraction')}))
    result=dict(primary=primary,stable_guard=stable,
                primary_useful_pass=primary['low']>c['minimum_useful_effect'],
                stable_guard_pass=stable['high']<c['stable_noninferiority_margin'],
                overall_pass=primary['low']>c['minimum_useful_effect'] and stable['high']<c['stable_noninferiority_margin'],
                table=table,secondary_contrasts=contrasts,local_B0_comparisons=comparisons,causal=causal,
                latency=latency,opportunity=opportunity,recurrence_return_loss=recurrence,probe_summary=probe_summary)
    write_json(folder/'ANALYSIS.json',result);return result

def choose_count(rows,c):
    lookup={(r['seed'],r['case'],r['method']):r for r in rows};seeds=sorted({r['seed'] for r in rows})
    d=[lookup[s,'local_swap','BE']['post_loss']-lookup[s,'local_swap','BEU']['post_loss'] for s in seeds]
    stable=[lookup[s,'stable','BEU']['life_loss']-lookup[s,'stable','BE']['life_loss'] for s in seeds]
    sd=float(np.std(d,ddof=1));ss=float(np.std(stable,ddof=1));widths={}
    for n in c['holdout_count_candidates']:
        widths[str(n)]=dict(primary=1.96*sd/math.sqrt(n),stable=1.96*ss/math.sqrt(n))
    qualifying=[n for n in c['holdout_count_candidates'] if widths[str(n)]['primary']<=c['precision_target'] and widths[str(n)]['stable']<=c['stable_precision_target']]
    n=min(qualifying) if qualifying else max(c['holdout_count_candidates'])
    return dict(count=n,development_primary_sd=sd,development_stable_sd=ss,expected_halfwidths=widths,
                expected_precision_met=bool(qualifying),selection_uses='Paired standard deviations only; no direction or winner selection.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);args=ap.parse_args()
    c=json.loads(Path(args.config).read_text());r=analyze(args.data,c)
    print(json.dumps({k:r[k] for k in ('primary','stable_guard','overall_pass')},indent=2))
