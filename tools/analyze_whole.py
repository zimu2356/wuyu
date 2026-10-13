"""World-cluster paired descriptive intervals; no step-level pseudo-replication."""
import json
from pathlib import Path
import argparse
import numpy as np

def analyze(path):
    path=Path(path);p=json.loads((path/'protocol.json').read_text())
    rows=[json.loads(q.read_text()) for q in sorted((path/'summaries').glob('*.json'))]
    expected=len(p['seeds'])*len(p['initializations'])*len(p['modes'])*len(p['cases'])
    if len(rows)!=expected:raise ValueError(f'incomplete {len(rows)}/{expected}')
    metrics=['challenge_burden','total_burden','final_alive','alive_fraction','blocked_takes',
             'route_denials','lost_yield_takes','changes_vs_base','mean_kl','low_benefit_body_productions',
             'completed_return_windows','both_dependent_token_roles_experienced','actor_updates',
             'mean_prediction_mse','elapsed_seconds']
    averages={};contrasts=[];rng=np.random.default_rng(884510)
    draws=rng.integers(0,len(p['seeds']),(10000,len(p['seeds'])))
    def values(case,mode,key):
        return np.array([np.mean([r[key] for r in rows if r['seed']==s and r['mode']==mode and r['case']==case]) for s in p['seeds']])
    for case in p['cases']:
        averages[case]={}
        for mode in p['modes']:
            selected=[r for r in rows if r['mode']==mode and r['case']==case]
            averages[case][mode]={k:float(np.mean([r[k] for r in selected])) for k in metrics}
            averages[case][mode].update(lives=len(selected),survivors=sum(r['final_alive'] for r in selected),
              per_initialization={str(i):dict(challenge_burden=float(np.mean([r['challenge_burden'] for r in selected if r['initialization']==i])),
                                    survivors=sum(r['final_alive'] for r in selected if r['initialization']==i)) for i in p['initializations']})
        for newer,older in [('D','A'),('B','A'),('C','A'),('D','B'),('D','C')]:
            for metric in ['challenge_burden','total_burden','final_alive','blocked_takes','completed_return_windows']:
                delta=values(case,newer,metric)-values(case,older,metric)
                ci=np.quantile(delta[draws].mean(1),[.025,.975])
                contrasts.append(dict(case=case,contrast=newer+'-'+older,metric=metric,
                     delta=float(delta.mean()),ci95=ci.tolist(),world_sd=float(delta.std(ddof=1)),
                     n_worlds=len(delta),paired_differences=delta.tolist()))
    diagnostic=[]
    for case in p['cases']:
        for mode in ('C','D'):
            ds=[d for r in rows if r['case']==case and r['mode']==mode for d in r['diagnostics']]
            if not ds:continue
            summary=dict(case=case,mode=mode,checkpoints=len(ds),alive_checkpoints=sum(d['alive'] for d in ds),
                exact_restores=sum(d['exact_restore'] for d in ds),matched_donors=sum(d['donor_available'] for d in ds))
            for kind in ('suppressed','matched_history'):
                summary[kind]=dict(mean_burden_delta=float(np.mean([d['branches'][kind]['mean_burden']-d['branches']['intact']['mean_burden'] for d in ds])),
                    mean_action_changes=float(np.mean([d['branches'][kind]['action_differences_vs_intact'] for d in ds])),
                    changed_checkpoints=sum(d['branches'][kind]['action_differences_vs_intact']>0 for d in ds))
            diagnostic.append(summary)
    result=dict(protocol_digest=p['protocol_digest'],complete_lives=len(rows),averages=averages,
                contrasts=contrasts,diagnostics=diagnostic,
                inference='Primary D-A challenge contrast predeclared. Other 95% intervals descriptive and unadjusted. World seed is the sampling unit, crossed fixed graph initializations are averaged within world.')
    (path/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory');args=a.parse_args();r=analyze(args.directory)
    print(json.dumps({'averages':r['averages'],'primary':[x for x in r['contrasts'] if x['contrast']=='D-A' and x['metric']=='challenge_burden'],'diagnostics':r['diagnostics']},indent=2))
