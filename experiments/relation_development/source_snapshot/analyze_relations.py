"""Predeclared world-level comparisons; all exploratory quantities labeled."""
import argparse,json
from pathlib import Path
import numpy as np

METRICS=('challenge_burden','total_burden','late_burden','final_alive','alive_fraction',
 'live_steps','movement_attempts','live_route_denials','live_structural_denials','live_noise_failures',
 'live_blocked_takes','live_action_changes','contradiction_observed','mean_movement_brier','mean_tau',
 'mean_entropy','actor_updates','mean_kl','reopened_crossings','elapsed_seconds')

def analyze(directory):
    root=Path(directory);p=json.loads((root/'protocol.json').read_text())
    rows=[json.loads(f.read_text()) for f in sorted((root/'summaries').glob('*.json'))]
    keys={(r['seed'],r['initialization'],r['mode'],r['case']) for r in rows}
    expected={(s,i,m,c) for s in p['seeds'] for i in p['initializations'] for m in p['modes'] for c in p['cases']}
    if keys!=expected or len(rows)!=len(expected):raise ValueError('incomplete or duplicate lives')
    rng=np.random.default_rng(987622)
    draws=rng.integers(0,len(p['seeds']),(10000,len(p['seeds'])))
    averages={};contrasts=[]
    def values(case,mode,key):
        return np.array([np.mean([r[key] for r in rows if r['seed']==s and r['case']==case and r['mode']==mode]) for s in p['seeds']])
    def record(case,name,key,delta):
        ci=np.quantile(delta[draws].mean(1),[.025,.975])
        contrasts.append(dict(case=case,contrast=name,metric=key,delta=float(delta.mean()),ci95=ci.tolist(),
             world_sd=float(delta.std(ddof=1)),n_worlds=len(delta),paired_differences=delta.tolist()))
    for case in p['cases']:
        averages[case]={}
        for mode in p['modes']:
            chosen=[r for r in rows if r['case']==case and r['mode']==mode]
            v={key:float(np.mean([r[key] for r in chosen])) for key in METRICS}
            v.update(lives=len(chosen),survivors=sum(r['final_alive'] for r in chosen),
                     diagnostics_triggered=sum(bool(r['diagnostics']) for r in chosen))
            averages[case][mode]=v
        pairs=[('R11','R00'),('R10','R00'),('R01','R00'),('R11','R10'),('R11','R01'),('R11','RT'),('R11','A'),('RT','R10')]
        for a,b in pairs:
            if a not in p['modes'] or b not in p['modes']:continue
            for key in ('challenge_burden','total_burden','final_alive','live_structural_denials','mean_movement_brier','mean_entropy'):
                record(case,a+'-'+b,key,values(case,a,key)-values(case,b,key))
        if set(('R00','R10','R01','R11')).issubset(p['modes']):
            for key in ('challenge_burden','final_alive'):
                record(case,'interaction',key,values(case,'R11',key)-values(case,'R10',key)-values(case,'R01',key)+values(case,'R00',key))
    diagnostic=[]
    for case in p['cases']:
        for mode in p['modes']:
            ds=[d for r in rows if r['case']==case and r['mode']==mode for d in r['diagnostics']]
            if not ds:continue
            summary=dict(case=case,mode=mode,checkpoints=len(ds),exact_restores=sum(d['exact_restore'] for d in ds),parent_unchanged=all(d['parent_unchanged'] for d in ds))
            for kind in ('related_old','matched_other'):
                summary[kind]=dict(mean_burden_delta=float(np.mean([d['branches'][kind]['mean_burden']-d['branches']['intact']['mean_burden'] for d in ds])),
                  mean_first_tv=float(np.mean([d['branches'][kind]['first_total_variation'] for d in ds])),
                  changed_checkpoints=sum(d['branches'][kind]['common_live_action_changes']>0 for d in ds),
                  mean_live_action_changes=float(np.mean([d['branches'][kind]['common_live_action_changes'] for d in ds])))
            diagnostic.append(summary)
    result=dict(protocol_digest=p['protocol_digest'],complete_lives=len(rows),averages=averages,contrasts=contrasts,diagnostics=diagnostic,
        interpretation='Primary contrast predeclared; secondary intervals unadjusted. Steps, branches and crossed fixed initializations are not independent worlds.')
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory');args=a.parse_args();r=analyze(args.directory)
    print(json.dumps(dict(averages={c:{m:{k:v[k] for k in ('challenge_burden','survivors','lives','mean_tau','mean_entropy','live_structural_denials')} for m,v in modes.items()} for c,modes in r['averages'].items()},
      primary=[v for v in r['contrasts'] if v['contrast']=='R11-R00' and v['metric']=='challenge_burden'],diagnostics=r['diagnostics']),indent=2))

