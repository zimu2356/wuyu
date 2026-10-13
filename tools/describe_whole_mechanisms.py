"""Descriptive readout only; does not alter the preregistered primary analysis."""
import json
from pathlib import Path
import sys
import numpy as np

def summarize(directory):
    root=Path(directory);rows=[]
    for path in sorted((root/'summaries').glob('*.json')):
        s=json.loads(path.read_text());tr=np.load(root/'traces'/(path.stem+'.npz'))['trace']
        active=np.r_[True,tr['alive'][:-1]]
        y=dict(seed=s['seed'],initialization=s['initialization'],case=s['case'],mode=s['mode'])
        y.update(live_blocked_takes=int(np.sum(tr['blocked_take']&active)),
                 live_route_denials=int(np.sum(tr['route_denied']&active)),
                 live_action_changes=int(np.sum((tr['action']!=tr['base_choice'])&active)),
                 live_prediction_mse=float(tr['prediction_mse'][active].mean()),
                 ghost_blocked_flags=int(np.sum(tr['blocked_take']&~active)),
                 ghost_route_flags=int(np.sum(tr['route_denied']&~active)))
        if s['mode']!='A':
            u=tr['write_gate'][active];e=tr['erase_reset_gate'][active]
            carry=1-u if s['mode']=='C' else (1-u)*(1-e)
            y.update(mean_write=float(u.mean()),mean_erase_or_reset=float(e.mean()),
                mean_direct_carry=float(carry.mean()),mean_state_norm=float(np.linalg.norm(tr['state'][active],axis=1).mean()),
                mean_residual_max=float(np.abs(tr['action_residual'][active]).max(1).mean()),
                max_residual=float(np.abs(tr['action_residual'][active]).max()))
        y['windows']=s['windows'];rows.append(y)
    agg={}
    for case in ('challenge','stable'):
        agg[case]={}
        for mode in 'ABCD':
            selected=[r for r in rows if r['case']==case and r['mode']==mode]
            v=dict(windows=[dict(start=start,mean_burden=float(np.mean([w['body_burden'] for r in selected for w in r['windows'] if w['start']==start])),
                 alive_at_start=sum(w['alive_at_start'] for r in selected for w in r['windows'] if w['start']==start),
                 partial_device_at_start=sum(w['partial_device_at_start'] for r in selected for w in r['windows'] if w['start']==start),
                 mean_blocked_takes=float(np.mean([w['blocked_takes'] for r in selected for w in r['windows'] if w['start']==start])))
                 for start in (1024,1408,1792,2176)])
            if mode!='A':
                for key in ('mean_write','mean_erase_or_reset','mean_direct_carry','mean_state_norm','mean_residual_max','max_residual'):
                    v[key]=float(np.mean([r[key] for r in selected]))
            for key in ('live_blocked_takes','live_route_denials','live_action_changes','live_prediction_mse','ghost_blocked_flags','ghost_route_flags'):
                v[key]=float(np.mean([r[key] for r in selected]))
            agg[case][mode]=v
    probes=[]
    for path in sorted((root/'diagnostics').glob('*_meta.json')):
        meta=json.loads(path.read_text());label=path.name[:-len('_meta.json')]
        intact=np.load(path.parent/(label+'_intact.npz'))['trace']
        base_live=np.r_[meta['alive'],intact['alive'][:-1]]
        q=dict(label=label,mode=label.split('_')[3],alive=meta['alive'])
        for kind in ('suppressed','matched_history'):
            branch=np.load(path.parent/(label+'_'+kind+'.npz'))['trace']
            common=base_live&np.r_[meta['alive'],branch['alive'][:-1]]
            q[kind]=dict(common_live_action_changes=int(np.sum((branch['action']!=intact['action'])&common)),
                         body_burden_delta=float((branch['burden']-intact['burden']).mean()))
        probes.append(q)
    result=dict(scope='Descriptive; gates are not identified goal contents and direct carry is not total memory lifetime.',
                counter_correction='Frozen raw blocked/movement flags describe issued actions, including absorbing-clock steps after death. Physical-behavior counts here filter to pre-step alive; primary loss and survival analysis unchanged.',
                aggregate=agg,lives=rows,live_diagnostics=probes)
    (root/'mechanism_readout.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    r=summarize(sys.argv[1]);print(json.dumps(r['aggregate'],indent=2))
