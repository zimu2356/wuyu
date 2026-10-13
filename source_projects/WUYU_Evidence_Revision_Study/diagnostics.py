"""Development-only competing explanations and world feasibility; separate from main lives."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, json, math, random
from copy import deepcopy
from pathlib import Path
import numpy as np
from core import Agent, World, features, physical_loss, digest
from experiment import branch, noiseless_deltas
from storage import read_gz_json, write_gz_json, write_json, read_events

def reconstruct_checkpoint(folder,seed,c):
    """Recover an omitted diagnostic snapshot by exact whole-life replay, not by inference."""
    stem=f'{seed}_local_swap_B0';rows=read_events(Path(folder)/(stem+'.npy.gz'))
    meta=read_gz_json(Path(folder)/(stem+'.json.gz'));w=World(seed,'local_swap',c);a=Agent(seed,c,'B0',5);cp=None
    for t,row in enumerate(rows):
        o=w.observation();act,d=a.decide(o);assert act==int(row['action'])
        np.testing.assert_array_equal(d['pred'][act],row['prediction'])
        after=w.step(act);np.testing.assert_array_equal([*after['body'],after['fatigue']],row['after'])
        a.observe(o,act,after,f'{seed}:local_swap:B0:{t}',d['pred'][act])
        if w.tick==w.tau+128: cp=dict(world=w.snapshot(),agent=a.snapshot())
    assert digest(dict(world=w.snapshot(),agent=a.snapshot()))==meta['summary']['final_state_sha']
    assert cp is not None
    return cp

def sampling(w,a,c,mode):
    events=[]
    for k in range(c['diagnostic_sampling_actions']):
        o=w.observation()
        if mode=='round_robin': action=1+k%4
        elif mode=='oldest': action=min(range(1,5),key=lambda v:(a.last_use[v],v))
        else: action=1+int(np.argmax(a.candidates(o)[2][1:]))
        for action in [action]+[0]*c['diagnostic_waits_per_probe']:
            o=w.observation();pred=a.models[action].predict_x(features(o));after=w.step(action)
            a.observe(o,action,after,f'fixed_stream:{o["tick"]}',pred)
            events.append(dict(before=o,after=after,action=action,prediction=pred.tolist()))
    return events

def predictor_grid(w,a):
    result=[]
    for action,m in enumerate(a.models):
        support=[s for s in list(m.samples)[-4:]]
        states=[]
        for s in support:
            x=s['x'];states.append(('recorded_support',dict(tick=w.tick,body=x[1:3],fatigue=x[3],material=x[4])))
        center=np.median(np.array([s['x'] for s in m.samples]),axis=0) if m.samples else np.array([1.,0.,0.,.25,.5,.125])
        states += [('material_interpolation',dict(tick=w.tick,body=center[1:3].tolist(),fatigue=float(center[3]),material=.5)),
                   ('body_fatigue_extrapolation',dict(tick=w.tick,body=[.8,-.8],fatigue=.95,material=.5)),
                   ('boundary',dict(tick=w.tick,body=[.99,-.99],fatigue=.99,material=.8))]
        for label,o in states:
            truth=noiseless_deltas(w,o)[action,:2]
            result.append(dict(group=label,action=action,n=len(m.samples),
                conditional_mse=float(np.mean((m.predict_x(features(o))[:2]-truth)**2)),
                mean_mse=float(np.mean((m.predict_x(features(o),'mean')[:2]-truth)**2)),coverage=m.evidence(o)['coverage']))
    return result

def development_diagnostics(folder,c,out):
    records=[]
    for seed in range(c['development_seeds']['start'],c['development_seeds']['start']+c['development_seeds']['count']):
        cp=read_gz_json(Path(folder)/f'{seed}_local_swap_B0.json.gz')['diagnostic_checkpoint']
        recovered=cp is None
        if recovered: cp=reconstruct_checkpoint(folder,seed,c)
        sampling_results={};fixed_events=None
        for mode in ('policy_ranked_tools','round_robin','oldest'):
            w=World.restore(cp['world'],c);a=Agent.restore(cp['agent'],c);events=sampling(w,a,c,mode)
            if mode=='round_robin': fixed_events=events
            evaluation=branch(w,a,c,c['diagnostic_followup'])
            sampling_results[mode]=dict(sampling_events=events,evaluation=evaluation,primitive_probe_steps=len(events),active_probes=sum(e['action']>0 for e in events))
        replays={}
        for rule in ('FIFO','calendar_decay','local_revision'):
            a=Agent.restore(cp['agent'],c);a.method='BU' if rule=='local_revision' else 'B0';errors=[];revisions=0
            for e in fixed_events:
                o=e['before'];act=e['action'];m=a.models[act]
                if rule=='calendar_decay':
                    # A fixed 64-tick half-life comparator, using exactly the same retained capacity/data.
                    for s in m.samples: s['w']=2**(-(o['tick']-s['tick'])/64)
                    m.rebuild();m.fit()
                p=m.predict_x(features(o));err,revision=a.observe(o,act,e['after'],f'fixed_stream:{o["tick"]}',p)
                errors.append(err**2/2);revisions+=revision
            w=World.restore(cp['world'],c)
            w.tick=fixed_events[-1]['after']['tick'];w.body=fixed_events[-1]['after']['body'];w.fatigue=fixed_events[-1]['after']['fatigue']
            replays[rule]=dict(prequential_mse=float(np.mean(errors)),revisions=revisions,grid=predictor_grid(w,a),final_agent=a.snapshot())
        coordination={}
        for mode in ('learned','truth'):
            coordination[mode]=branch(World.restore(cp['world'],c),Agent.restore(cp['agent'],c),c,c['diagnostic_followup'],mode)
        records.append(dict(seed=seed,source_checkpoint=cp,checkpoint_recovered_by_verified_full_replay=recovered,sampling=sampling_results,same_data=replays,
                            predictor_before=predictor_grid(World.restore(cp['world'],c),Agent.restore(cp['agent'],c)),coordination=coordination))
    write_gz_json(Path(out)/'development_diagnostics.json.gz',records)
    summary=dict(n=len(records),recovered_checkpoints=sum(r['checkpoint_recovered_by_verified_full_replay'] for r in records),sampling={mode:float(np.mean([r['sampling'][mode]['evaluation']['mean_loss'] for r in records])) for mode in ('policy_ranked_tools','round_robin','oldest')},
        same_data_mse={rule:float(np.mean([r['same_data'][rule]['prequential_mse'] for r in records])) for rule in ('FIFO','calendar_decay','local_revision')},
        coordination={mode:float(np.mean([r['coordination'][mode]['mean_loss'] for r in records])) for mode in ('learned','truth')})
    summary['predictor_before']={}
    for group in ('recorded_support','material_interpolation','body_fatigue_extrapolation','boundary'):
        rr=[q for r in records for q in r['predictor_before'] if q['group']==group]
        summary['predictor_before'][group]={k:float(np.mean([q[k] for q in rr])) for k in ('conditional_mse','mean_mse','coverage')}
    write_json(Path(out)/'DIAGNOSTICS_SUMMARY.json',summary);return summary

def feasibility(c,out):
    records=[]
    for seed in range(c['pilot_seeds']['start'],c['pilot_seeds']['start']+c['pilot_seeds']['count']):
        for case in c['cases']:
            for kind in ('noiseless_one_step','random'):
                w=World(seed,case,c);rng=random.Random(seed+12119);loss=[];actions=[]
                for t in range(c['life_ticks']):
                    o=w.observation()
                    if kind=='random': act=rng.randrange(w.n_actions)
                    else:
                        aft=[w.expected_after(o,k) for k in range(w.n_actions)]
                        values=[physical_loss(z)+c['fatigue_penalty']*z['fatigue']**2+(c['action_cost'] if k else 0.) for k,z in enumerate(aft)]
                        act=int(np.argmin(values))
                    after=w.step(act);loss.append(physical_loss(after));actions.append(act)
                records.append(dict(seed=seed,case=case,kind=kind,post_loss=float(np.mean(loss[w.tau:w.tau+c['post_window']])),life_loss=float(np.mean(loss)),spare_uses=sum(x==5 for x in actions[w.tau:w.tau+c['post_window']]),tau=w.tau))
        # Opportunity genuinely improves attainable outcomes under a fixed privileged reference.
        for allowed in (5,6):
            w=World(seed,'opportunity',c);w.tick=w.tau;w.body=[.4,.4];w.fatigue=.25;loss=[]
            for _ in range(512):
                o=w.observation();aft=[w.expected_after(o,k) for k in range(allowed)]
                act=min(range(allowed),key=lambda k:physical_loss(aft[k])+c['fatigue_penalty']*aft[k]['fatigue']**2+(c['action_cost'] if k else 0.))
                loss.append(physical_loss(w.step(act)))
            records.append(dict(seed=seed,case='spare_feasibility',kind=f'allowed_{allowed}',post_loss=float(np.mean(loss))))
    write_json(Path(out)/'FEASIBILITY.json',records);return records

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);ap.add_argument('--data');args=ap.parse_args()
    c=json.loads(Path(args.config).read_text());Path(args.out).mkdir(parents=True,exist_ok=True)
    print(json.dumps(development_diagnostics(args.data,c,args.out) if args.data else feasibility(c,args.out),indent=2))
