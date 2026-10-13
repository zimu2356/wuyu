"""All ordinary lives retain every step. Privileged forks are separately marked."""
from copy import deepcopy
import math
import numpy as np
from core import Agent, World, features, digest, physical_loss
from storage import DTYPE, write_events, write_gz_json, sha

def noiseless_deltas(w,o):
    out=[]
    for a in range(w.n_actions):
        after=w.expected_after(o,a)
        out.append([after['body'][i]-o['body'][i] for i in range(2)]+[after['fatigue']-o['fatigue']])
    return np.asarray(out)

def probes(w,a,c):
    current=w.observation();states=[('actual',current)]
    states += [('matched_axis0',dict(tick=w.tick,body=[.6,0.],fatigue=.25,material=.5)),
               ('matched_axis1',dict(tick=w.tick,body=[0.,.6],fatigue=.25,material=.5))]
    rows=[]
    for label,o in states:
        pred,_,scores,_=a.candidates(o);truth=noiseless_deltas(w,o);selected=int(np.argmax(scores))
        for action,m in enumerate(a.models):
            e=m.evidence(o);old=sum(s['w'] for s in m.samples if s['tick']<w.tau)
            rows.append(dict(state=label,action=action,selected=action==selected,n=len(m.samples),**e,
                             old_mass_fraction=old/m.mass if m.mass>0 else 0.,version=m.version,revisions=m.revisions,
                             mse=float(np.mean((pred[action,:2]-truth[action,:2])**2)),
                             mean_mse=float(np.mean((m.predict_x(features(o),'mean')[:2]-truth[action,:2])**2))))
    return dict(tick=w.tick,rows=rows)

def branch(w,a,c,steps,mode='learned'):
    rows=[]
    for _ in range(steps):
        o=w.observation();override=noiseless_deltas(w,o) if mode=='truth' else None
        action,decision=a.decide(o,override=override);after=w.step(action)
        error,revision=a.observe(o,action,after,f'diagnostic:{o["tick"]}',decision['pred'][action])
        rows.append(dict(tick=o['tick'],action=action,before=o,after=after,prediction=decision['pred'][action].tolist(),
                         residual=error,revision=revision,loss=physical_loss(after)))
    return dict(events=rows,mean_loss=float(np.mean([r['loss'] for r in rows])),final_world=w.snapshot(),final_agent=a.snapshot())

def causal_forks(w,pre_agent,normal,event,c):
    blocked=Agent.restore(pre_agent,c)
    # pre_agent is captured after deciding; bookkeeping is equal, target evidence is not incorporated.
    blocked.observe(event['before'],event['action'],event['after'],event['id'],np.asarray(event['prediction']),block=True)
    restored=Agent.restore(blocked.snapshot(),c)
    target=event['action'];restored.models[target]=deepcopy(normal.models[target])
    assert digest(restored.snapshot())==digest(normal.snapshot())
    snapshots={'normal':normal.snapshot(),'block_target':blocked.snapshot(),
               'block_unrelated_sham':normal.snapshot(),'restore':restored.snapshot()}
    results={}
    for name,s in snapshots.items():
        results[name]=branch(World.restore(w.snapshot(),c),Agent.restore(s,c),c,c['causal_followup'])
    assert digest(results['normal'])==digest(results['restore'])==digest(results['block_unrelated_sham'])
    return dict(event=event,pre_agent=pre_agent,post_world=w.snapshot(),post_agents=snapshots,branches=results,
                interpretation='Target chosen as first BEU revision of a changed-axis tool after tau; missing cases kept. Unrelated block is an exact sham because only selected model receives the observation.')

def recovery(loss,tau,c):
    window=c['post_window'];threshold=max(.02,2*float(np.mean(loss[tau-256:tau])))
    for delay in range(window-127):
        i=tau+delay
        if np.mean(loss[i:i+64])<=threshold and np.mean(loss[i+64:i+128])<=threshold:
            return dict(recovered=True,delay=delay+128,threshold=threshold)
    return dict(recovered=False,delay=window,threshold=threshold)

def run_life(seed,case,method,c,outdir,development=False):
    w=World(seed,case,c);a=Agent(seed,c,method,w.n_actions);n=c['life_ticks'];rows=np.zeros(n,dtype=DTYPE)
    checkpoint_ticks={w.tau,w.tau+128,w.tau+c['post_window'],n}
    if case=='recurrence': checkpoint_ticks.add(w.tau+c['recurrence_gap']+c['post_window'])
    checkpoints=[];causal=None;diagnostic_checkpoint=None
    target_actions=[i for i in range(1,5) if w.tools[i]['axis']==w.changed_axis]
    first_use={str(x):None for x in target_actions};evidence_three={str(x):None for x in target_actions};new_counts={x:0 for x in target_actions}
    first_revision=None;first_prediction_accuracy=None;unaffected_revisions=0;pre_revisions=0;post_revisions=0
    for tick in range(n):
        o=w.observation();action,d=a.decide(o);m=a.models[action]
        mass=m.mass;ess=mass**2/max(m.mass2,1e-30);age=tick-a.last_use[action]
        # Capture only potential causal event states; no evaluator information enters agent.
        candidate=case=='local_swap' and method=='BEU' and causal is None and tick>=w.tau and action in target_actions
        pre=a.snapshot() if candidate else None
        after=w.step(action);event_id=f'{seed}:{case}:{method}:{tick}'
        residual,revision=a.observe(o,action,after,event_id,d['pred'][action])
        rows[tick]=(action,d['purpose'],revision,m.streak,d['base_choice'],[*after['body'],after['fatigue']],
                    d['pred'][action],float(d['score'][action]),residual,age,min(a.visits[action]-1,c['model_window']),mass,ess,m.version-1)
        if tick>=w.tau and action in target_actions:
            new_counts[action]+=1
            if first_use[str(action)] is None: first_use[str(action)]=tick-w.tau
            if new_counts[action]==3: evidence_three[str(action)]=tick-w.tau
            if first_prediction_accuracy is None:
                errors=[]
                for material in (.2,.8):
                    test=dict(tick=w.tick,body=[.2,.2],fatigue=.25,material=material)
                    truth=noiseless_deltas(w,test)
                    for target in target_actions:
                        errors.append(float(np.mean((a.models[target].predict_x(features(test))[:2]-truth[target,:2])**2)))
                if max(errors)<.0025: first_prediction_accuracy=tick-w.tau
        if revision:
            if tick<w.tau: pre_revisions+=1
            else:
                post_revisions+=1
                if first_revision is None and action in target_actions: first_revision=tick-w.tau
                if action not in target_actions: unaffected_revisions+=1
            if candidate:
                event=dict(id=event_id,tick=tick,action=action,before=o,after=after,prediction=d['pred'][action].tolist(),
                           pre_model_sha=digest(pre['models'][action]),post_model_sha=digest(a.models[action].snapshot()))
                causal=causal_forks(w,pre,a,event,c)
        if w.tick in checkpoint_ticks:
            checkpoints.append(probes(w,a,c))
            if development and case=='local_swap' and method=='B0' and w.tick==w.tau+128:
                diagnostic_checkpoint=dict(world=w.snapshot(),agent=a.snapshot())
    loss=np.sum(rows['after'][:,:2]**2,axis=1);post=slice(w.tau,w.tau+c['post_window'])
    summary=dict(seed=seed,case=case,method=method,tau=w.tau,changed_axis=w.changed_axis,n_actions=w.n_actions,
                 post_loss=float(np.mean(loss[post])),life_loss=float(np.mean(loss)),pre_loss=float(np.mean(loss[w.tau-256:w.tau])),
                 late_loss=float(np.mean(loss[-400:])),active_fraction=float(np.mean(rows['action']>0)),
                 post_active_fraction=float(np.mean(rows['action'][post]>0)),retest_fraction=float(np.mean(rows['purpose']>0)),
                 activity_cost=float(np.sum(rows['action']>0)*c['action_cost']),updates=int(sum(a.visits)),
                 pre_revisions=pre_revisions,post_revisions=post_revisions,unaffected_post_revisions=unaffected_revisions,
                 first_retest=first_use,three_new_samples=evidence_three,first_revision=first_revision,first_prediction_accuracy=first_prediction_accuracy,
                 recovery=recovery(loss,w.tau,c),causal_event=causal['event']['tick'] if causal else None,
                 final_state_sha=digest(dict(world=w.snapshot(),agent=a.snapshot())))
    if case=='opportunity':
        uses=np.flatnonzero(rows['action'][post]==5)
        summary['first_spare_use']=int(uses[0]) if len(uses) else None
        summary['post_spare_uses']=int(len(uses))
    if case=='recurrence': summary['return_loss']=float(np.mean(loss[w.tau+c['recurrence_gap']:w.tau+c['recurrence_gap']+c['post_window']]))
    if causal:
        summary['causal_block_minus_normal']=causal['branches']['block_target']['mean_loss']-causal['branches']['normal']['mean_loss']
        summary['causal_first_action_differs']=causal['branches']['block_target']['events'][0]['action']!=causal['branches']['normal']['events'][0]['action']
    stem=f'{seed}_{case}_{method}';ep=outdir/(stem+'.npy.gz');mp=outdir/(stem+'.json.gz')
    write_events(ep,rows)
    write_gz_json(mp,dict(summary=summary,checkpoints=checkpoints,final_world=w.snapshot(),final_agent=a.snapshot(),
                          diagnostic_checkpoint=diagnostic_checkpoint,causal=causal))
    return dict(summary=summary,files={ep.name:sha(ep),mp.name:sha(mp)})
