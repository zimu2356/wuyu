"""Finite continuing lives and isolated, matched diagnostic branches."""
from copy import deepcopy
import math

from core import Agent,World,digest,evaluation_loss,intervene,physical_loss,predicted_observation


def execute(world,agent,seed,branch,emit):
    before=world.observation();action,decision=agent.decide(before)
    after=world.step(action);event_id=f'{seed}:{branch}:{before["tick"]}'
    agent.observe(before,action,after,event_id)
    row=dict(seed=seed,branch=branch,event_id=event_id,before=before,action=action,decision=decision,after=after)
    emit('events',row)
    chosen=decision['candidates'][action]
    pred=predicted_observation(before,chosen['predicted_delta'])
    return row,dict(loss=physical_loss(after),wait=int(action==0),fatigue=after['fatigue'],
                    prediction_mse=sum((pred['body'][i]-after['body'][i])**2 for i in range(2))/2,
                    saturated=int(any(abs(b)>=.999999 for b in after['body'])))


def observation_probe(world,task,config):
    world.body=[config['probe_body_level'] if i==task else 0.0 for i in range(2)]
    world.fatigue=config['probe_fatigue'];world.material_override=config['probe_material']
    return world.observation()


def probe_predictions(world,agent,obs,kind):
    details=[]
    for action in range(5):
        pred=predicted_observation(obs,agent.models[action].predict(obs,kind))
        target=predicted_observation(obs,world.mean_delta(obs,action))
        details.append(dict(action=action,predicted=pred,target_noiseless=target,
                            body_mse=sum((pred['body'][i]-target['body'][i])**2 for i in range(2))/2,
                            fatigue_mse=(pred['fatigue']-target['fatigue'])**2))
    return details


def checkpoint_diagnostics(seed,world,agent,config,emit):
    checkpoint=world.tick;world_state=world.snapshot();agent_state=agent.snapshot()
    original_hash=digest(agent_state)
    key=dict(seed=seed,checkpoint=checkpoint)
    emit('checkpoints',dict(**key,world=world_state,agent=agent_state,agent_sha256=original_hash))
    rows=[];prediction_rows=[]
    for intervention in config['interventions']:
        fork_state,change=intervene(agent_state,intervention,world.tools,config)
        emit('interventions',dict(**key,**change,agent=fork_state))
        for task in config['probe_tasks']:
            probe_world=World.restore(world_state,config)
            obs=observation_probe(probe_world,task,config)
            branch_agent=Agent.restore(fork_state,config)
            pred=probe_predictions(probe_world,branch_agent,obs,'conditional')
            for kind in config['predictor_comparisons']:
                if kind!='conditional' and intervention!='retain': continue
                details=pred if kind=='conditional' else probe_predictions(probe_world,branch_agent,obs,kind)
                for d in details:
                    pr=dict(**key,intervention=intervention,task=task,predictor=kind,**d)
                    emit('predictions',pr)
                    prediction_rows.append({k:pr[k] for k in ('seed','checkpoint','intervention','task','predictor','action','body_mse','fatigue_mse')})
            true_losses=[evaluation_loss(d['target_noiseless'],config)+(config['action_cost'] if d['action'] else 0) for d in pred]
            oracle=min(true_losses)
            branch=f'probe_{checkpoint}_{intervention}_{task}'
            outcomes=[];first_action=None
            for step in range(config['probe_ticks']):
                event,metrics=execute(probe_world,branch_agent,seed,branch,emit)
                if first_action is None:first_action=event['action']
                outcomes.append(metrics)
            row=dict(**key,intervention=intervention,task=task,body_loss=sum(x['loss'] for x in outcomes)/len(outcomes),
                     final_body_loss=outcomes[-1]['loss'],mean_fatigue=sum(x['fatigue'] for x in outcomes)/len(outcomes),
                     waits=sum(x['wait'] for x in outcomes),first_action=first_action,
                     first_action_regret=true_losses[first_action]-oracle,uniform_action_regret=sum(true_losses)/5-oracle,
                     predictor_body_mse=sum(x['body_mse'] for x in pred)/5,
                     full_tool_buffers=int(all(len(m['samples'])==config['model_window'] for m in agent_state['models'][1:])),
                     removed_samples=sum(x['n_samples'] for x in change['removed']),
                     initial_agent_sha256=digest(fork_state),original_agent_sha256=original_hash)
            rows.append(row)
    if digest(agent.snapshot())!=original_hash or world.snapshot()!=world_state:
        raise AssertionError('Diagnostic branches contaminated main life')
    return rows,prediction_rows


def summarize_window(seed,branch,start,end,metrics):
    selected=metrics[start:end]
    return dict(seed=seed,branch=branch,start=start,end=end,n=len(selected),
                mean_body_loss=sum(x['loss'] for x in selected)/len(selected),
                mean_fatigue=sum(x['fatigue'] for x in selected)/len(selected),
                prediction_mse=sum(x['prediction_mse'] for x in selected)/len(selected),
                waits=sum(x['wait'] for x in selected),saturated=sum(x['saturated'] for x in selected))


def run_individual(seed,config,emit):
    world=World(seed,config);agent=Agent(seed,config)
    probes=[];predictions=[];main_metrics=[];windows=[];frozen_origin=None
    for _ in range(config['life_ticks']):
        _,metric=execute(world,agent,seed,'life',emit);main_metrics.append(metric)
        if world.tick==config['change_tick']:
            frozen_origin=(world.snapshot(),agent.snapshot())
            emit('checkpoints',dict(seed=seed,checkpoint=world.tick,world=frozen_origin[0],agent=frozen_origin[1],
                                    agent_sha256=digest(frozen_origin[1]),purpose='hidden-change matched continuation'))
        if world.tick in config['checkpoints']:
            p,d=checkpoint_diagnostics(seed,world,agent,config,emit);probes.extend(p);predictions.extend(d)
    change=config['change_tick'];total=config['life_ticks']
    bounds=[(0,400),(change-400,change),(change,change+100),(change+100,change+400),(total-400,total)]
    for start,end in bounds:windows.append(summarize_window(seed,'online',start,end,main_metrics))
    fw=World.restore(frozen_origin[0],config);fa=Agent.restore(frozen_origin[1],config);fa.learn=False
    frozen_metrics=[]
    for _ in range(total-change):
        _,metric=execute(fw,fa,seed,'frozen_after_change',emit);frozen_metrics.append(metric)
    for start,end in [(change,change+100),(change+100,change+400),(total-400,total)]:
        row=summarize_window(seed,'frozen',start-change,end-change,frozen_metrics)
        row['start']=start;row['end']=end;windows.append(row)
    emit('final_states',dict(seed=seed,branch='life',world=world.snapshot(),agent=agent.snapshot()))
    emit('final_states',dict(seed=seed,branch='frozen_after_change',world=fw.snapshot(),agent=fa.snapshot()))
    return probes,predictions,windows
