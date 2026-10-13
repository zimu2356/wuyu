"""Paired observer interventions; probe states never enter formal lives."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import copy,json
import numpy as np
from wuyu06.state import load,read_tree
from wuyu06.network import Network
from wuyu06.world import World
from wuyu06.experiment import rollout

ROOT=Path(__file__).resolve().parent

def tv(a,b):return .5*np.abs(a-b).sum(1)

def response(agent,obs,goal=None,forecast=True):
    a=copy.deepcopy(agent)
    if goal is not None:a.goal=goal.copy();a.need_goal[:]=False
    if not forecast:a.mode='no_forecast'
    return a.act(obs,uniforms=(np.full(a.n,.37),np.full(a.n,.53)))


if __name__=='__main__':
    out=ROOT/'results';all_data=[];arrays={}
    for i in range(3):
        w,a=load(out/f'checkpoint_full_{i}_128.npz');obs=w.observation();live=w.alive
        low=obs.copy();high=obs.copy();low[:,7]=.2;high[:,7]=.85
        energy_tv=tv(response(a,low)['ap'],response(a,high)['ap'])
        row=np.arange(a.n);eg=a.goals[row,a.goals[:,:,4].argmax(1)];rg=a.goals[row,a.goals[:,:,5].argmax(1)]
        both=live & (eg[:,4]>.1) & (rg[:,5]>.1)
        goal_tv=tv(response(a,obs,eg)['ap'],response(a,obs,rg)['ap'])
        original=response(a,obs);forecast_tv=tv(original['ap'],response(a,obs,forecast=False)['ap'])
        # Evaluate all seven actions from identical cloned worlds, not only
        # the actions selected by the policy. Future outcomes are evaluator-only.
        pred=original['pred'].reshape(a.n,7,3)
        frozen=copy.deepcopy(a);frozen.fast[:]=0;frozen.fast_b[:]=0
        pred_frozen=response(frozen,obs)['pred'].reshape(a.n,7,3)
        targets=[]
        for action in range(7):
            v=copy.deepcopy(w);targets.append(v.step(np.full(a.n,action))['target'])
        targets=np.stack(targets,axis=1)
        current_mse=((pred-targets)**2).mean((1,2));zero_mse=((pred_frozen-targets)**2).mean((1,2))
        result={'initialization':i,'alive_count':int(live.sum()),'both_goal_count':int(both.sum()),
                'body_action_tv':float(energy_tv[live].mean()),'goal_action_tv':float(goal_tv[both].mean()) if both.any() else None,
                'forecast_action_tv':float(forecast_tv[live].mean()),'all_action_prediction_mse':float(current_mse[live].mean()),
                'zero_fast_prediction_mse':float(zero_mse[live].mean())}
        # Same birth, different actual developmental histories. Rejoin only on
        # observer-controlled probe copies with matched current physical inputs.
        selected=json.loads((out/f'training_full_{i}.json').read_text())['selected']['checkpoint'];net=Network.from_state(read_tree(out/selected))
        histories=[];worlds=[]
        for variant in [0,1]:
            _,_,v,b,_=rollout(net,np.arange(990000,990032),128,'full',variant=variant);histories.append(b);worlds.append(v)
        common=World(np.arange(990000,990032),256,variant=2);common.body[:]=[.7,.7];common_obs=common.observation()
        paired=worlds[0].alive & worlds[1].alive
        r0=response(histories[0],common_obs);r1=response(histories[1],common_obs)
        history_tv=tv(r0['ap'],r1['ap']);result['history_pair_count']=int(paired.sum());result['history_action_tv']=float(history_tv[paired].mean()) if paired.any() else None
        result['history_note']='Different world histories include different sensory/action/body histories; the probe equalizes current position/body/observation, not all internal states. Difference is not evidence of personality or beneficial transfer.'
        all_data.append(result)
        arrays.update({f'{i}_'+k:v for k,v in dict(live=live,both_goals=both,energy_tv=energy_tv,goal_tv=goal_tv,forecast_tv=forecast_tv,
                     prediction_mse=current_mse,zero_fast_mse=zero_mse,history_tv=history_tv,history_live=paired).items()})
    (out/'probes.json').write_text(json.dumps(all_data,indent=2));np.savez_compressed(out/'probes_raw.npz',**arrays)
    print(json.dumps(all_data,indent=2))
