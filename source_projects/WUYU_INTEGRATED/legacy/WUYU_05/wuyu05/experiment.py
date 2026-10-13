import copy
from pathlib import Path
import time
import numpy as np
from .core import Agent,World,step,evaluate_b,save,KINDS,array_bytes


def stack(records): return {key:np.stack([r[key] for r in records]) for key in records[0]}


def run(seeds,mode,theta,formal=False,pool=None,output=None,record=False):
    started=time.perf_counter();seeds=np.asarray(seeds)
    a=Agent(seeds,mode,theta,pool);w=World(seeds,'prefix',formal)
    prefix=[]
    while w.t<w.total:
        rec=step(w,a)
        if record:prefix.append(rec)
    prefix_eval=evaluate_b(a,seeds) if record else None
    if output is not None and mode=='learned':save(Path(output)/'checkpoint_prefix.npz',w,a)
    result={'rewards':[],'branches':{},'final_b':[],'seconds':0}
    for kind in KINDS:
        b=copy.deepcopy(a);b.branch=True;v=World(seeds,kind,formal);v.charge=w.charge.copy()
        total=np.zeros(len(seeds));records=[]
        while v.t<v.total:
            if output is not None and mode=='learned' and kind=='change' and v.t==400:
                save(Path(output)/'checkpoint_mid.npz',v,b)
            rec=step(v,b);total+=rec['reward']
            if record:records.append(rec)
        result['rewards'].append(total/v.total)
        if record:
            result['branches'][kind]=stack(records);result['final_b'].append(evaluate_b(b,seeds))
            if output is not None:
                np.savez_compressed(Path(output)/('weights_'+mode+'_'+kind+'.npz'),
                    **{k:val for k,val in vars(b.net).items() if isinstance(val,np.ndarray)})
        result['resources']={'neural_parameters':b.net.parameter_count,'policy_parameters':int(np.prod(a.theta.shape[1:])),
             'agent_arrays_bytes_per_individual':array_bytes(b)/b.n,'forward_micro':b.forward_micro,
             'backward_micro':b.backward_micro,'gradient_calls':b.gradient_calls,
             'fresh_nonzero_mean':float(b.fresh_nonzero.mean()),'replay_nonzero_mean':float(b.replay_nonzero.mean())}
    result['rewards']=np.stack(result['rewards']);result['fitness']=result['rewards'].mean(axis=0)
    result['seconds']=time.perf_counter()-started
    if record:
        result['prefix']=stack(prefix);result['prefix_b']=prefix_eval;result['final_b']=np.stack(result['final_b'])
    return result


def broadcast(theta,n): return np.broadcast_to(np.asarray(theta),(n,)+np.asarray(theta).shape).copy()


def candidates(seeds,mode,population):
    seeds=np.asarray(seeds);pop=np.asarray(population)
    result=run(np.tile(seeds,len(pop)),mode,np.repeat(pop,len(seeds),axis=0))
    return result['fitness'].reshape(len(pop),len(seeds)),result['seconds']
