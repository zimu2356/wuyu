"""Pre-specified outer search, held-out selection, and one formal evaluation."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from wuyu05.experiment import candidates,run,broadcast

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
TRAIN=np.arange(851000,851008);VAL=np.arange(852000,852008);TEST=np.arange(853000,853032)


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')


def hashes():
    paths=[ROOT/'PROTOCOL.md',ROOT/'run_experiment.py',*sorted((ROOT/'wuyu05').glob('*.py')),*sorted((ROOT/'tests').glob('*.py'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train(mode,seed):
    rng=np.random.default_rng(seed);shape=(2,8) if mode=='learned' else (2,)
    mean=np.zeros(shape)
    if mode=='learned':mean[:,0]=-1
    else:mean[:]=-1
    std=np.full(shape,.8);history=[]
    for generation in range(4):
        population=mean+rng.normal(size=(12,)+shape)*std;population[0]=mean
        values,seconds=candidates(TRAIN,mode,population);scores=values.mean(axis=1)
        for i in range(12):history.append({'generation':generation,'candidate':i,'theta':population[i].tolist(),
                                        'fitness':float(scores[i]),'individual_fitness':values[i].tolist()})
        elite=np.argsort(-scores,kind='stable')[:3]
        mean=.5*mean+.5*population[elite].mean(axis=0)
        std=np.maximum(.15,.5*std+.5*population[elite].std(axis=0))
        write('search_'+mode+'.json',{'mode':mode,'history':history,'mean':mean,'std':std})
        print(json.dumps({'stage':'train','mode':mode,'generation':generation,'best':float(scores.max()),'seconds':round(seconds,2)}),flush=True)
    ordered=sorted(enumerate(history),key=lambda v:(-v[1]['fitness'],v[0]))[:3]
    validation=[]
    for index,h in ordered:
        r=run(VAL,mode,broadcast(h['theta'],len(VAL)))
        validation.append({'search_index':index,'theta':h['theta'],'fitness':float(r['fitness'].mean()),
                           'individual_fitness':r['fitness'].tolist()})
    winner=max(range(3),key=lambda i:(validation[i]['fitness'],-i))
    write('validation_'+mode+'.json',{'candidates':validation,'winner':winner})
    print(json.dumps({'stage':'selected','mode':mode,'validation':validation[winner]['fitness']}),flush=True)
    return validation[winner]


def save_raw(mode,result):
    data={'seeds':TEST,'fitness':result['fitness'],'branch_rewards':result['rewards'],
          'prefix_b':result['prefix_b'],'final_b':result['final_b']}
    for key,values in result['prefix'].items():data['prefix__'+key]=values
    for kind,records in result['branches'].items():
        for key,values in records.items():data[kind+'__'+key]=values
    np.savez_compressed(OUT/('raw_'+mode+'.npz'),**data)
    write('runtime_'+mode+'.json',{'seconds':result['seconds'],'resources':result['resources'],
                                 'fitness':result['fitness'],'prefix_b':result['prefix_b'],'final_b':result['final_b']})


def main():
    OUT.mkdir(exist_ok=True);frozen=hashes();write('frozen_sources.json',frozen)
    start=time.perf_counter();selection={}
    write('manifest.json',{'status':'running','started':datetime.now(timezone.utc).isoformat(),'numpy':np.__version__,
                           'train':TRAIN,'validation':VAL,'test':TEST})
    for mode,seed in [('fixed',85001),('heuristic',85002),('learned',85003)]:selection[mode]=train(mode,seed)
    write('selection.json',selection)
    dev=run(TRAIN,'learned',broadcast(selection['learned']['theta'],len(TRAIN)),record=True)
    traces=[dev['prefix'],*dev['branches'].values()]
    all_rates=np.concatenate([np.stack([r['rate_current'],r['rate_replay']],axis=-1).reshape(-1,2) for r in traces])
    valid=np.concatenate([r['valid'].ravel() for r in traces]);rv=np.concatenate([r['replay_valid'].ravel() for r in traces])
    matched=np.array([all_rates[valid,0].mean(),all_rates[rv,1].mean()])
    pool=all_rates[rv].copy()
    np.savez_compressed(OUT/'development_rates.npz',rates=all_rates,valid=valid,replay_valid=rv,pool=pool,matched=matched)
    selection['matched_mean']={'theta':matched.tolist()};selection['shuffled']={'pool_rows':len(pool)}
    selection['frozen']={'theta':selection['fixed']['theta']};write('selection.json',selection)
    print(json.dumps({'stage':'controls_ready','matched':matched.tolist(),'pool_rows':len(pool)}),flush=True)
    for mode in ('fixed','heuristic','learned','matched_mean','shuffled','frozen'):
        theta=np.zeros(2) if mode=='shuffled' else np.asarray(selection[mode]['theta'])
        r=run(TEST,mode,broadcast(theta,len(TEST)),formal=True,pool=pool if mode=='shuffled' else None,output=OUT,record=True)
        save_raw(mode,r)
        print(json.dumps({'stage':'formal','mode':mode,'reward':float(r['fitness'].mean()),
                          'branch_rewards':r['rewards'].mean(axis=1).tolist(),'seconds':round(r['seconds'],2)}),flush=True)
    assert frozen==hashes(),'Numerical experiment source changed after freeze'
    write('manifest.json',{'status':'completed','completed':datetime.now(timezone.utc).isoformat(),
                           'seconds':time.perf_counter()-start,'numpy':np.__version__,
                           'train':TRAIN,'validation':VAL,'test':TEST,'frozen_sources_unchanged':True,
                           'formal_modes':['fixed','heuristic','learned','matched_mean','shuffled','frozen']})
    print(json.dumps({'stage':'completed','seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()
