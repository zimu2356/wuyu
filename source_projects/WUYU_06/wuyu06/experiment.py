from pathlib import Path
import copy,json,time
import numpy as np
from .world import World
from .agent import Agent
from .network import Network
from .state import save,write_tree,read_tree


def transition(world,agent):
    obs=world.observation();r=agent.act(obs);out=world.step(r['action'])
    agent.feedback(r,world.observation(),out['reward']);r['world']=out
    return r


def metrics(raw):
    alive=raw['alive'];n=alive.shape[1]
    output={'survival':alive.mean(0),'body_area':(raw['body'].sum(2)*alive).sum(0)/len(alive),
            'reward':raw['reward'].sum(0),'prediction_mse':((raw['pred']-raw['target'])**2).mean(2).sum(0)/np.maximum(1,alive.sum(0)),
            'replay_rate':((raw['action']==4)*alive).sum(0)/np.maximum(1,alive.sum(0)),
            'goal_achievement':(raw['goal_achieved']*alive).sum(0)}
    # Prediction error is only meaningful while the individual can act.
    output['prediction_mse']=(((raw['pred']-raw['target'])**2).mean(2)*alive).sum(0)/np.maximum(1,alive.sum(0))
    for p in range(4):
        mask=raw['phase']==p
        output['phase_'+str(p)+'_survival']=(alive*mask).sum(0)/np.maximum(1,mask.sum(0))
    return output


def rollout(net,seeds,total=2048,mode='full',output=None,name=None,checkpoint=False,variant=0):
    w=World(seeds,total,formal=True,variant=variant);a=Agent(copy.deepcopy(net),seeds,mode)
    records=[];started=time.perf_counter()
    while w.t<w.total:
        if checkpoint and w.t in (128,512):save(Path(output)/f'checkpoint_{name}_{w.t}.npz',w,a)
        r=transition(w,a);rows=np.arange(len(seeds));out=r['world']
        records.append(dict(alive=r['alive'],action=r['action'].astype(np.int8),body=out['body'],position=out['position'].astype(np.int8),
          cue=r['obs'][:,4:7].argmax(1).astype(np.int8),reward=r['reward'],pred=r['pred'].reshape(-1,7,3)[rows,r['action']],target=r['target'],
          goal=r['goal'],goal_origin=r['goal_origin'],goal_achieved=r['goal_achieved'],goal_age=r['goal_age_after'],
          probabilities=r['ap'],replay_origin=r['replay_origin'],phase=out['phase'].astype(np.int8)))
    raw={k:np.stack([r[k] for r in records]) for k in records[0]};m=metrics(raw)
    if output is not None:
        path=Path(output);np.savez_compressed(path/f'raw_{name}.npz',seeds=np.asarray(seeds),**raw)
        save(path/f'final_{name}.npz',w,a)
    return m,raw,w,a,time.perf_counter()-started


def advantages(records,bootstrap,gamma=.99,lam=.95):
    n=len(bootstrap);gae=np.zeros(n);ret=[];adv=[];nv=bootstrap
    for r in reversed(records):
        cont=r['after_alive'];delta=r['reward']+gamma*nv*cont-r['value']
        gae=(delta+gamma*lam*cont*gae)*r['alive'];adv.append(gae.copy());ret.append(gae+r['value']);nv=r['value']
    ret=np.stack(ret[::-1]);adv=np.stack(adv[::-1]);mask=np.stack([r['alive'] for r in records])
    vals=adv[mask]
    if len(vals):adv=np.clip((adv-vals.mean())/(vals.std()+1e-6),-5,5)
    return ret,adv


def train(config,mode,initialization,output):
    start=time.perf_counter();output=Path(output);net=Network(config['model_seeds'][initialization],mode)
    logs=[];selection=[];w=a=None
    for update in range(config['updates']):
        if w is None or w.t>=w.total:
            batch=update//(config['lifetime']//config['unroll'])
            seeds=960000+initialization*100000+batch*config['batch']+np.arange(config['batch'])
            w=World(seeds,config['lifetime']);a=Agent(net,seeds,mode,fast_lr=config['fast_lr'])
        records=[transition(w,a) for _ in range(config['unroll'])]
        x=a.inputs(w.observation());hh=net.encode(x,a.h);bootstrap=hh@net.p['V']+net.p['vb'][0]
        ret,adv=advantages(records,bootstrap,config['gamma'],config['lambda'])
        grad,loss=net.gradients(records,ret,adv,config['entropy'],config['model_coef'])
        gn=net.update(grad,config['learning_rate'])
        if (update+1)%32==0:
            item={'update':update+1,'loss':loss.tolist(),'gradient_norm':gn,'alive':float(w.alive.mean()),'body':w.body.mean(0).tolist(),'seconds':time.perf_counter()-start}
            logs.append(item);print(json.dumps({'train':mode,'init':initialization,**item}),flush=True)
        if (update+1)%config['validate_every']==0:
            seeds=970000+np.arange(config['validation_worlds'])
            m,_,_,_,_=rollout(net,seeds,config['validation_length'],mode)
            ck=f'model_{mode}_{initialization}_{update+1}.npz';write_tree(output/ck,net.state())
            selection.append({'update':update+1,'checkpoint':ck,'survival':float(m['survival'].mean()),'body_area':float(m['body_area'].mean()),'reward':float(m['reward'].mean())})
            print(json.dumps({'validation':mode,'init':initialization,**selection[-1]}),flush=True)
    best=max(selection,key=lambda z:(z['survival'],z['body_area']));result={'mode':mode,'initialization':initialization,'log':logs,'validation':selection,'selected':best,'seconds':time.perf_counter()-start,'neural_parameters':net.count}
    (output/f'training_{mode}_{initialization}.json').write_text(json.dumps(result,indent=2));return result
