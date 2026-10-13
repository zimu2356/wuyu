import copy, json, time
from pathlib import Path
import numpy as np
from .world import World
from .agent import Agent
from .rnn import Network,RNNAgent,advantages
from .state import write_tree,read_tree,save_raw
from .student import Student


def checkpoint(path,w,agents):
    kind='student' if isinstance(agents,Student) else ('rnn' if isinstance(agents,RNNAgent) else 'hybrid')
    state=agents.state() if kind in ('rnn','student') else [a.state() for a in agents]
    write_tree(path,dict(kind=kind,world=w.state(),agents=state))


def restore(path):
    s=read_tree(path);w=World.from_state(s['world'])
    agents=Student.from_state(s['agents']) if s['kind']=='student' else (RNNAgent.from_state(s['agents']) if s['kind']=='rnn' else [Agent.from_state(a) for a in s['agents']])
    return w,agents


def step(w,agents):
    obs=w.observation()
    if isinstance(agents,RNNAgent):
        r=agents.act(obs);out=w.step(r['action']);agents.feedback(r,w.observation(),out['reward'])
        extras=dict(goal=np.full(w.n,-1),target=np.full(w.n,-1),source=np.zeros(w.n,int),
                    prediction=np.zeros((w.n,3)),skill_origin=np.full(w.n,-1),skills=np.zeros(w.n,int))
        if isinstance(agents,Student):
            extras['source'][:]=3;extras['skills']=np.array([m.valid.sum() for m in agents.memories])
    else:
        records=[a.act(o) for a,o in zip(agents,obs)];action=np.array([r['action'] for r in records]);out=w.step(action)
        nobs=w.observation()
        for i,a in enumerate(agents):a.feedback(records[i],nobs[i],out['reward'][i])
        r=dict(action=action,obs=obs,alive=obs[:,14:16].min(1)>.001)
        extras={k:np.array([x[k] for x in records]) for k in ['goal','target','source','skill_origin']}
        extras['prediction']=np.array([x['pred'] for x in records]);extras['skills']=np.array([a.valid.sum() for a in agents])
    r['out']=out;r['extras']=extras;return r


def metrics(raw):
    alive=raw['alive'];T=len(alive)
    m=dict(survival=alive.mean(0),body_area=(raw['body'].sum(2)*alive).mean(0),deliveries=(raw['product']<4).sum(0),
           skill_fraction=((raw['source']==2)*alive).sum(0)/np.maximum(1,alive.sum(0)),final_skills=raw['skills'][-1])
    for p in range(4):
        mask=raw['phase']==p;m[f'phase_{p}_survival']=(alive*mask).sum(0)/np.maximum(1,mask.sum(0))
    return m


def rollout(seeds,initialization=701,mode='full',total=1536,net=None,output=None,name=None,config=None,save_points=()):
    start=time.perf_counter();w=World(seeds,total)
    cfg=config or {};kw={k:cfg[k] for k in ['model_lr','skill_lr','exploration'] if k in cfg}
    agents=Student(copy.deepcopy(net),seeds,initialization,mode) if mode.startswith('neural_student') else (RNNAgent(copy.deepcopy(net),seeds) if mode=='plain_rnn' else [Agent(s,initialization,mode,capacity=cfg.get('memory_capacity',256),**kw) for s in seeds])
    if mode=='neural_student_frozen':
        for m in agents.memories:m.mode='frozen'
    records=[]
    while w.t<w.total:
        if output and w.t in save_points:checkpoint(Path(output)/f'checkpoint_{name}_{w.t}.npz',w,agents)
        r=step(w,agents);out=r['out']
        records.append(dict(obs=r['obs'].astype(np.float32),action=r['action'].astype(np.int8),alive=r['alive'],
            body=out['body'],position=out['position'].astype(np.int8),stage=out['stage'].astype(np.int8),inventory=out['inventory'].astype(np.int8),
            product=out['product'].astype(np.int8),phase=out['phase'].astype(np.int8),reward=out['reward'],**r['extras']))
    raw={k:np.stack([r[k] for r in records]) for k in records[0]}
    if output:
        save_raw(Path(output)/f'raw_{name}.npz',seeds=np.asarray(seeds),**raw)
        checkpoint(Path(output)/f'checkpoint_{name}_{w.t}.npz',w,agents)
    return metrics(raw),raw,w,agents,time.perf_counter()-start


def train_rnn(config,initialization,output):
    output=Path(output);seed=config['initializations'][initialization];net=Network(seed,config['rnn_hidden']);start=time.perf_counter()
    w=a=None;logs=[];selection=[]
    for update in range(config['train_updates']):
        if w is None or w.t>=w.total:
            birth=update//(config['train_lifetime']//config['train_unroll'])
            seeds=720000+initialization*1000000+birth*config['train_batch']+np.arange(config['train_batch'])
            w=World(seeds,config['train_lifetime']);a=RNNAgent(net,seeds)
        records=[step(w,a) for _ in range(config['train_unroll'])]
        obs=w.observation();x=np.concatenate([obs,a.prev],1);h=np.tanh(x@net.p['W']+a.h@net.p['R']+net.p['b'])
        ret,adv=advantages(records,h@net.p['V']+net.p['vb'][0]);net.update(net.gradients(records,ret,adv))
        if (update+1)%64==0:
            item=dict(update=update+1,alive=float(w.alive.mean()),body=w.body.mean(0).tolist(),seconds=time.perf_counter()-start)
            logs.append(item);print(json.dumps(dict(training=initialization,**item)),flush=True)
        if (update+1)%config['validation_every']==0:
            m,_,_,_,_=rollout(730000+np.arange(config['validation_worlds']),seed,'plain_rnn',net=net,total=1536)
            filename=f'model_rnn_{initialization}_{update+1}.npz';write_tree(output/filename,net.state())
            sel=dict(update=update+1,file=filename,survival=float(m['survival'].mean()),body_area=float(m['body_area'].mean()));selection.append(sel)
            print(json.dumps(dict(validation=initialization,**sel)),flush=True)
    best=max(selection,key=lambda x:(x['survival'],x['body_area']))
    result=dict(initialization=initialization,parameters=net.count,logs=logs,validation=selection,selected=best,seconds=time.perf_counter()-start)
    (output/f'training_rnn_{initialization}.json').write_text(json.dumps(result,indent=2));return result
