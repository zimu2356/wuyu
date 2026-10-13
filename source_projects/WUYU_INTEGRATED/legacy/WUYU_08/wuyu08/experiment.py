import copy,time,json,hashlib
from pathlib import Path
import numpy as np
from .world import World
from .memory import Agent
from .individual import Individual
from .network import Network
from .state import write_tree,read_tree,save_raw

def checkpoint(path,w,a):write_tree(path,dict(world=w.state(),individual=a.state()))
def restore(path):
    s=read_tree(path);return World.from_state(s['world']),Individual.from_state(s['individual'])

def neural_step(w,a,greedy=False,override=None):
    r=a.act(w.observation(),greedy)
    if override is not None:r['action']=np.full(w.n,override,int)
    out=w.step(r['action']);a.feedback(r,w.observation());r['out']=out;return r

def common_history(seeds,init,total=384):
    w=World(seeds,1536);agents=[Agent(s,init) for s in seeds];obs=[];actions=[]
    for t in range(total):
        o=w.observation();obs.append(o);records=[a.act(x,explore=True) for a,x in zip(agents,o)]
        action=np.array([r['action'] for r in records]);actions.append(action);w.step(action);no=w.observation()
        for a,r,x in zip(agents,records,no):a.feedback(r,x)
    return dict(obs=np.array(obs),action=np.array(actions),last_obs=w.observation(),world=copy.deepcopy(w.state()))

def replay_history(net,history,init):
    a=Individual(net,history['world']['seeds'],init)
    for t,o in enumerate(history['obs']):
        r=a.act(o,greedy=True);r['action']=history['action'][t].copy()
        nxt=history['obs'][t+1] if t+1<len(history['obs']) else history['last_obs']
        a.feedback(r,nxt)
    return World.from_state(copy.deepcopy(history['world'])),a

def condition_state(w,a,condition):
    w=copy.deepcopy(w);a=copy.deepcopy(a);w.shocks=False;w.training_varied=False
    w.regime_override=0 if condition=='independent' else 1
    if condition in ('long_chain','changed'):
        req=w.requirements()
        if condition=='long_chain':
            req[w.products<2]=1;req[w.products==3]=0;req[w.products==2]=-1
        else:req[w.products<2]=1-req[w.products<2]
        w.req_override=req
    # Controlled setup, before the physical survey and the paired body intervention.
    # Tokens expire and devices become unprepared; no facts are injected into memory.
    w.inventory[:]=0;w.stage[:]=0;w.last[:]=4
    for m in a.memories:m.learn_enabled=False
    for _ in range(6):neural_step(w,a,greedy=True,override=1)
    return w,a

def pair_state(w,a):
    ws=copy.deepcopy(w.state())
    for k,v in ws.items():
        if isinstance(v,np.ndarray):ws[k]=np.repeat(v,2,axis=0)
    ws['n']*=2;w=World.from_state(ws)
    s=a.state()
    for k in ['seeds','h','prev']:s[k]=np.repeat(s[k],2,axis=0)
    s['memories']=[copy.deepcopy(m) for m in s['memories'] for _ in range(2)];a=Individual.from_state(s)
    need=np.tile([0,1],len(w.seeds)//2);live=w.alive.copy();w.body[live]=.9
    rows=np.flatnonzero(live);w.body[rows,need[rows]]=.25
    return w,a,need

def probe(w,a,condition,intervention='none',steps=40):
    w,a=condition_state(w,a,condition);w,a,need=pair_state(w,a)
    if intervention=='body_clamp':a.body_clamp=True
    if intervention=='hidden_reset':a.h[:]=0
    if intervention=='one_round':a.rounds=1
    if intervention=='zero_round':a.rounds=0
    if intervention=='cut_edges':a.cut_edges=True
    start=copy.deepcopy(w.state());initial=copy.deepcopy(a.state());records=[]
    if intervention=='planner':
        for m in a.memories:m.exploration=0
    for _ in range(steps):
        if intervention=='planner':
            obs=w.observation();rr=[m.act(o,explore=False) for m,o in zip(a.memories,obs)]
            act=np.array([r['action'] for r in rr]);out=w.step(act);nobs=w.observation()
            for m,r,o in zip(a.memories,rr,nobs):m.feedback(r,o)
            record=dict(action=act,obs=obs,goal=np.array([r['goal'] for r in rr]),target=np.array([r['target'] for r in rr]))
        else:
            r=neural_step(w,a,True);out=r['out'];record=dict(action=r['action'],obs=r['obs'],goal=r['gp'].argmax(1),target=r['tp'].argmax(1))
        record.update(body=out['body'],product=out['product'],alive=out['was'],position=out['position'],stage=out['stage'],inventory=out['inventory'],shock=out['shock'])
        records.append(record)
    raw={k:np.stack([r[k] for r in records]) for k in records[0]}
    met=probe_metrics(raw,need,start['body']);met['goal_need']=(w.products[np.arange(w.n),raw['goal'][0]]==need)&start['alive']
    return dict(start=start,initial=initial,final=a.state(),need=need,raw=raw,metrics=met,condition=condition,intervention=intervention)

def probe_metrics(raw,need,start_body):
    p=raw['product'];has=p<2;first_index=has.argmax(0);any_body=has.any(0)
    first=np.where(any_body,p[first_index,np.arange(len(need))],-1)
    wanted=p==need[None,:];any_need=wanted.any(0);idx=wanted.argmax(0)
    return dict(first_success=(first==need),any_success=any_need,no_body=~any_body,first_product=first,
                first_cost=np.where(first==need,first_index+1,len(p)),any_cost=np.where(any_need,idx+1,len(p)),
                body_gain=raw['body'][-1,np.arange(len(need)),need]-start_body[np.arange(len(need)),need],
                final_min=raw['body'][-1].min(1),survival=raw['alive'].mean(0))

def validate(net,history,init):
    w,a=replay_history(net,history,init);scores=[]
    for c in ['independent','dependency']:
        q=probe(w,a,c);scores.append(float(q['metrics']['first_success'].mean()))
    # Tie-break uses goal/target/action agreement on the same visible histories.
    obs=w.observation();r=a.act(obs,True);ys=[m.act(o,explore=False) for m,o in zip(a.memories,obs)]
    agree=float(np.mean(r['ap'].argmax(1)==np.array([z['action'] for z in ys])))
    return dict(score=float(np.mean(scores)),independent=scores[0],dependency=scores[1],agreement=agree)

def train(config,cell,i,out,pilot=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);kind,coverage=cell.split('_');init=config['initializations'][i]
    net=Network(init,kind,config['rounds']);b=config['batch'];u=config['unroll'];updates=config['updates'];start=time.perf_counter()
    val=common_history(config['validation_start']+np.arange(config['validation_worlds']),init)
    w=a=None;birth=0;logs=[];selected=[];active=0;mixrng=np.random.default_rng(init+8808)
    for update in range(updates):
        if w is None or w.t+u>w.total:
            seeds=config['train_start']+i*1000000+birth*b+np.arange(b);birth+=1
            w=World(seeds,1536,training_varied=coverage=='varied');a=Individual(net,seeds,init)
        records=[];labels=[]
        for t in range(u):
            obs=w.observation();r=a.act(obs);teach=[m.act(o,explore=False) for m,o in zip(a.memories,obs)]
            y=np.array([[z['action'],z['goal'],z['target']] for z in teach]);y=np.maximum(y,0)
            fraction=1. if update<updates//2 else max(0.,1-(update-updates//2)/(updates/4))
            chosen=mixrng.random(b)<fraction
            if chosen.any():
                actual=[m.act(o,explore=True)['action'] if use else r['action'][j] for j,(m,o,use) in enumerate(zip(a.memories,obs,chosen))]
                r['action']=np.asarray(actual)
            outc=w.step(r['action']);a.feedback(r,w.observation());records.append(r);labels.append(y);active+=int(r['alive'].sum())
        loss=net.loss(records,labels);net.update(net.gradients(records,labels),config['lr'])
        if (update+1)%config['log_every']==0:
            z=dict(update=update+1,loss=loss,active=active,alive=float(w.alive.mean()),seconds=time.perf_counter()-start)
            logs.append(z);print(json.dumps(dict(cell=cell,initialization=i,**z)),flush=True)
        if (update+1) in config['checkpoints']:
            v=validate(net,val,init);name=f'model_{cell}_{i}_{update+1}.npz';write_tree(out/name,net.state())
            sel=dict(update=update+1,file=name,**v);selected.append(sel);print(json.dumps(dict(validation=cell,initialization=i,**sel)),flush=True)
    best=max(selected,key=lambda v:(v['score'],v['agreement'])) if selected else None
    result=dict(cell=cell,initialization=i,parameters=net.count,recurrent_state_scalars=int(net.zero(1).size),
                logs=logs,candidates=selected,selected=best,active_examples=active,planned_examples=updates*u*b,seconds=time.perf_counter()-start,pilot=pilot)
    (out/f'training_{cell}_{i}.json').write_text(json.dumps(result,indent=2));return result

def life(net,seeds,init,out=None,name='',save_points=(384,768,1536)):
    w=World(seeds,1536,shocks=True);a=Individual(net,seeds,init);records=[]
    if out:checkpoint(Path(out)/f'checkpoint_{name}_0.npz',w,a)
    while w.t<w.total:
        r=neural_step(w,a);z=r['out'];records.append(dict(obs=r['obs'],action=r['action'],alive=r['alive'],body=z['body'],product=z['product'],position=z['position'],stage=z['stage'],inventory=z['inventory'],shock=z['shock'],goal=r['gp'].argmax(1),target=r['tp'].argmax(1)))
        if out and w.t in save_points:checkpoint(Path(out)/f'checkpoint_{name}_{w.t}.npz',w,a)
    raw={k:np.stack([r[k] for r in records]) for k in records[0]}
    metrics=dict(survival=raw['alive'].mean(0),body_area=(raw['body'].sum(2)*raw['alive']).mean(0),deliveries=(raw['product']<4).sum(0))
    if out:
        save_raw(Path(out)/f'life_{name}.npz',seeds=np.array(seeds),**raw)
        (Path(out)/f'life_metrics_{name}.json').write_text(json.dumps({k:v.tolist() for k,v in metrics.items()},indent=2))
    return metrics,raw,w,a
