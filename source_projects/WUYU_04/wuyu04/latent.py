"""Finite learned rule bank and observable-evidence routing.

World owns all hidden rule labels. Hidden agents receive cue, optional requested
calibration label, and post-decision correctness. They never receive a rule id.
Two learned apprenticeship modules are deliberately frozen; one slot is plastic.
"""
import json
from pathlib import Path
import numpy as np

MODES=('single','oracle','passive','active')
SCHEDULE=(0,1,0,2,0,1,2,0,0)


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))


class World:
    def __init__(self,seeds):
        self.seeds=np.asarray(seeds,dtype=np.int64); self.n=len(seeds); self.t=0
        rules=[]
        for s in seeds:
            rng=np.random.default_rng(int(s)+1200000); bank=[]
            while len(bank)<3:
                row=np.zeros(8,dtype=np.int8); row[rng.choice(8,4,replace=False)]=1
                if all(np.count_nonzero(row!=old)>=4 for old in bank):bank.append(row)
            rules.append(bank)
        self.rules=np.array(rules)
        self.event_rng=np.random.default_rng(int(seeds[0])+1300000)
        self.probe_rng=np.random.default_rng(int(seeds[0])+1400000)
        self.cue=np.zeros(self.n,dtype=np.int8); self.target=np.zeros(self.n,dtype=np.int8)
        self.probe_flip=np.zeros(self.n,bool); self.probed=np.zeros(self.n,bool)
        self.pending=False

    def condition(self):
        if self.t<2048:return self.t//1024,.9
        phase=(self.t-2048)//512
        if phase>=len(SCHEDULE): raise StopIteration('finite protocol completed')
        return SCHEDULE[phase],.6 if phase==7 else .9

    def begin(self):
        if self.pending:raise RuntimeError('feedback must finish previous event')
        context,reliability=self.condition()
        self.cue=self.event_rng.integers(0,8,self.n,dtype=np.int8)
        flip=self.event_rng.random(self.n)>=reliability
        self.target=self.rules[np.arange(self.n),context,self.cue]^flip
        # This independent noise is consumed regardless of query decisions.
        self.probe_flip=self.probe_rng.random(self.n)>=reliability
        self.probed[:]=False; self.pending=True
        return self.cue.copy()

    def probe(self,requested_cue,mask):
        if not self.pending:raise RuntimeError('no current event')
        mask=np.asarray(mask,bool)
        if np.any(self.probed & mask):raise RuntimeError('one probe per ordinary event')
        context,_=self.condition(); label=np.full(self.n,-1,dtype=np.int8)
        rows=np.flatnonzero(mask)
        label[rows]=self.rules[rows,context,requested_cue[rows]]^self.probe_flip[rows]
        self.probed |= mask
        return label

    def finish(self,answer):
        if not self.pending:raise RuntimeError('no current event')
        reward=np.where(answer==self.target,1.,-1.)
        self.pending=False; self.t+=1
        return reward


class Agent:
    def __init__(self,n,mode='active',rate=.2,hazard=.02,threshold=.35,cost=.10):
        if mode not in MODES:raise ValueError(mode)
        self.n=n; self.mode=mode; self.rate=rate; self.hazard=hazard
        self.threshold=threshold; self.cost=cost; self.t=0
        self.w=np.zeros((n,1 if mode=='single' else 3,8))
        self.belief=np.full((n,3),1/3)
        self.apprentice_bank=np.zeros((n,2,8))
        self.probes=np.zeros(n,dtype=np.int64)
        self.updates=np.zeros(n,dtype=np.int64)
        self.prediction_values=np.zeros(n,dtype=np.int64)

    def probabilities(self):
        self.prediction_values += self.w.shape[1]*8
        return sigmoid(self.w)

    def start(self,oracle_context=None):
        if self.mode in ('active','passive'):
            self.belief=(1-self.hazard)*self.belief+self.hazard/3
        elif self.mode=='oracle':
            if oracle_context is None:raise ValueError('oracle needs explicitly privileged label')
            self.belief[:]=0; self.belief[:,oracle_context]=1

    def query(self):
        if self.mode!='active':return np.zeros(self.n,int),np.zeros(self.n,bool)
        p=self.probabilities()
        mean=np.sum(self.belief[:,:,None]*p,axis=1)
        variance=np.sum(self.belief[:,:,None]*(p-mean[:,None,:])**2,axis=1)
        chosen=variance.argmax(axis=1)
        entropy=-np.sum(self.belief*np.log(np.maximum(self.belief,1e-30)),axis=1)/np.log(3)
        return chosen,entropy>self.threshold

    def predict(self,cue):
        ps=self.probabilities()[np.arange(self.n),:,cue]
        p=ps[:,0] if self.mode=='single' else (self.belief*ps).sum(axis=1)
        return (p>=.5).astype(np.int8),p

    def observe(self,cue,label,mask=None,apprentice_context=None):
        mask=np.ones(self.n,bool) if mask is None else np.asarray(mask,bool)
        rows=np.flatnonzero(mask)
        if not len(rows):return
        ps=self.probabilities()[np.arange(self.n),:,cue]
        if apprentice_context is not None:
            slot=0 if self.mode=='single' else apprentice_context
            self.w[rows,slot,cue[rows]]-=self.rate*(ps[rows,slot]-label[rows])
        elif self.mode=='single':
            self.w[rows,0,cue[rows]]-=self.rate*(ps[rows,0]-label[rows])
        else:
            if self.mode in ('active','passive'):
                likelihood=np.where(label[rows,None]==1,ps[rows],1-ps[rows])
                posterior=self.belief[rows]*np.clip(likelihood,1e-6,1)
                self.belief[rows]=posterior/posterior.sum(axis=1,keepdims=True)
            # Slots 0/1 were learned, then explicitly frozen. Only slot 2 adapts.
            self.w[rows,2,cue[rows]]-=self.rate*self.belief[rows,2]*(ps[rows,2]-label[rows])
        self.updates[rows]+=1

    def freeze_apprenticeship(self):
        if self.mode!='single':self.apprentice_bank=self.w[:,:2].copy()
        self.belief[:]=1/3


def step(world,agent):
    t=world.t; cue=world.begin(); apprenticeship=t<2048
    if apprenticeship:
        ctx=t//1024
        if agent.mode!='single':agent.belief[:]=0;agent.belief[:,ctx]=1
        requested=np.zeros(world.n,int); query=np.zeros(world.n,bool)
    else:
        # Only the explicitly named oracle gets the hidden condition.
        agent.start(world.condition()[0] if agent.mode=='oracle' else None)
        requested,query=agent.query()
        if np.any(query):
            labels=world.probe(requested,query)
            agent.observe(requested,labels,query)
            agent.probes+=query
    before_answer=agent.belief.copy()
    answer,p=agent.predict(cue); reward=world.finish(answer)
    labels=np.where(reward>0,answer,1-answer)
    agent.observe(cue,labels,apprentice_context=t//1024 if apprenticeship else None)
    agent.t+=1
    if world.t==2048:agent.freeze_apprenticeship()
    return dict(cue=cue,answer=answer,probability=p,correct=reward>0,
                reward=reward,query=query,probe_cue=requested,belief_before=before_answer,
                belief_after=agent.belief.copy(),net_reward=reward-agent.cost*query)


def save(path,world,agent):
    arrays={}; meta={'world':{},'agent':{},'event_rng':world.event_rng.bit_generator.state,
                   'probe_rng':world.probe_rng.bit_generator.state}
    for prefix,obj in [('world',world),('agent',agent)]:
        for name,value in vars(obj).items():
            if isinstance(value,np.ndarray):arrays[prefix+'__'+name]=value
            elif isinstance(value,(int,float,bool,str)):meta[prefix][name]=value
    arrays['meta']=np.asarray(json.dumps(meta))
    np.savez_compressed(path,**arrays)


def load(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['meta'])); world=World(data['world__seeds'])
        agent=Agent(world.n,meta['agent']['mode'])
        for prefix,obj in [('world',world),('agent',agent)]:
            for name,value in meta[prefix].items():setattr(obj,name,value)
            for key in data.files:
                if key.startswith(prefix+'__'):setattr(obj,key.split('__',1)[1],data[key].copy())
        world.event_rng.bit_generator.state=meta['event_rng']
        world.probe_rng.bit_generator.state=meta['probe_rng']
    return world,agent


def mapping_accuracy(world,agent):
    if agent.mode=='single':return ((sigmoid(agent.w[:,0])>=.5)[:,None,:]==world.rules).mean(axis=2).T
    return ((sigmoid(agent.w)>=.5)==world.rules).mean(axis=2).T


def array_bytes(obj):
    return sum(v.nbytes for v in vars(obj).values() if isinstance(v,np.ndarray))
