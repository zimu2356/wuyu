"""One learned recurrent decoder, causal tool use, and two plasticity rates.

Agent inputs never include reversal, reliability, phase, or hidden target.
World/external random streams and evaluator labels stay in the runner.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .neural_base import RNN
from .control import Control, encode

KINDS = ('return', 'change', 'noise')
FEATURES = ('bias','error','fast','slow','trend','entropy','replay_error','error_change')
MAX_RATE = .12


class VectorRNG:
    """Per-individual xorshift64*: independent of candidate batching/order."""
    def __init__(self, seeds, offset):
        self.state = np.asarray(seeds,dtype=np.uint64) + np.uint64(offset)
        self.state = self.state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        self.state[self.state==0] = 1
    def random(self, n=None):
        if n is not None and n!=len(self.state): raise ValueError('one draw per individual')
        x=self.state.copy(); x ^= x >> np.uint64(12); x ^= x << np.uint64(25); x ^= x >> np.uint64(27)
        self.state=x
        z=x*np.uint64(2685821657736338717)
        return (z >> np.uint64(11)).astype(np.float64) / 9007199254740992.


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))


class World:
    def __init__(self,seeds,kind='prefix',formal=False):
        self.seeds=np.asarray(seeds,dtype=np.int64); self.n=len(seeds)
        self.kind=kind; self.formal=bool(formal); self.t=0
        self.total=(1536 if formal else 1024) if kind=='prefix' else (1536 if formal else 960)
        self.charge=np.zeros((self.n,2),dtype=np.int16)
        self.polarity=np.asarray([np.random.default_rng(int(s)+300000).integers(0,2) for s in seeds])
        cache={}; arrays=[]; d1=[]; d2=[]
        for s in seeds:
            s=int(s)
            if s not in cache:
                # All three branches have paired ordinary exogenous streams.
                rng=np.random.default_rng(s+(8100000 if kind=='prefix' else 8200000))
                val=(rng.integers(0,2,self.total,dtype=np.int8),rng.random((self.total,3)),
                     rng.random(self.total),rng.integers(0,5,self.total),rng.random((self.total,5)))
                sched=np.random.default_rng(s+8300000)
                if kind=='prefix':
                    a=int(sched.integers(512,769)) if formal else 512; b=0
                else:
                    a=int(sched.integers(512,769)) if formal else int(sched.integers(256,385))
                    b=int(sched.integers(256,385)) if formal else int(sched.integers(128,257))
                cache[s]=(val,a,b)
            val,a,b=cache[s]; arrays.append(val); d1.append(a);d2.append(b)
        self.target,self.noise,self.explore,self.random_action,self.tie=(np.stack([a[k] for a in arrays],axis=1) for k in range(5))
        self.d1=np.asarray(d1);self.d2=np.asarray(d2)

    def conditions(self):
        if self.t>=self.total: raise StopIteration('finite protocol complete')
        if self.kind=='prefix':
            phase=(self.t>=self.d1).astype(int); ctx=phase.copy()
        else:
            phase=np.where(self.t<self.d1,0,np.where(self.t<self.d1+self.d2,1,2))
            ctx=(phase==1).astype(int)
        reverse=(phase==0) & (self.kind=='change')
        degraded=(phase==0) & (self.kind=='noise')
        return ctx,reverse,degraded,phase

    def acquire(self,action):
        ctx,reverse,degraded,phase=self.conditions()
        rows=np.arange(self.n); channel=np.clip(action,0,2)
        valid=action==0
        for j in range(2):
            use=action==j+1
            available=use & (self.charge[:,j]>0)
            valid |= available;self.charge[available,j]-=1
            self.charge[action==j+3,j]=4
        reliability=np.where(degraded,.6,np.array([.6,.9,.9])[channel])
        noisy=self.noise[self.t,rows,channel]>=reliability
        encoded=self.target[self.t]^self.polarity^ctx^reverse
        cue=np.where(valid,encoded^noisy,-1).astype(np.int8)
        return cue,valid

    def finish(self,answer,valid):
        reward=np.where(valid,np.where(answer==self.target[self.t],1.,-1.),-.2)
        self.t+=1
        return reward


class Replay:
    def __init__(self,seeds,capacity=256,burn=8):
        n=len(seeds);self.capacity=capacity;self.burn=burn;self.tick=0
        self.x=np.zeros((n,capacity,4,17));self.y=np.zeros((n,capacity))
        self.past=np.zeros((n,capacity,burn,4,17));self.length=np.zeros((n,capacity),dtype=np.int16)
        self.count=np.zeros(n,dtype=np.int64)
        self.history=np.zeros((n,burn,4,17));self.rng=VectorRNG(seeds,8400000)

    def sample(self,net):
        slots=(self.rng.random()*np.maximum(1,np.minimum(self.count,self.capacity))).astype(int)
        rows=np.arange(len(self.count));x=self.x[rows,slots].transpose(1,0,2)
        past=self.past[rows,slots];length=self.length[rows,slots]
        h=np.zeros_like(net.h)
        for j in range(self.burn):
            nh=net.forward(past[:,j].transpose(1,0,2),h)[1][-1]
            h=np.where((j<length)[:,None],nh,h)
        return x,h,self.y[rows,slots].copy(),self.count>0

    def add(self,x,y,valid):
        self.count+=valid.astype(np.int64)
        candidates=(self.rng.random()*np.maximum(1,self.count)).astype(np.int64)
        slots=np.where(self.count<=self.capacity,self.count-1,candidates)
        keep=valid & (slots<self.capacity);rows=np.flatnonzero(keep);ix=slots[rows]
        self.x[rows,ix]=x[:,rows].transpose(1,0,2);self.y[rows,ix]=y[rows]
        length=min(self.tick,self.burn)
        self.past[rows,ix]=self.history[rows];self.length[rows,ix]=length
        event=x.transpose(1,0,2)
        if self.tick<self.burn: self.history[:,self.tick]=event
        else:
            self.history[:,:-1]=self.history[:,1:];self.history[:,-1]=event
        self.tick+=1


class Agent:
    def __init__(self,seeds,mode,theta,pool=None):
        self.seeds=np.asarray(seeds,dtype=np.int64);self.n=len(seeds);self.mode=mode
        self.theta=np.asarray(theta,float).copy();self.pool=np.zeros((1,2)) if pool is None else np.asarray(pool,float).copy()
        self.net=RNN(seeds);self.ctrl=Control(self.n,50,'dyna',extra=4,seed=1,epsilon=.1)
        self.ctrl.rng=VectorRNG(seeds,8500000)
        self.memory=Replay(seeds);self.policy_rng=VectorRNG(seeds,8600000)
        self.fast=np.full(self.n,.5);self.slow=np.full(self.n,.5);self.previous_error=np.full(self.n,.5)
        self.previous_reward=np.zeros(self.n);self.previous_answer=np.zeros(self.n)
        self.t=0;self.branch=False;self.forward_micro=0;self.backward_micro=0
        self.gradient_calls=0;self.fresh_nonzero=np.zeros(self.n,dtype=np.int64);self.replay_nonzero=np.zeros(self.n,dtype=np.int64)

    def inputs(self,context,before,action,cue,valid):
        rows=np.arange(self.n);x=np.zeros((4,self.n,17))
        x[0,rows,context]=1;x[0,:,2:4]=before/4;x[0,rows,4+action]=1
        x[1,:,9]=np.where(valid,2*cue-1,0);x[1,:,10]=valid
        for t in range(4):x[t,:,11+t]=1
        x[0,:,15]=self.previous_reward;x[0,:,16]=self.previous_answer
        return x

    def features(self,p,y,valid,rp,ry,rv):
        err=np.where(valid,np.abs(p-y),self.previous_error)
        re=np.where(rv,np.abs(rp-ry),.5)
        pc=np.clip(p,1e-12,1-1e-12)
        entropy=-(pc*np.log(pc)+(1-pc)*np.log(1-pc))/np.log(2)
        f=np.stack([np.ones(self.n),2*err-1,2*self.fast-1,2*self.slow-1,
                    4*(self.fast-self.slow),2*entropy-1,2*re-1,2*(err-self.previous_error)],axis=1)
        return np.clip(f,-1,1),err,re

    def rates(self,f,err,re):
        if self.mode=='learned':rates=MAX_RATE*sigmoid(np.einsum('noi,ni->no',self.theta,f))
        elif self.mode in ('fixed','frozen'):rates=MAX_RATE*sigmoid(self.theta)
        elif self.mode=='heuristic':rates=np.minimum(MAX_RATE,MAX_RATE*sigmoid(self.theta)*np.stack([.5+err,.5+re],axis=1))
        elif self.mode=='matched_mean':rates=self.theta.copy()
        elif self.mode=='shuffled':rates=self.pool[(self.policy_rng.random()*len(self.pool)).astype(int)]
        else:raise ValueError(self.mode)
        if self.mode=='frozen' and self.branch:rates=np.zeros_like(rates)
        return rates

    def learn(self,x,h,y,valid,rate):
        g=self.net.gradients(x,h,y,valid)
        norm=np.sqrt(sum((a.reshape(self.n,-1)**2).sum(axis=1) for a in g.values()))
        scale=np.minimum(1.,5/np.maximum(norm,1e-12))*rate
        for name,a in g.items():setattr(self.net,name,getattr(self.net,name)-a*scale.reshape((self.n,)+(1,)*(a.ndim-1)))
        self.net.updates += (valid & (rate>0)).astype(np.int64)
        self.forward_micro+=4;self.backward_micro+=4;self.gradient_calls+=1

    def feedback(self,x,h0,p,answer,reward,valid,state,action,after):
        y=np.where(reward>0,answer,1-answer)
        sx,sh,sy,sv=self.memory.sample(self.net)
        self.forward_micro+=self.memory.burn*4
        rp=self.net.forward(sx,sh)[0];self.forward_micro+=4
        f,err,re=self.features(p,y,valid,rp,sy,sv)
        rates=self.rates(f,err,re)
        self.learn(x,h0,y,valid,rates[:,0]);self.learn(sx,sh,sy,sv,rates[:,1])
        self.fresh_nonzero+=(valid & (rates[:,0]>0));self.replay_nonzero+=(sv & (rates[:,1]>0))
        self.ctrl.update(state,action,reward,after)
        self.memory.add(x,y,valid)
        self.fast=np.where(valid,self.fast+(err-self.fast)/8,self.fast)
        self.slow=np.where(valid,self.slow+(err-self.slow)/64,self.slow)
        self.previous_error=np.where(valid,err,self.previous_error)
        self.previous_reward=reward.copy();self.previous_answer=np.where(valid,2*answer-1,0)
        self.t+=1
        return rates,f,sv


def step(world,agent):
    t=world.t;ctx,_,_,phase=world.conditions();before=world.charge.copy();state=encode(before,ctx)
    action=agent.ctrl.choose(state,world.explore[t],world.random_action[t],world.tie[t])
    cue,valid=world.acquire(action);x=agent.inputs(ctx,before,action,cue,valid)
    h0=agent.net.h.copy();p=agent.net.observe(x);agent.forward_micro+=4
    answer=(p>=.5).astype(np.int8);reward=world.finish(answer,valid)
    rates,features,replay_valid=agent.feedback(x,h0,p,answer,reward,valid,state,action,encode(world.charge,ctx))
    return dict(reward=reward,action=action.astype(np.int8),cue=cue,valid=valid,answer=answer,
                probability=p,rate_current=rates[:,0],rate_replay=rates[:,1],replay_valid=replay_valid,
                context=ctx.astype(np.int8),phase=phase.astype(np.int8),features=features)


def evaluate_b(agent,seeds,length=256):
    # Evaluator works on a separate clone, provides neither labels nor scores to agent.
    a=copy.deepcopy(agent);rngs=[np.random.default_rng(int(s)+8800000) for s in seeds]
    targets=np.stack([r.integers(0,2,length) for r in rngs],axis=1)
    flips=np.stack([r.random(length)>=.9 for r in rngs],axis=1)
    polarity=np.asarray([np.random.default_rng(int(s)+300000).integers(0,2) for s in seeds])
    correct=[]
    for t in range(length):
        cue=targets[t]^polarity^1^flips[t]
        x=a.inputs(np.ones(a.n,int),np.full((a.n,2),4),np.ones(a.n,int),cue,np.ones(a.n,bool))
        ans=(a.net.observe(x)>=.5).astype(int);reward=np.where(ans==targets[t],1.,-1.)
        correct.append(ans==targets[t]);a.previous_reward=reward;a.previous_answer=2*ans-1
    return np.mean(correct,axis=0)


def save(path,world,agent):
    objects={'world':world,'agent':agent,'net':agent.net,'ctrl':agent.ctrl,'mem':agent.memory,
             'mem_rng':agent.memory.rng,'ctrl_rng':agent.ctrl.rng,'policy_rng':agent.policy_rng}
    arrays={};meta={}
    for prefix,obj in objects.items():
        meta[prefix]={}
        for name,v in vars(obj).items():
            if isinstance(v,np.ndarray):arrays[prefix+'__'+name]=v
            elif isinstance(v,(str,int,float,bool)):meta[prefix][name]=v
    arrays['meta']=np.asarray(json.dumps(meta,sort_keys=True))
    np.savez_compressed(path,**arrays)


def load(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['meta']));wm=meta['world'];am=meta['agent']
        world=World(data['world__seeds'],wm['kind'],wm['formal'])
        agent=Agent(data['agent__seeds'],am['mode'],data['agent__theta'],data['agent__pool'])
        objects={'world':world,'agent':agent,'net':agent.net,'ctrl':agent.ctrl,'mem':agent.memory,
                 'mem_rng':agent.memory.rng,'ctrl_rng':agent.ctrl.rng,'policy_rng':agent.policy_rng}
        for prefix,obj in objects.items():
            for key,v in meta[prefix].items():setattr(obj,key,v)
            for key in data.files:
                if key.startswith(prefix+'__'):setattr(obj,key.split('__',1)[1],data[key].copy())
    return world,agent


def array_bytes(agent):
    return sum(v.nbytes for obj in (agent,agent.net,agent.ctrl,agent.memory,agent.memory.rng,
                                    agent.ctrl.rng,agent.policy_rng) for v in vars(obj).values() if isinstance(v,np.ndarray))
