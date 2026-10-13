"""A persistent hybrid: learned recurrent decoding plus learned tool control.

Hidden targets and encoding rules live only in the world. The interface gives
cues, availability, charge and reward. Feedback in this binary task permits
the agent to reconstruct a training label AFTER it has answered.
"""
from dataclasses import dataclass
import copy
import json
import numpy as np
from .neural import RNN, Memory
from .control import Control, encode


@dataclass
class Exogenous:
    target: np.ndarray
    noise: np.ndarray
    explore: np.ndarray
    random_action: np.ndarray
    tie: np.ndarray


def events(seeds, length, offset=0):
    values=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed)+offset)
        values.append((rng.integers(0, 2, length), rng.random((length,3)),
                       rng.random(length), rng.integers(0,5,length),rng.random((length,5))))
    return Exogenous(*(np.stack([v[k] for v in values],axis=1) for k in range(5)))


class World:
    def __init__(self, seeds):
        self.charge=np.zeros((len(seeds),2),dtype=np.int16)
        self.polarity=np.asarray([np.random.default_rng(int(s)+300000).integers(0,2) for s in seeds])

    def acquire(self, action, context, reversed_rule, target, noise):
        n=len(action); rows=np.arange(n)
        channel=np.clip(action,0,2)
        valid=action==0
        for j in range(2):
            use=action==j+1
            valid |= use & (self.charge[:,j]>0)
            self.charge[use & (self.charge[:,j]>0),j]-=1
            self.charge[action==j+3,j]=4
        correct=noise[rows,channel]<np.array([.6,.9,.9])[channel]
        encoded=target ^ self.polarity ^ np.asarray(context) ^ int(reversed_rule)
        cue=np.where(valid,np.where(correct,encoded,1-encoded),-1).astype(np.int8)
        return cue,valid

    @staticmethod
    def feedback(answer,target,valid):
        return np.where(valid,np.where(answer==target,1.,-1.),-.2)


class Agent:
    def __init__(self,seeds,mode='online',rate=.02):
        self.mode,self.rate=mode,float(rate)
        self.net=RNN(seeds,memory=mode!='no_memory')
        self.ctrl=Control(len(seeds),50,'dyna',extra=4,seed=int(seeds[0])+440000,epsilon=.10)
        self.memory=Memory(len(seeds))
        self.rng=np.random.default_rng(int(seeds[0])+550000)
        self.previous_reward=np.zeros(len(seeds))
        self.previous_answer=np.zeros(len(seeds))
        self.t=0

    def inputs(self, context, before, action, cue, valid):
        n=self.net.n; rows=np.arange(n)
        x=np.zeros((4,n,17))
        x[0,rows,np.asarray(context)]=1
        x[0,:,2:4]=before/4
        x[0,rows,4+action]=1
        x[1,:,9]=np.where(valid,2*cue-1,0)
        x[1,:,10]=valid
        for t in range(4): x[t,:,11+t]=1
        x[0,:,15]=self.previous_reward
        x[0,:,16]=self.previous_answer
        return x

    def answer(self,x):
        h0=self.net.h.copy()
        p=self.net.observe(x)
        return (p>=.5).astype(np.int8),p,h0

    def feedback(self,x,h0,answer,reward,valid,state,action,after,learn=True,frozen=False):
        # Reconstruct a label using only own answer and observable correctness.
        label=np.where(reward>0,answer,1-answer)
        if learn:
            self.ctrl.update(state,action,reward,after)
            self.memory.add(x,h0,label,valid)
            if not frozen:
                self.net.learn(x,h0,label,valid,self.rate)
                if self.mode=='repeat':
                    self.net.learn(x,h0,label,valid,self.rate)
                elif self.mode=='replay':
                    self.net.learn(*self.memory.sample(self.rng),self.rate)
        self.previous_reward=reward.copy()
        self.previous_answer=np.where(valid,2*answer-1,0)
        self.t+=1


def step(world,agent,ev,t,context,reverse=False,next_context=None,learn=True,frozen=False,forced=None):
    n=agent.net.n
    ctx=np.full(n,context,dtype=int) if np.ndim(context)==0 else np.asarray(context)
    nxt=ctx if next_context is None else np.full(n,next_context,dtype=int)
    before=world.charge.copy(); state=encode(before,ctx)
    action=(agent.ctrl.choose(state,ev.explore[t],ev.random_action[t],ev.tie[t])
            if forced is None else np.full(n,forced,dtype=int))
    cue,valid=world.acquire(action,ctx,reverse,ev.target[t],ev.noise[t])
    x=agent.inputs(ctx,before,action,cue,valid)
    answer,p,h0=agent.answer(x)
    reward=world.feedback(answer,ev.target[t],valid)
    agent.feedback(x,h0,answer,reward,valid,state,action,encode(world.charge,nxt),learn,frozen)
    return {'action':action,'cue':cue,'valid':valid,'answer':answer,'probability':p,
            'reward':reward,'charge':before,'correct':(answer==ev.target[t]) & valid}


def evaluate(world,agent,seeds,context,reverse=False,length=256,offset=1000000):
    w,a=copy.deepcopy(world),copy.deepcopy(agent)
    ev=events(seeds,length,offset)
    correct=[]; loss=[]
    for t in range(length):
        out=step(w,a,ev,t,context,reverse,learn=False,forced=0)
        correct.append(out['correct'])
        p=np.clip(out['probability'],1e-9,1-1e-9)
        loss.append(-(ev.target[t]*np.log(p)+(1-ev.target[t])*np.log(1-p)))
    return {'accuracy':np.mean(correct,axis=0),'log_loss':np.mean(loss,axis=0)}


def save_checkpoint(path,world,agent):
    data={}
    meta={'mode':agent.mode,'rate':agent.rate,'t':agent.t,'memory':agent.net.memory,
          'ctrl':{},'agent_rng':agent.rng.bit_generator.state,
          'ctrl_rng':agent.ctrl.rng.bit_generator.state}
    for prefix,obj in [('world',world),('net',agent.net),('ctrl',agent.ctrl),('memory',agent.memory)]:
        for k,v in vars(obj).items():
            if isinstance(v,np.ndarray): data[prefix+'__'+k]=v
            elif prefix=='ctrl' and isinstance(v,(int,float,str)): meta['ctrl'][k]=v
    data['previous_reward']=agent.previous_reward
    data['previous_answer']=agent.previous_answer
    data['meta']=np.asarray(json.dumps(meta))
    np.savez_compressed(path,**data)


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['meta']))
        n=data['net__w'].shape[0]
        a=Agent(np.arange(n),meta['mode'],meta['rate']);w=World(np.arange(n))
        for prefix,obj in [('world',w),('net',a.net),('ctrl',a.ctrl),('memory',a.memory)]:
            for key in data.files:
                if key.startswith(prefix+'__'): setattr(obj,key.split('__',1)[1],data[key].copy())
        for key,value in meta['ctrl'].items():setattr(a.ctrl,key,value)
        a.net.memory=meta['memory'];a.t=meta['t']
        a.rng.bit_generator.state=meta['agent_rng'];a.ctrl.rng.bit_generator.state=meta['ctrl_rng']
        a.previous_reward=data['previous_reward'].copy();a.previous_answer=data['previous_answer'].copy()
    return w,a
