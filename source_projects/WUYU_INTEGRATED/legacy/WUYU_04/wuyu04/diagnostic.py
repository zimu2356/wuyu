"""Controlled old-RNN diagnosis; no environmental controller confound."""
import copy
import time
from collections import deque
import numpy as np
from .legacy_neural import RNN


def input_events(cue, context, dense=False):
    n = len(cue); x = np.zeros((4, n, 17)); rows = np.arange(n)
    x[0, rows, context] = 1
    if dense:
        for t in range(1, 4): x[t, rows, context] = 1
    x[0, :, 2:4] = 1; x[0, :, 4] = 1
    x[1, :, 9] = 2*cue-1; x[1, :, 10] = 1
    for t in range(4): x[t, :, 11+t] = 1
    return x


def stream(seeds, length, offset):
    # Draw a fixed-length paired stream once per experimental condition.
    rs = [np.random.default_rng(int(s)+offset) for s in seeds]
    cue = np.stack([r.integers(0, 2, length) for r in rs], axis=1)
    flip = np.stack([r.random(length) >= .9 for r in rs], axis=1)
    polarity = np.array([np.random.default_rng(int(s)+300000).integers(0, 2) for s in seeds])
    return cue, flip, polarity


class Replay:
    def __init__(self, n, k, seed, capacity=256, burn=8):
        self.x=np.zeros((n, capacity, 4, 17)); self.h=np.zeros((n, capacity, k))
        self.y=np.zeros((n, capacity)); self.past=np.zeros((n, capacity, burn, 4, 17))
        self.length=np.zeros((n, capacity), dtype=np.int16)
        self.count=0; self.capacity=capacity; self.burn=burn
        self.rng=np.random.default_rng(seed); self.history=deque(maxlen=burn)

    def add(self, x, h, y):
        self.count += 1; n=len(y)
        slots=(np.full(n, self.count-1) if self.count<=self.capacity
               else self.rng.integers(0, self.count, n))
        rows=np.flatnonzero(slots<self.capacity); slots=slots[rows]
        self.x[rows, slots]=x[:, rows].transpose(1, 0, 2)
        self.h[rows, slots]=h[rows]; self.y[rows, slots]=y[rows]
        self.past[rows, slots]=0; self.length[rows, slots]=len(self.history)
        if self.history:
            past=np.array(self.history).transpose(2,0,1,3)
            self.past[rows, slots, :len(self.history)]=past[rows]
        self.history.append(x.copy())

    def sample(self):
        rows=np.arange(len(self.y)); slots=self.rng.integers(0,min(self.count,self.capacity),len(rows))
        return (self.x[rows,slots].transpose(1,0,2), self.h[rows,slots].copy(),
                self.y[rows,slots], self.past[rows,slots], self.length[rows,slots])


class Decoder:
    def __init__(self, seeds, mode, rate):
        self.mode=mode; self.rate=rate; self.n=len(seeds)
        self.nets=[RNN(seeds) for _ in range(2 if mode=='gated' else 1)]
        self.replay=Replay(self.n,24,int(seeds[0])+600000) if mode in ('stale','burnin') else None
        self.forward_micro=0; self.backward_micro=0; self.gradient_calls=0

    @property
    def params(self): return sum(n.parameter_count for n in self.nets)

    def predict(self, x, ctx, commit=True):
        p=np.zeros(self.n); hs=[]
        for j,net in enumerate(self.nets):
            h0=net.h.copy(); hs.append(h0)
            pr,h,_=net.forward(x,h0); mask=(ctx==j) if len(self.nets)>1 else np.ones(self.n,bool)
            p[mask]=pr[mask]
            if commit: net.h[mask]=h[-1,mask]
            self.forward_micro+=4
        return p,hs

    def update(self,x,ctx,h,y):
        for j,net in enumerate(self.nets):
            mask=(ctx==j) if len(self.nets)>1 else np.ones(self.n,bool)
            net.learn(x,h[j],y,mask,self.rate)
            self.forward_micro+=4; self.backward_micro+=4; self.gradient_calls+=1
        if self.mode=='repeat':
            self.nets[0].learn(x,h[0],y,np.ones(self.n,bool),self.rate)
            self.forward_micro+=4; self.backward_micro+=4; self.gradient_calls+=1
        if self.replay is not None:
            self.replay.add(x,h[0],y)
            sx,sh,sy,past,length=self.replay.sample()
            if self.mode=='burnin':
                sh=np.zeros_like(sh)
                for k in range(self.replay.burn):
                    nh=self.nets[0].forward(past[:,k].transpose(1,0,2),sh)[1][-1]
                    sh=np.where((k<length)[:,None],nh,sh)
                    self.forward_micro+=4
            self.nets[0].learn(sx,sh,sy,np.ones(self.n,bool),self.rate)
            self.forward_micro+=4; self.backward_micro+=4; self.gradient_calls+=1


def evaluate(agent,seeds,reverse_a=False):
    values=[]
    for context in (0,1):
        a=copy.deepcopy(agent); cue,flip,polarity=stream(seeds,256,1000000+context*1000)
        correct=[]
        for t in range(256):
            ctx=np.full(len(seeds),context)
            x=input_events(cue[t],ctx,a.mode=='dense')
            p,_=a.predict(x,ctx)
            y=cue[t]^polarity^context^int(reverse_a and context==0)^flip[t]
            correct.append((p>=.5)==y)
        values.append(np.mean(correct,axis=0))
    return np.asarray(values)


def counterfactual(agent,seeds):
    polarity=stream(seeds,1,333333)[2]; probs=[]
    for ctx in (0,1):
        row=[]
        for cue in (0,1):
            x=input_events(np.full(len(seeds),cue),np.full(len(seeds),ctx),agent.mode=='dense')
            row.append(agent.predict(x,np.full(len(seeds),ctx),commit=False)[0])
        probs.append(row)
    p=np.asarray(probs)
    target=np.arange(2)[None,:,None]^polarity[None,None,:]^np.arange(2)[:,None,None]
    return p,((p[0]>=.5)!=(p[1]>=.5)).mean(axis=0),((p>=.5)==target).mean(axis=1)


def run(seeds,mode,rate,length=4096,validation=False):
    start=time.perf_counter(); a=Decoder(seeds,mode,rate)
    cue,flip,polarity=stream(seeds,length,800000)
    correct=[]; evals=[]
    for t in range(length):
        if validation:
            phase=t//1024; context=int(phase==1); reverse=phase==2
        else:
            context=t%2 if mode=='joint' else int(t>=length//2); reverse=False
        ctx=np.full(len(seeds),context)
        x=input_events(cue[t],ctx,mode=='dense'); p,h=a.predict(x,ctx)
        target=cue[t]^polarity^context^int(reverse)^flip[t]
        answer=(p>=.5).astype(int); reward=np.where(answer==target,1,-1)
        y=np.where(reward>0,answer,1-answer)  # feedback only, after answer
        a.update(x,ctx,h,y); correct.append(answer==target)
        boundary=(validation and t in (2047,3071)) or (not validation and t in (length//2-1,length-1))
        if boundary: evals.append(evaluate(a,seeds,reverse_a=reverse))
    costs=dict(parameters=a.params,forward_micro_batches=a.forward_micro,backward_micro_batches=a.backward_micro,
               gradient_batches=a.gradient_calls,seconds=time.perf_counter()-start,
               nonzero_gradient_updates_per_individual=np.sum([net.updates for net in a.nets],axis=0).tolist())
    p,flips,mapping=counterfactual(a,seeds)
    return a,dict(correct=np.array(correct),evaluations=np.array(evals),counterfactual_probability=p,
                  context_flips=flips,mapping=mapping),costs


def count_baseline(seeds):
    cue,flip,polarity=stream(seeds,4096,800000)
    counts=np.ones((len(seeds),2,2,2)); rows=np.arange(len(seeds)); evals=[]
    for t in range(4096):
        ctx=int(t>=2048); target=cue[t]^polarity^ctx^flip[t]
        counts[rows,ctx,cue[t],target.astype(int)]+=1
        if t in (2047,4095):
            vals=[]
            for context in (0,1):
                c,f,p=stream(seeds,256,1000000+context*1000)
                pred=counts[rows[None,:],context,c].argmax(axis=-1)
                vals.append((pred==(c^p^context^f)).mean(axis=0))
            evals.append(vals)
    return counts,np.array(evals)
