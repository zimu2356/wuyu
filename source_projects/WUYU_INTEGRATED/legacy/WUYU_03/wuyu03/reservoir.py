"""Supplement-only memory policy; no changes to the main experiment classes."""
import json
import numpy as np
from .neural import Memory
from .core import save_checkpoint,load_checkpoint


class ReservoirMemory(Memory):
    def __init__(self,n,capacity=256,inputs=17,hidden=24,seed=1):
        super().__init__(n,capacity,inputs,hidden)
        self.reservoir_rng=np.random.default_rng(seed)

    def add(self,x,h,y,valid):
        rows=np.flatnonzero(valid)
        seen=self.count[rows]
        draws=(self.reservoir_rng.random(len(rows))*(seen+1)).astype(np.int64)
        slots=np.where(seen<self.capacity,seen,draws)
        keep=slots<self.capacity
        rr=rows[keep];ss=slots[keep]
        self.x[rr,ss]=x[:,rr].transpose(1,0,2)
        self.h[rr,ss]=h[rr];self.y[rr,ss]=y[rr]
        self.count[rows]+=1


def save(path,world,agent):
    save_checkpoint(path,world,agent)
    with np.load(path,allow_pickle=False) as d:data={k:d[k].copy() for k in d.files}
    meta=json.loads(str(data['meta']))
    meta['reservoir_rng']=agent.memory.reservoir_rng.bit_generator.state
    data['meta']=np.asarray(json.dumps(meta))
    np.savez_compressed(path,**data)


def load(path):
    w,a=load_checkpoint(path)
    m=ReservoirMemory(a.net.n)
    for k,v in vars(a.memory).items():setattr(m,k,v)
    with np.load(path,allow_pickle=False) as d:meta=json.loads(str(d['meta']))
    m.reservoir_rng.bit_generator.state=meta['reservoir_rng'];a.memory=m
    return w,a
