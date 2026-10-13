"""Array + JSON checkpoints; numpy allow_pickle=False throughout."""
import json
from pathlib import Path
import numpy as np
from .network import Network
from .world import World
from .agent import Agent,Stream


def write_tree(path,tree):
    arrays={}
    def pack(v,key):
        if isinstance(v,np.ndarray):arrays[key]=v;return {'array':key}
        if isinstance(v,dict):return {'dict':{k:pack(a,key+'/'+k) for k,a in v.items()}}
        if isinstance(v,(list,tuple)):return {'list':[pack(a,key+'/'+str(i)) for i,a in enumerate(v)]}
        if isinstance(v,np.generic):v=v.item()
        if v is None or isinstance(v,(str,int,float,bool)):return {'value':v}
        raise TypeError(type(v))
    meta=pack(tree,'root');arrays['__metadata__']=np.asarray(json.dumps(meta,sort_keys=True))
    np.savez_compressed(path,**arrays)


def read_tree(path):
    with np.load(path,allow_pickle=False) as z:
        def unpack(v):
            if 'array' in v:return z[v['array']].copy()
            if 'dict' in v:return {k:unpack(a) for k,a in v['dict'].items()}
            if 'list' in v:return [unpack(a) for a in v['list']]
            return v['value']
        return unpack(json.loads(str(z['__metadata__'])))


def save(path,world,agent):
    a={k:v for k,v in vars(agent).items() if k not in ('net','policy_rng','memory_rng')}
    write_tree(path,{'network':agent.net.state(),'world':vars(world),'agent':a,
                     'policy_rng':agent.policy_rng.state,'memory_rng':agent.memory_rng.state})


def load(path):
    s=read_tree(path);w=World.__new__(World);w.__dict__.update(s['world']);a=Agent.__new__(Agent);a.__dict__.update(s['agent'])
    a.net=Network.from_state(s['network']);a.policy_rng=Stream(a.seeds,0);a.memory_rng=Stream(a.seeds,0)
    a.policy_rng.state=s['policy_rng'];a.memory_rng.state=s['memory_rng'];return w,a
