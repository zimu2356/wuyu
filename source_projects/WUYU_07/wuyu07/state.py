"""Atomic, pickle-free array + JSON serialization; adapted from WUYU_06."""
import json, os
from pathlib import Path
import numpy as np


def write_tree(path,tree):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);arrays={}
    def pack(v,key):
        if isinstance(v,np.ndarray):arrays[key]=v;return {'array':key}
        if isinstance(v,dict):return {'dict':{str(k):pack(a,key+'/'+str(k)) for k,a in v.items()}}
        if isinstance(v,(list,tuple)):return {'list':[pack(a,key+'/'+str(i)) for i,a in enumerate(v)]}
        if isinstance(v,np.generic):v=v.item()
        if v is None or isinstance(v,(str,int,float,bool)):return {'value':v}
        raise TypeError(type(v))
    arrays['__metadata__']=np.asarray(json.dumps(pack(tree,'root'),sort_keys=True))
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def read_tree(path):
    with np.load(path,allow_pickle=False) as z:
        def unpack(v):
            if 'array' in v:return z[v['array']].copy()
            if 'dict' in v:return {k:unpack(a) for k,a in v['dict'].items()}
            if 'list' in v:return [unpack(a) for a in v['list']]
            return v['value']
        return unpack(json.loads(str(z['__metadata__'])))


def save_raw(path,**arrays):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
