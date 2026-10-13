"""Shared information interface. All facts originate in observable experience.

Dependency links and ring distances are declared deterministic input priors, not
learned reasoning. Both flat and graph cores receive the exact same arrays.
"""
import numpy as np
from .memory import softmax
F,G,N=31,18,6

def encode(obs,prev,memories,body_clamp=False):
    b=len(obs);node=np.zeros((b,N,F));edge=np.zeros((b,N,N));local=obs[:,:6]
    for j,(o,m) in enumerate(zip(obs,memories)):
        pos,stage,need,inventory,body=m.decode(o)
        known=m.effect_count>0;products=softmax(m.effect)*known[:,None]
        node[j,:,:4]=products;node[j,:,4]=known
        node[j,:,5:9]=np.eye(4)[m.requirement+2]
        node[j,:,9]=m.valid
        node[j,pos,10:13]=np.eye(3)[stage]
        node[j,pos,13:22]=softmax(m.transition[pos,min(stage,1)]).ravel()
        node[j,pos,22:28]=softmax(m.motor[pos]).ravel()*m.valid[pos]
        node[j,:,28]=(np.arange(6)-pos)%6/5
        node[j,:,29]=(pos-np.arange(6))%6/5
        node[j,pos,30]=1
        kinds=np.where(known,m.effect.argmax(1),-1)
        for site,req in enumerate(m.requirement):
            if req>=0 and inventory[req]==0:
                providers=(kinds==req+2).astype(float)
                if providers.sum():edge[j,site]=providers/providers.sum()
    body=obs[:,14:16].copy()
    if body_clamp:body[:]=[.8,.85]
    glob=np.concatenate([body,obs[:,12:14],prev,obs[:,16:21]],axis=1)
    assert node.shape==(b,N,F) and glob.shape==(b,G)
    return dict(node=node,glob=glob,edge=edge,local=local.copy())
