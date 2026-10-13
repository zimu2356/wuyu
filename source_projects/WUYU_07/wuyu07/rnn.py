"""Direct recurrent actor-critic; explicit BPTT gradients, no privileged input."""
import copy
import numpy as np
from .agent import softmax

D=30;A=7


class Network:
    def __init__(self,seed,hidden=24,input_dim=D):
        self.hidden=hidden;self.input_dim=input_dim;r=np.random.default_rng(seed);h=hidden
        self.p={'W':r.normal(0,1/np.sqrt(input_dim),(input_dim,h)),'R':.8*np.linalg.qr(r.normal(size=(h,h)))[0],
            'b':np.zeros(h),'A':r.normal(0,.02,(h,A)),'ab':np.zeros(A),'V':np.zeros(h),'vb':np.zeros(1)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v=copy.deepcopy(self.m);self.updates=0
    @property
    def count(self):return sum(a.size for a in self.p.values())
    def forward(self,x,h0,u):
        p=self.p;h=np.tanh(x@p['W']+h0@p['R']+p['b']);ap=softmax(h@p['A']+p['ab'])
        action=np.minimum((u[:,None]>ap.cumsum(1)).sum(1),6)
        return dict(x=x.copy(),h0=h0.copy(),h=h,ap=ap,action=action,value=h@p['V']+p['vb'][0])
    def gradients(self,records,returns,advantages,entropy=.018):
        p=self.p;gr={k:np.zeros_like(v) for k,v in p.items()};dh_next=np.zeros_like(records[0]['h'])
        denom=max(1,sum(r['alive'].sum() for r in records))
        for ti in reversed(range(len(records))):
            r=records[ti];live=r['alive'].astype(float);norm=live/denom;ap=r['ap']
            dl=ap.copy();dl[np.arange(len(live)),r['action']]-=1;dl*=(advantages[ti]*norm)[:,None]
            logap=np.log(np.maximum(ap,1e-12));en=(ap*logap).sum(1,keepdims=True)
            dl+=entropy*ap*(logap-en)*norm[:,None]
            gr['A']+=r['h'].T@dl;gr['ab']+=dl.sum(0);dh=dl@p['A'].T+dh_next
            dv=.5*(r['value']-returns[ti])*norm
            gr['V']+=r['h'].T@dv;gr['vb'][0]+=dv.sum();dh+=dv[:,None]*p['V']
            dz=dh*(1-r['h']**2)*live[:,None]
            gr['W']+=r['x'].T@dz;gr['R']+=r['h0'].T@dz;gr['b']+=dz.sum(0);dh_next=dz@p['R'].T
        return gr
    def update(self,grad,lr=.001):
        self.updates+=1;norm=np.sqrt(sum((v*v).sum() for v in grad.values()));scale=min(1,1/max(norm,1e-12))
        for k,g in grad.items():
            g=g*scale;self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            self.p[k]-=lr*(self.m[k]/(1-.9**self.updates))/(np.sqrt(self.v[k]/(1-.999**self.updates))+1e-8)
    def imitation_gradients(self,records,labels):
        gr={k:np.zeros_like(v) for k,v in self.p.items()};dh_next=np.zeros_like(records[0]['h']);p=self.p
        denom=max(1,sum(r['alive'].sum() for r in records))
        for t in reversed(range(len(records))):
            r=records[t];live=r['alive'].astype(float);dl=r['ap'].copy();dl[np.arange(len(live)),labels[t]]-=1;dl*=live[:,None]/denom
            gr['A']+=r['h'].T@dl;gr['ab']+=dl.sum(0)
            dz=(dl@p['A'].T+dh_next)*(1-r['h']**2)*live[:,None]
            gr['W']+=r['x'].T@dz;gr['R']+=r['h0'].T@dz;gr['b']+=dz.sum(0);dh_next=dz@p['R'].T
        return gr
    def state(self):return copy.deepcopy(self.__dict__)
    @classmethod
    def from_state(cls,state):
        n=cls.__new__(cls);n.__dict__.update(copy.deepcopy(state));return n


class RNNAgent:
    def __init__(self,net,seeds):
        self.net=net;self.seeds=np.asarray(seeds);self.h=np.zeros((len(seeds),net.hidden));self.prev=np.zeros((len(seeds),9))
        with np.errstate(over='ignore'):
            self.rng=(self.seeds.astype(np.uint64)+707)*np.uint64(6364136223846793005)+np.uint64(1442695040888963407)
        self.rng[self.rng==0]=1;self.tick=0
    def uniform(self):
        x=self.rng.copy();x^=x>>np.uint64(12);x^=x<<np.uint64(25);x^=x>>np.uint64(27);self.rng=x
        with np.errstate(over='ignore'):return ((x*np.uint64(2685821657736338717))>>np.uint64(11)).astype(float)/9007199254740992.
    def act(self,obs):
        r=self.net.forward(np.concatenate([obs,self.prev],1),self.h,self.uniform());self.h=r['h'].copy()
        r['obs']=obs.copy();r['alive']=obs[:,14:16].min(1)>.001;return r
    def feedback(self,r,next_obs,reward):
        r['reward']=reward.copy();r['after_alive']=next_obs[:,14:16].min(1)>.001
        self.prev=np.concatenate([np.eye(7)[r['action']],(next_obs[:,14:16]-r['obs'][:,14:16])/.28],1);self.tick+=1
    def state(self):return dict(network=self.net.state(),seeds=self.seeds,h=self.h,prev=self.prev,rng=self.rng,tick=self.tick)
    @classmethod
    def from_state(cls,s):
        a=cls(Network.from_state(s['network']),s['seeds'])
        for k in ['h','prev','rng','tick']:setattr(a,k,s[k])
        return a


def advantages(records,bootstrap,gamma=.99,lam=.95):
    gae=np.zeros_like(bootstrap);ret=[];adv=[];nv=bootstrap
    for r in reversed(records):
        cont=r['after_alive'];delta=r['reward']+gamma*nv*cont-r['value']
        gae=(delta+gamma*lam*cont*gae)*r['alive'];adv.append(gae.copy());ret.append(gae+r['value']);nv=r['value']
    ret=np.stack(ret[::-1]);adv=np.stack(adv[::-1]);mask=np.stack([r['alive'] for r in records]);vals=adv[mask]
    if len(vals):adv=np.clip((adv-vals.mean())/(vals.std()+1e-6),-5,5)
    return ret,adv
