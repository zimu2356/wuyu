import copy
import numpy as np
from .memory import Agent
from .network import Network
from .features import encode

class Individual:
    def __init__(self,net,seeds,initialization=801):
        self.net=net;self.seeds=np.array(seeds);self.initialization=initialization
        self.memories=[Agent(int(s),initialization) for s in seeds]
        self.h=net.zero(len(seeds));self.prev=np.zeros((len(seeds),9));self.tick=0
        self.rng=np.random.default_rng(initialization+int(self.seeds[0])*41)
        self.rounds=None;self.cut_edges=False;self.body_clamp=False
    def act(self,obs,greedy=False):
        x=encode(obs,self.prev,self.memories,self.body_clamp)
        r=self.net.forward(x,self.h,self.rounds,self.cut_edges);self.h=r['h'].copy()
        if greedy:action=r['ap'].argmax(1)
        else:action=np.minimum((self.rng.random(len(obs))[:,None]>r['ap'].cumsum(1)).sum(1),6)
        r.update(action=action,obs=obs.copy(),alive=obs[:,14:16].min(1)>.001)
        return r
    def feedback(self,r,nobs):
        for i,m in enumerate(self.memories):m.feedback(dict(action=int(r['action'][i]),obs=r['obs'][i]),nobs[i])
        self.prev=np.concatenate([np.eye(7)[r['action']],(nobs[:,14:16]-r['obs'][:,14:16])/.28],1);self.tick+=1
    def state(self):
        return dict(network=self.net.state(),seeds=self.seeds,initialization=self.initialization,h=self.h,prev=self.prev,tick=self.tick,
                    rng=self.rng.bit_generator.state,memories=[m.state() for m in self.memories],rounds=self.rounds,cut_edges=self.cut_edges,body_clamp=self.body_clamp)
    @classmethod
    def from_state(cls,s):
        x=cls(Network.from_state(s['network']),s['seeds'],s['initialization'])
        for k in ['h','prev','tick','rounds','cut_edges','body_clamp']:setattr(x,k,copy.deepcopy(s[k]))
        x.rng.bit_generator.state=s['rng'];x.memories=[Agent.from_state(m) for m in s['memories']];return x
    def one(self,i):
        s=self.state();s['seeds']=s['seeds'][i:i+1];s['h']=s['h'][i:i+1];s['prev']=s['prev'][i:i+1];s['memories']=[s['memories'][i]]
        return self.from_state(s)
