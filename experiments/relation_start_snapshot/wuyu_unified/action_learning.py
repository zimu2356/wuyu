"""Real-trajectory residual actor/critic with explicitly saved truncated BPTT.

B: new state cleared at every decision; C: conventional GRU; D: learned
write/retain/erase cell. Existing graph-policy recurrence is retained in all.
All three allocate exactly the same parameters and use the same optimizer.
No task ID, desired target, teacher action or world object is accepted here.
"""
from copy import deepcopy
import numpy as np

SCHEMA = 'whole-residual-actor-critic/1'
DEFAULTS = dict(hidden=16, block=32, rate=.001, gamma=.98, gae=.95,
                value_weight=.25, kl=.01, entropy=.001, residual_cap=2.,
                gradient_cap=1., advantage_cap=4., body_value_prior=True)

def softmax(z):
    p = np.exp(z - np.max(z)); return p / p.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))

class ActionLearner:
    def __init__(self, inputs, seed, mode='D', configuration=None):
        if mode not in ('B','C','D'): raise ValueError(mode)
        self.schema=SCHEMA; self.mode=mode; self.inputs=int(inputs)
        self.c={**DEFAULTS, **(configuration or {})}; h=self.c['hidden']
        r=np.random.default_rng(seed)
        self.p={}
        for k in ('c','u','e'):
            self.p['W'+k]=r.normal(0, 1/np.sqrt(inputs), (inputs,h))*.5
            self.p['R'+k]=r.normal(0, 1/np.sqrt(h), (h,h))*.3
            self.p['b'+k]=np.zeros(h)
        self.p['bu'].fill(-2.)
        # C uses a conventional reset gate; D uses an erase gate. Same bias,
        # parameter count and initial random tensors, different state semantics.
        self.p['be'].fill(-2.)
        self.p['O']=np.zeros((inputs+h,8)); self.p['bo']=np.zeros(8)
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v=deepcopy(self.m); self.h=np.zeros(h)
        self.buffer=[]; self.updates=0; self.transitions=0
        self.last_update=None

    @property
    def count(self): return sum(v.size for v in self.p.values())

    def forward(self, x, base, old=None, suppress=False, value_offset=0.):
        """Pure pre-feedback prediction; the caller commits the returned state."""
        x=np.asarray(x,float); base=np.asarray(base,float)
        if x.shape!=(self.inputs,) or base.shape!=(7,): raise ValueError('input shape')
        p=self.p; old=self.h.copy() if old is None else np.asarray(old).copy()
        if self.mode=='B' or suppress: old.fill(0.)
        u=sigmoid(x@p['Wu']+old@p['Ru']+p['bu'])
        e=sigmoid(x@p['We']+old@p['Re']+p['be'])
        recurrent=e*old if self.mode=='C' else old
        c=np.tanh(x@p['Wc']+recurrent@p['Rc']+p['bc'])
        intermediate=(1-u)*old+u*c
        h=intermediate if self.mode=='C' else (1-e)*intermediate
        z=np.concatenate([x,h]); out=z@p['O']+p['bo']
        cap=self.c['residual_cap']; residual=cap*np.tanh(out[:7]/cap)
        probability=softmax(np.log(np.maximum(base,1e-12))+residual)
        return dict(x=x.copy(),base=base.copy(),old=old,u=u,e=e,c=c,
                    intermediate=intermediate,h=h,z=z,out=out,residual=residual,
                    probability=probability,value=float(out[7]+value_offset),value_offset=float(value_offset),suppress=bool(suppress))

    def commit(self, cache):
        self.h=cache['h'].copy()

    def feedback(self, cache, action, reward, alive, next_value, end=False):
        item=deepcopy(cache)
        item.update(action=int(action),reward=float(reward),alive=bool(alive))
        self.buffer.append(item); self.transitions+=1
        if len(self.buffer)>=self.c['block'] or not alive or end:
            self.last_update=self.learn(float(next_value)); self.buffer=[]
            return deepcopy(self.last_update)
        return None

    def targets(self, bootstrap):
        adv=np.zeros(len(self.buffer)); follow=0.; nv=float(bootstrap)
        for i in reversed(range(len(self.buffer))):
            q=self.buffer[i]
            # Absorbing death has the known constant physical deficit 2. Its
            # discounted tail is a terminal value, not a subgoal bonus.
            if not q['alive']: nv=-2/(1-self.c['gamma']); follow=0.
            delta=q['reward']+self.c['gamma']*nv-q['value']
            follow=delta+self.c['gamma']*self.c['gae']*follow*int(q['alive'])
            adv[i]=follow; nv=q['value']
        returns=adv+np.array([q['value'] for q in self.buffer])
        # No per-batch centering: the sign remains relative to the value model.
        return np.clip(adv,-self.c['advantage_cap'],self.c['advantage_cap']),returns

    def gradients(self, advantages, returns):
        p=self.p; g={k:np.zeros_like(v) for k,v in p.items()}
        carry=np.zeros_like(self.h); n=max(1,len(self.buffer)); loss=0.; kl_sum=0.
        for q,a,ret in zip(self.buffer[::-1],advantages[::-1],returns[::-1]):
            prob=q['probability']; lp=np.log(np.maximum(prob,1e-12))
            ratio=lp-np.log(np.maximum(q['base'],1e-12)); kl=float(prob@ratio)
            gradlog=a*prob; gradlog[q['action']]-=a
            gradlog+=self.c['kl']*prob*(ratio-kl)
            gradlog+=self.c['entropy']*prob*(lp-float(prob@lp))
            do=np.zeros(8)
            do[:7]=gradlog*(1-(q['residual']/self.c['residual_cap'])**2)/n
            do[7]=self.c['value_weight']*(q['value']-ret)/n
            loss+=(-a*lp[q['action']]+self.c['kl']*kl+
                   self.c['entropy']*float(prob@lp)+
                   .5*self.c['value_weight']*(q['value']-ret)**2)/n
            kl_sum+=kl/n
            g['O']+=np.outer(q['z'],do); g['bo']+=do
            dh=do@p['O'][self.inputs:].T+carry
            x,old,u,e,c=q['x'],q['old'],q['u'],q['e'],q['c']
            if self.mode=='C':
                di=dh; de=np.zeros_like(e)
            else:
                di=dh*(1-e); de=-dh*q['intermediate']
            du=di*(c-old); dc=di*u; dold=di*(1-u)
            dzc=dc*(1-c*c)
            recurrent=e*old if self.mode=='C' else old
            g['Wc']+=np.outer(x,dzc);g['Rc']+=np.outer(recurrent,dzc);g['bc']+=dzc
            dr=dzc@p['Rc'].T
            if self.mode=='C': de+=dr*old; dold+=dr*e
            else: dold+=dr
            dzu=du*u*(1-u); dze=de*e*(1-e)
            for k,dz in (('u',dzu),('e',dze)):
                g['W'+k]+=np.outer(x,dz);g['R'+k]+=np.outer(old,dz);g['b'+k]+=dz
                dold+=dz@p['R'+k].T
            carry=np.zeros_like(dold) if self.mode=='B' or q['suppress'] else dold
        return float(loss),g,float(kl_sum)

    def learn(self, bootstrap):
        advantages,returns=self.targets(bootstrap)
        loss,g,kl=self.gradients(advantages,returns)
        norm=float(np.sqrt(sum(np.sum(v*v) for v in g.values())))
        scale=min(1.,self.c['gradient_cap']/max(norm,1e-12));self.updates+=1
        for k,v in g.items():
            v=v*scale;self.m[k]=.9*self.m[k]+.1*v;self.v[k]=.999*self.v[k]+.001*v*v
            self.p[k]-=self.c['rate']*(self.m[k]/(1-.9**self.updates))/(np.sqrt(self.v[k]/(1-.999**self.updates))+1e-8)
        return dict(loss=loss,gradient_norm=norm,kl=kl,updates=self.updates,
                    transitions=len(self.buffer),mean_return=float(returns.mean()),
                    clipped_advantages=int(np.sum(np.abs(advantages)>=self.c['advantage_cap'])))

    def state(self): return deepcopy(vars(self))

    @classmethod
    def restore(cls,s):
        if s.get('schema')!=SCHEMA: raise ValueError('incompatible action learner')
        obj=cls.__new__(cls);obj.__dict__.update(deepcopy(s));return obj
