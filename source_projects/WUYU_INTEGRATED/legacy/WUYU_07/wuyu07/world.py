"""Batch world. observation() is the only interface presented to agents."""
import copy
import numpy as np

ACTIONS = ('left','right','button0','button1','button2','take','wait')
OBS = 21
COST = np.array([.003,.003,.004,.004,.004,.004,.0018])


class World:
    def __init__(self, seeds, total=1536):
        self.seeds=np.asarray(seeds,dtype=np.int64);self.n=len(seeds);self.total=total;self.t=0
        self.pos=self.seeds%6;self.body=np.tile([.8,.85],(self.n,1));self.alive=np.ones(self.n,bool)
        self.stage=np.zeros((self.n,6),int);self.inventory=np.zeros((self.n,2),int)
        self.last=np.full(self.n,4,int);self.regime_override=None
        products=[];codes=[];requirements=[];changed=[];replacement=[]
        for seed in self.seeds:
            r=np.random.default_rng(int(seed)+707071)
            product=r.permutation([0,0,1,1,2,3]);products.append(product)
            code=r.integers(0,3,(6,2));codes.append(code)
            req=np.where(product<2,r.integers(0,2,6),-1);requirements.append(req)
            site=int(r.integers(6));changed.append(site)
            replacement.append((code[site]+r.integers(1,3,2))%3)
        self.products=np.array(products);self.codes=np.array(codes);self.base_req=np.array(requirements)
        self.changed=np.array(changed);self.replacement=np.array(replacement)

    def phase(self):
        return min(3,self.t//max(1,self.total//4)) if self.regime_override is None else int(self.regime_override)

    def requirements(self):
        if self.phase()==0:return np.full((self.n,6),-1,int)
        req=self.base_req.copy()
        if self.phase()==3:
            # Two first physical positions that produce body resources change their input.
            for i in range(self.n):
                sites=np.flatnonzero(self.products[i]<2)[:2];req[i,sites]=1-req[i,sites]
        return req

    def codes_now(self):
        codes=self.codes.copy()
        if self.phase()==3:codes[np.arange(self.n),self.changed]=self.replacement
        return codes

    def observation(self):
        rows=np.arange(self.n);req=self.requirements()[rows,self.pos]
        return np.concatenate([np.eye(6)[self.pos],np.eye(3)[self.stage[rows,self.pos]],
            np.eye(3)[req+1],self.inventory,self.body,np.eye(5)[self.last]],axis=1)

    def step(self,action):
        if self.t>=self.total:raise StopIteration
        action=np.asarray(action,int)
        if action.shape!=(self.n,) or np.any((action<0)|(action>6)):raise ValueError('action shape/range')
        rows=np.arange(self.n);before=self.body.copy();was=self.alive.copy();oldpos=self.pos.copy();oldstage=self.stage[rows,self.pos].copy()
        phase=self.phase();cost=COST[action]*(.55 if phase>=2 else 1.)
        allowed=was & (self.body[:,0]>=cost) & (self.body[:,1]>.001)
        self.body[:,0]=np.maximum(0,self.body[:,0]-cost*was)
        self.body[:,1]=np.maximum(0,self.body[:,1]-(.0018 if phase>=2 else .0007)*was)
        self.last[:]=4
        move=allowed & (action<2)
        self.pos=(self.pos+np.where(move,np.where(action==0,-1,1),0))%6
        button=allowed & (action>=2)&(action<=4)&(oldstage<2)
        indices=np.flatnonzero(button)
        codes=self.codes_now()
        correct=codes[indices,oldpos[indices],oldstage[indices]]==(action[indices]-2)
        self.stage[indices,oldpos[indices]]=np.where(correct,oldstage[indices]+1,0)
        req=self.requirements()[rows,self.pos]
        has=(req<0)|(self.inventory[rows,np.maximum(req,0)]>0)
        take=allowed&(action==5)&(self.stage[rows,self.pos]==2)&has
        rr=np.flatnonzero(take);kind=self.products[rr,self.pos[rr]]
        need=rr[req[rr]>=0];self.inventory[need,req[need]]-=1
        for k in range(4):
            ii=rr[kind==k]
            if k<2:self.body[ii,k]=np.minimum(1,self.body[ii,k]+.28)
            else:self.inventory[ii,k-2]=1
        self.last[rr]=kind;self.stage[rr,self.pos[rr]]=0
        self.alive=was & allowed & (self.body.min(1)>.001)
        self.body[~was]=before[~was]
        delta=self.body-before
        reward=delta.sum(1)+.006*self.alive-1.*(was & ~self.alive)
        reward=np.where(was,reward,0.)
        self.t+=1
        return dict(reward=reward,alive=self.alive.copy(),was=was,allowed=allowed,
            phase=np.full(self.n,phase,int),body=self.body.copy(),position=self.pos.copy(),
            stage=self.stage.copy(),inventory=self.inventory.copy(),product=self.last.copy(),delta=delta)

    def one(self,index):
        w=copy.deepcopy(self)
        for key,value in list(w.__dict__.items()):
            if isinstance(value,np.ndarray):setattr(w,key,value[index:index+1].copy())
        w.n=1;return w

    def state(self):return self.__dict__.copy()

    @classmethod
    def from_state(cls,state):
        w=cls.__new__(cls);w.__dict__.update(state);return w
