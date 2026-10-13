"""Online neural associative models + learned motor options + explicit planner.

One-hot inputs make the softmax heads transparent; these do not generalise like
a large shared encoder. Planning and the six-site topology are engineering priors.
"""
import copy
import numpy as np


def softmax(x):
    z=np.exp(x-np.max(x,axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)


class Agent:
    def __init__(self,seed,initialization=701,mode='full',model_lr=1.2,skill_lr=1.,exploration=.08,capacity=256):
        self.seed=int(seed);self.initialization=initialization;self.mode=mode
        self.rng=np.random.default_rng(self.seed*13+initialization*100003)
        prior=np.random.default_rng(initialization)
        self.transition=prior.normal(0,.001,(6,2,3,3))
        self.motor=prior.normal(0,.001,(6,2,3));self.effect=prior.normal(0,.001,(6,4))
        self.model_lr=model_lr;self.skill_lr=skill_lr;self.exploration=exploration;self.capacity=capacity
        self.seen=np.zeros(6,int);self.effect_count=np.zeros(6,int);self.requirement=np.full(6,-2,int)
        self.valid=np.zeros(6,bool);self.skill_count=np.zeros(6,int);self.skill_origin=np.full((6,2),-1,int)
        self.path_action=np.full((6,2),-1,int);self.path_tick=np.full((6,2),-1,int)
        self.memory=[];self.skill_events=[];self.tick=0;self.updates=0;self.calls=np.zeros(3,int)
        self.fixed_weights=(1-np.array([.8,.85]))**2+.02
        self.goal=-1;self.target=-1;self.learn_enabled=True

    @property
    def parameter_count(self):return self.transition.size+self.motor.size+self.effect.size

    def products(self):return np.where(self.effect_count>0,self.effect.argmax(1),-1)

    @staticmethod
    def decode(obs):
        return int(obs[:6].argmax()),int(obs[6:9].argmax()),int(obs[9:12].argmax())-1,obs[12:14],obs[14:16]

    @staticmethod
    def distance(a,b):return min((a-b)%6,(b-a)%6)

    def _route(self,site,pos,inventory,products,visited=()):
        if site in visited:return None,100.
        need=int(self.requirement[site]);cost=float(self.distance(pos,site)+3)
        if need<0 or inventory[need]>0:return site,cost
        providers=np.flatnonzero(products==need+2)
        routes=[]
        for p in providers:
            first,up=self._route(int(p),pos,inventory,products,visited+(site,))
            if first is not None:routes.append((first,up+self.distance(int(p),site)+3))
        if routes:return min(routes,key=lambda z:z[1])
        unknown=np.flatnonzero((products<0)&(self.requirement<0))
        if len(unknown):
            u=min(unknown,key=lambda s:self.distance(pos,int(s)))
            return int(u),cost+6
        return None,100.

    def plan(self,obs,allow_exploration=True):
        pos,stage,need,inventory,body=self.decode(obs);products=self.products()
        weights=self.fixed_weights if self.mode=='fixed_value' else (1-body)**2+.02
        proposals=[]
        for site,product in enumerate(products):
            if product in (2,3):continue
            first,cost=self._route(site,pos,inventory,products)
            if first is None:continue
            gain=.28*(weights[product] if product>=0 else weights.max()*.75)
            # Unobserved sites are optimistic hypotheses, not externally labelled tasks.
            proposals.append((gain/cost,site,first))
        if not proposals:return pos,pos
        best=max(proposals,key=lambda z:z[0]);goal,target=best[1:]
        if allow_exploration and self.rng.random()<self.exploration and body.min()>.25:
            goal=int(self.rng.integers(6));target,_=self._route(goal,pos,inventory,products)
            if target is None:target=goal
        return int(goal),int(target)

    def act(self,obs,forced_goal=None,explore=True):
        obs=np.asarray(obs);pos,stage,need,inventory,body=self.decode(obs)
        self.seen[pos]+=1;self.requirement[pos]=need
        if body.min()<=.001:
            return dict(action=6,obs=obs.copy(),goal=-1,target=-1,source=0,pred=np.zeros(3),skill_origin=-1)
        if forced_goal is None:goal,target=self.plan(obs,explore)
        else:
            goal=int(forced_goal);target,_=self._route(goal,pos,inventory,self.products())
            if target is None:target=goal
        self.goal=goal;self.target=target;source=0;prediction=np.zeros(3);origin=-1
        if pos!=target:
            action=1 if (target-pos)%6<=(pos-target)%6 else 0
        elif stage==2:action=5
        else:
            use_skill=self.mode!='no_skill' and self.valid[pos]
            if use_skill:
                scores=softmax(self.motor[pos,stage]);source=2;self.calls[1]+=1
                origin=int(self.skill_origin[pos,stage])
            else:
                dist=softmax(self.transition[pos,stage]);scores=dist[:,stage+1];source=1;self.calls[0]+=1
            if explore and self.rng.random()<self.exploration:button=int(self.rng.integers(3))
            else:button=int(np.argmax(scores))
            action=button+2;prediction=softmax(self.transition[pos,stage,button]);self.calls[2]+=1
        return dict(action=action,obs=obs.copy(),goal=goal,target=target,source=source,pred=prediction,skill_origin=origin)

    def feedback(self,record,next_obs,reward=0.):
        obs=record['obs'];pos,stage,need,inventory,body=self.decode(obs)
        if body.min()<=.001:self.tick+=1;return
        npos,nstage,_,_,_=self.decode(next_obs);action=record['action'];new_product=int(next_obs[16:21].argmax())
        entry=dict(t=self.tick,pos=pos,stage=stage,action=action,npos=npos,nstage=nstage,product=new_product)
        learning=self.mode!='frozen' and getattr(self,'learn_enabled',True)
        if 2<=action<=4 and stage<2 and pos==npos:
            button=action-2
            if learning:
                grad=softmax(self.transition[pos,stage,button]);grad[nstage]-=1
                self.transition[pos,stage,button]-=self.model_lr*grad;self.updates+=1
            if nstage==stage+1:
                if stage==0:self.path_action[pos]=-1;self.path_tick[pos]=-1
                self.path_action[pos,stage]=button;self.path_tick[pos,stage]=self.tick
                if nstage==2 and np.all(self.path_tick[pos]>=0) and learning:
                    for step in range(2):
                        target=int(self.path_action[pos,step])
                        for _ in range(4):
                            grad=softmax(self.motor[pos,step]);grad[target]-=1
                            self.motor[pos,step]-=self.skill_lr*grad
                    self.valid[pos]=True;self.skill_count[pos]+=1;self.skill_origin[pos]=self.path_tick[pos]
                    self.skill_events.append(dict(t=self.tick,site=pos,actions=self.path_action[pos].tolist(),origins=self.path_tick[pos].tolist()))
                    self.skill_events=self.skill_events[-self.capacity:]
            else:
                self.path_action[pos]=-1;self.path_tick[pos]=-1
                if learning:self.valid[pos]=False
        if new_product<4 and learning:
            grad=softmax(self.effect[pos]);grad[new_product]-=1
            self.effect[pos]-=self.model_lr*grad;self.effect_count[pos]+=1;self.updates+=1
        self.memory.append(entry);self.memory=self.memory[-self.capacity:];self.tick+=1

    def state(self):
        d=copy.deepcopy(self.__dict__);d['rng']=self.rng.bit_generator.state;return d

    @classmethod
    def from_state(cls,state):
        a=cls.__new__(cls);a.__dict__.update(copy.deepcopy(state))
        a.rng=np.random.default_rng();a.rng.bit_generator.state=state['rng'];return a
