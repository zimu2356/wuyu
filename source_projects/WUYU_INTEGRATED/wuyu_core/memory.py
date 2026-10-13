"""Observed transition and motor memory from WUYU_07/08; planner removed."""
import copy
import numpy as np


def softmax(x):
    z=np.exp(x-np.max(x,axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)


class ExperienceMemory:
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
