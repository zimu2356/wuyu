"""Neural direct-action student; no dependency planner at deployment.

Training labels are produced by the hybrid teacher using only its own past.
Runtime features contain learned models, not hidden world state. The actor is
pretrained; the 168 associative/skill parameters per individual remain plastic.
"""
import copy,time,json
from pathlib import Path
import numpy as np
from .agent import Agent,softmax
from .world import World
from .rnn import Network,RNNAgent
from .state import write_tree

FEATURES=105


def features(obs,prev,memories):
    extra=[]
    for o,m in zip(obs,memories):
        pos,stage,need,_,_=m.decode(o)
        if o[14:16].min()>.001:m.seen[pos]+=1;m.requirement[pos]=need
        known=m.effect_count>0
        products=softmax(m.effect)*known[:,None]
        req=np.eye(4)[m.requirement+2]
        trans=softmax(m.transition[pos,min(stage,1)]).ravel()
        skill=softmax(m.motor[pos]).ravel()*m.valid[pos]
        extra.append(np.concatenate([products.ravel(),known,req.ravel(),trans,skill,m.valid]))
    result=np.concatenate([obs,prev,np.asarray(extra)],1)
    assert result.shape[1]==FEATURES
    return result


class Student(RNNAgent):
    def __init__(self,net,seeds,initialization=701,mode='neural_student'):
        super().__init__(net,seeds);self.initialization=initialization;self.mode=mode
        self.memories=[Agent(s,initialization) for s in seeds]
    def act(self,obs):
        x=features(obs,self.prev,self.memories)
        r=self.net.forward(x,self.h,self.uniform());self.h=r['h'].copy()
        r['obs']=obs.copy();r['alive']=obs[:,14:16].min(1)>.001;return r
    def feedback(self,r,next_obs,reward):
        for i,m in enumerate(self.memories):
            m.feedback(dict(action=int(r['action'][i]),obs=r['obs'][i]),next_obs[i],reward[i])
        super().feedback(r,next_obs,reward)
    def state(self):
        s=super().state();s.update(initialization=self.initialization,mode=self.mode,memories=[m.state() for m in self.memories]);return s
    @classmethod
    def from_state(cls,s):
        a=cls(Network.from_state(s['network']),s['seeds'],s['initialization'],s['mode'])
        for k in ['h','prev','rng','tick']:setattr(a,k,s[k])
        a.memories=[Agent.from_state(m) for m in s['memories']];return a


def train_student(config,initialization,output):
    from .experiment import rollout
    seed=config['initializations'][initialization];net=Network(seed,32,FEATURES)
    start=time.perf_counter();w=None;h=None;logs=[];selection=[];output=Path(output)
    for update in range(512):
        if w is None or w.t>=w.total:
            birth=update//48;seeds=760000+initialization*1000000+birth*32+np.arange(32)
            w=World(seeds,1536);memories=[Agent(s,seed) for s in seeds];prev=np.zeros((32,9));h=np.zeros((32,32))
        records=[];labels=[]
        for _ in range(32):
            obs=w.observation();x=features(obs,prev,memories)
            # Target action uses the teacher's learned knowledge, never the simulator's code/map.
            targets=np.array([a.act(o,explore=False)['action'] for a,o in zip(memories,obs)])
            actual=[a.act(o,explore=True) for a,o in zip(memories,obs)];action=np.array([r['action'] for r in actual])
            r=net.forward(x,h,np.zeros(32));h=r['h'].copy();r['alive']=w.alive.copy();records.append(r);labels.append(targets)
            out=w.step(action);nobs=w.observation()
            for i,a in enumerate(memories):a.feedback(actual[i],nobs[i],out['reward'][i])
            prev=np.concatenate([np.eye(7)[action],(nobs[:,14:16]-obs[:,14:16])/.28],1)
        net.update(net.imitation_gradients(records,labels),lr=.002)
        if (update+1)%64==0:
            acc=float(np.mean([np.mean(r['ap'].argmax(1)==y) for r,y in zip(records,labels)]))
            logs.append(dict(update=update+1,teacher_accuracy=acc,seconds=time.perf_counter()-start));print(json.dumps(dict(student=initialization,**logs[-1])),flush=True)
        if (update+1)%256==0:
            m,_,_,_,_=rollout(730000+np.arange(config['validation_worlds']),seed,'neural_student',net=net)
            filename=f'model_student_{initialization}_{update+1}.npz';write_tree(output/filename,net.state())
            sel=dict(update=update+1,file=filename,survival=float(m['survival'].mean()),body_area=float(m['body_area'].mean()));selection.append(sel)
            print(json.dumps(dict(student_validation=initialization,**sel)),flush=True)
    best=max(selection,key=lambda x:(x['survival'],x['body_area']))
    result=dict(initialization=initialization,parameters=net.count,active_actor_parameters=net.count-net.hidden-1,
                per_individual_plastic_parameters=Agent(0).parameter_count,logs=logs,validation=selection,selected=best,seconds=time.perf_counter()-start)
    (output/f'training_student_{initialization}.json').write_text(json.dumps(result,indent=2));return result
