"""Closed simulation. Only observation() crosses the agent boundary."""
import numpy as np

ACTIONS = ('left','right','interact','inspect','replay','rest','reconsider')
COST = np.array([.022,.022,.017,.025,.027,.009,.012])


class World:
    def __init__(self,seeds,total=512,formal=False,variant=0):
        self.seeds=np.asarray(seeds,dtype=np.int64);self.n=len(seeds)
        self.total=total;self.formal=formal;self.variant=variant;self.t=0
        self.pos=self.seeds%4
        self.body=np.tile([.68,.78],(self.n,1))
        self.alive=np.ones(self.n,bool);self.cool=np.zeros((self.n,4),int)
        self.inspect=np.zeros(self.n,bool)
        maps=[];random=[];bounds=[]
        for s in self.seeds:
            rng=np.random.default_rng(int(s)+1660000+variant*13001)
            maps.append(np.stack([rng.permutation([0,0,1,2]) for _ in range(3)]))
            random.append(rng.random((total+1,4)))
            q=total//4;j=max(1,total//32)
            bounds.append([q+int(rng.integers(-j,j+1)),2*q+int(rng.integers(-j,j+1)),3*q+int(rng.integers(-j,j+1))])
        self.maps=np.stack(maps);self.random=np.stack(random,axis=1);self.bounds=np.array(bounds)

    def phase(self):return (self.t>=self.bounds).sum(axis=1)

    def map_now(self):
        p=self.phase();idx=np.where(p==0,0,np.where(p<3,1,2))
        return self.maps[np.arange(self.n),idx]

    def observation(self):
        p=self.phase();kind=self.map_now()[np.arange(self.n),self.pos]
        reliability=(.55+.43*self.body[:,1])*np.where(p==2,.68,np.where(p==3,.82,1.))
        reliability=np.where(self.inspect,.99,reliability)
        u=self.random[self.t]
        wrong=(kind+1+(u[:,1]>.5))%3
        cue=np.where(u[:,0]<reliability,kind,wrong)
        return np.concatenate([np.eye(4)[self.pos],np.eye(3)[cue],self.body],axis=1)

    def step(self,action):
        if self.t>=self.total:raise StopIteration
        action=np.asarray(action,int);before=self.body.copy();oldpos=self.pos.copy();was=self.alive.copy()
        p=self.phase();u=self.random[self.t];self.inspect[:]=False
        # Paying for the operation is a physical precondition, not only reward shaping.
        cost=COST[action]*np.where(p==3,1.18,1.)
        allowed=was & (self.body[:,0]>=cost)
        self.body[:,0]=np.maximum(0,self.body[:,0]-np.where(was,cost,0))
        self.body[:,1]=np.maximum(.02,self.body[:,1]-np.where(was,.0014,0))
        self.cool=np.maximum(0,self.cool-1)
        move=allowed & (action<2) & (u[:,2] < (.72+.28*before[:,1])*np.where(p==2,.62,np.where(p==3,.82,1.)))
        self.pos=(self.pos+np.where(move,np.where(action==0,-1,1),0))%4
        self.inspect=allowed & (action==3)
        kind=self.map_now()[np.arange(self.n),self.pos]
        available=self.cool[np.arange(self.n),self.pos]==0
        use=allowed & (action==2) & available & (kind<2)
        energy=use & (kind==0);repair=use & (kind==1)
        self.body[energy,0]=np.minimum(1,self.body[energy,0]+.36)
        self.body[repair,1]=np.minimum(1,self.body[repair,1]+.32)
        self.cool[np.flatnonzero(use),self.pos[use]]=14
        self.alive=was & allowed & (self.body[:,0]>.001)
        # Failed/dead bodies are absorbing; no automatic rescue in a formal lifetime.
        self.body[~was]=before[~was]
        delta=self.body-before
        reward=delta.sum(axis=1)+.02*self.alive-1.*(was & ~self.alive)
        reward=np.where(was,reward,0.)
        target=np.column_stack([delta/.36,(self.pos!=oldpos).astype(float)])
        self.t+=1
        return dict(reward=reward,target=target,delta=delta,was_alive=was,alive=self.alive.copy(),
                    allowed=allowed,successful_interaction=use,position=self.pos.copy(),body=self.body.copy(),phase=p)
