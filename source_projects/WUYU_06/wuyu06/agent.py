import numpy as np
from .network import H,D,A,G,P


class Stream:
    def __init__(self,seeds,offset):
        self.state=(np.asarray(seeds,dtype=np.uint64)+np.uint64(offset))*np.uint64(6364136223846793005)+np.uint64(1442695040888963407)
        self.state[self.state==0]=1
    def draw(self):
        x=self.state.copy();x^=x>>np.uint64(12);x^=x<<np.uint64(25);x^=x>>np.uint64(27);self.state=x
        return ((x*np.uint64(2685821657736338717))>>np.uint64(11)).astype(float)/9007199254740992.


class Agent:
    def __init__(self,net,seeds,mode='full',fast_lr=.018,capacity=64,burn=8):
        self.net=net;self.seeds=np.asarray(seeds);self.n=len(seeds);self.mode=mode;self.fast_lr=fast_lr
        self.h=np.zeros((self.n,H));self.fast=np.zeros((self.n,H,P));self.fast_b=np.zeros((self.n,P))
        self.prev_action=np.zeros((self.n,A));self.prev_delta=np.zeros((self.n,2));self.recalled=np.zeros((self.n,G))
        self.goal=np.zeros((self.n,G));self.goal_age=np.zeros(self.n,int);self.need_goal=np.ones(self.n,bool);self.goal_origin=np.full(self.n,-1,int)
        self.goals=np.zeros((self.n,8,G));self.goal_valid=np.zeros((self.n,8),bool);self.goal_origins=np.full((self.n,8),-1,int)
        self.capacity=capacity;self.burn=burn;self.tick=0;self.count=np.zeros(self.n,int)
        self.mx=np.zeros((self.n,capacity,D));self.mpast=np.zeros((self.n,capacity,burn,D));self.mlen=np.zeros((self.n,capacity),int)
        self.ma=np.zeros((self.n,capacity),int);self.my=np.zeros((self.n,capacity,3));self.mt=np.full((self.n,capacity),-1,int)
        self.history=np.zeros((self.n,burn,D));self.history_len=np.zeros(self.n,int)
        self.policy_rng=Stream(seeds,1741000);self.memory_rng=Stream(seeds,1751000)
        self.replay_updates=np.zeros(self.n,int);self.replay_reads=np.zeros(self.n,int);self.online_updates=np.zeros(self.n,int);self.goal_switches=np.zeros(self.n,int)
        self.rule_kind=np.full((self.n,4),-1,int);self.rule_used=np.full((self.n,4),-100,int)

    def inputs(self,obs):
        goal=self.goal if self.mode not in ('plain_rnn','no_goal') else np.zeros_like(self.goal)
        recalled=self.recalled if self.mode!='plain_rnn' else np.zeros_like(self.recalled)
        return np.concatenate([obs,self.prev_action,self.prev_delta,recalled,(self.goal_age/24)[:,None],goal],axis=1)

    def act(self,obs,uniforms=None):
        if uniforms is None:ug,ua=self.policy_rng.draw(),self.policy_rng.draw()
        else:ug,ua=uniforms
        x=self.inputs(obs);cand=self.goals.copy();valid=self.goal_valid.copy()
        empty=~valid.any(1);valid[empty,0]=True
        if self.mode in ('plain_rnn','no_goal'):cand[:]=0
        choose=self.need_goal.copy();oldgoal=self.goal.copy()
        r=self.net.forward(x,self.h,self.fast,self.fast_b,cand,valid,self.goal,choose,ug,ua,forecast=self.mode!='no_forecast')
        self.h=r['h'].copy();self.goal=r['goal'].copy();self.goal_age[choose]=0;self.need_goal[:]=False
        self.goal_origin=np.where(choose,self.goal_origins[np.arange(self.n),r['gi']],self.goal_origin)
        r['goal_origin']=self.goal_origin.copy()
        self.goal_switches+=(choose & np.any(np.abs(oldgoal-self.goal)>1e-10,axis=1))
        if self.mode=='rule':r['action']=self.rule_action(obs)
        r['alive']=obs[:,7]>.001;r['obs']=obs.copy()
        return r

    def rule_action(self,obs):
        pos=obs[:,:4].argmax(1);cue=obs[:,4:7].argmax(1);body=obs[:,7:9];action=np.full(self.n,5,int)
        for i in range(self.n):
            wanted=0 if body[i,0]<.62 else (1 if body[i,1]<.68 else 0)
            known=np.flatnonzero(self.rule_kind[i]==wanted)
            if (cue[i]==wanted or self.rule_kind[i,pos[i]]==wanted) and self.tick-self.rule_used[i,pos[i]]>=14:
                action[i]=2;continue
            if len(known):
                distance=np.minimum((known-pos[i])%4,(pos[i]-known)%4)
                target=int(known[np.argmin(distance)])
                if target==pos[i]:continue
                action[i]=1 if (target-pos[i])%4 <= (pos[i]-target)%4 else 0
            else:action[i]=1
        return action

    def _fast_update(self,h,action,target,mask):
        rows=np.arange(self.n);pred=np.tanh(h@self.net.p['M']+np.einsum('nh,nhp->np',h,self.fast)+self.net.p['mb']+self.fast_b).reshape(self.n,A,3)
        error=(pred[rows,action]-target)*(1-pred[rows,action]**2)/3
        error*=mask[:,None];flat=np.zeros((self.n,A,3));flat[rows,action]=error;flat=flat.reshape(self.n,P)
        self.fast-=self.fast_lr*h[:,:,None]*flat[:,None,:];self.fast_b-=self.fast_lr*flat
        np.clip(self.fast,-2,2,out=self.fast);np.clip(self.fast_b,-2,2,out=self.fast_b)

    def feedback(self,r,next_obs,reward):
        # All targets are reconstructed from the two visible body/position observations.
        delta=next_obs[:,7:9]-r['obs'][:,7:9]
        moved=(next_obs[:,:4].argmax(1)!=r['obs'][:,:4].argmax(1)).astype(float)
        target=np.column_stack([delta/.36,moved]);r['target']=target;r['reward']=np.asarray(reward).copy()
        r['after_alive']=next_obs[:,7]>.001
        learn=r['alive'] & (self.mode not in ('frozen_fast','rule','plain_rnn'))
        self._fast_update(r['h'],r['action'],target,learn);self.online_updates+=learn
        u=self.memory_rng.draw();replay=r['alive'] & r['after_alive'] & (r['action']==4) & (self.count>0)
        replay &= self.mode not in ('no_replay','rule','plain_rnn')
        # A bounded context-matching sampler. Selection is engineered, not learned.
        usable=np.arange(self.capacity)[None,:]<np.minimum(self.count,self.capacity)[:,None]
        similarity=np.einsum('nkd,nd->nk',self.mx[:,:,:4],r['x'][:,:4])
        weights=np.where(usable,1+3*similarity,0.);probs=weights/np.maximum(1,weights.sum(1,keepdims=True))
        slot=np.minimum((u[:,None]>np.cumsum(probs,axis=1)).sum(1),self.capacity-1)
        rows=np.arange(self.n);mh=np.zeros_like(self.h)
        if replay.any():
            for j in range(self.burn):
                nh=self.net.encode(self.mpast[rows,slot,j],mh)
                mh=np.where((j<self.mlen[rows,slot])[:,None],nh,mh)
            mh=self.net.encode(self.mx[rows,slot],mh)
            self._fast_update(mh,self.ma[rows,slot],self.my[rows,slot],replay & (self.mode!='frozen_fast'))
            recalled=np.concatenate([self.mx[rows,slot,:4],self.my[rows,slot,:2]],axis=1)
            self.recalled=np.where(replay[:,None],recalled,self.recalled)
        self.replay_reads+=replay;self.replay_updates+=replay & (self.mode!='frozen_fast')
        r['replay_applied']=replay.copy();r['replay_origin']=np.where(replay,self.mt[rows,slot],-1)
        # Only already experienced, positive body outcomes enter the goal pool.
        positive=(delta.max(1)>.06) & r['alive'];kind=delta.argmax(1);pos=next_obs[:,:4].argmax(1);gslot=2*pos+kind
        goal=np.concatenate([next_obs[:,:4],np.maximum(0,delta/.36)],axis=1)
        rr=np.flatnonzero(positive);gs=gslot[rr]
        self.goals[rr,gs]=goal[rr];self.goal_valid[rr,gs]=True;self.goal_origins[rr,gs]=self.tick
        self.rule_kind[rr,pos[rr]]=kind[rr];self.rule_used[rr,pos[rr]]=self.tick
        achieved=positive & (self.goal[:,:4].argmax(1)==pos) & ((self.goal[:,4:]*goal[:,4:]).sum(1)>.04)
        self.goal_age+=r['alive'].astype(int);self.need_goal=(self.goal_age>=24) | achieved | (r['action']==6)
        r['goal_achieved']=achieved;r['goal_age_after']=self.goal_age.copy();r['positive_outcome']=positive
        # Append only after all current replay reads; preserve all pre-event inputs.
        keep=np.flatnonzero(r['alive']);ms=self.count[keep]%self.capacity
        self.mx[keep,ms]=r['x'][keep];self.mpast[keep,ms]=self.history[keep];self.mlen[keep,ms]=self.history_len[keep]
        self.ma[keep,ms]=r['action'][keep];self.my[keep,ms]=target[keep];self.mt[keep,ms]=self.tick
        self.count+=r['alive'].astype(int)
        for i in keep:
            if self.history_len[i]<self.burn:self.history[i,self.history_len[i]]=r['x'][i];self.history_len[i]+=1
            else:self.history[i,:-1]=self.history[i,1:];self.history[i,-1]=r['x'][i]
        self.prev_action=np.eye(A)[r['action']];self.prev_delta=delta/.36;self.tick+=1
        return r
