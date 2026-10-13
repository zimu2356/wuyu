"""Explicit recurrent actor-critic gradients. No autodiff dependency."""
import numpy as np

H=32;G=6;A=7;D=31;P=A*3


def softmax(x):
    z=np.exp(x-x.max(axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)


class Network:
    def __init__(self,seed,mode='full'):
        self.mode=mode;r=np.random.default_rng(seed)
        self.p={'W':r.normal(0,1/np.sqrt(D),(D,H)),
          'R':.75*np.linalg.qr(r.normal(size=(H,H)))[0],'b':np.zeros(H),
          'M':r.normal(0,.03,(H,P)),'mb':np.zeros(P),
          'A':r.normal(0,.02,(H,A)),'ab':np.zeros(A),
          'GA':r.normal(0,.02,(G,A)),'PA':r.normal(0,.02,(P,A)),
          'Q':r.normal(0,.03,(H,G)),'V':np.zeros(H),'vb':np.zeros(1)}
        self.adam_m={k:np.zeros_like(v) for k,v in self.p.items()};self.adam_v={k:np.zeros_like(v) for k,v in self.p.items()};self.updates=0

    @property
    def count(self):return sum(v.size for v in self.p.values())

    def encode(self,x,h):return np.tanh(x@self.p['W']+h@self.p['R']+self.p['b'])

    def forward(self,x,h0,fast,fast_b,candidates,valid,current_goal,choose,ug,ua,forecast=True):
        p=self.p;h=self.encode(x,h0)
        pred=np.tanh(h@p['M']+np.einsum('nh,nhp->np',h,fast)+p['mb']+fast_b)
        query=h@p['Q'];scores=np.einsum('ng,nkg->nk',query,candidates)
        scores=np.where(valid,scores,-1e9);gp=softmax(scores)
        gi=(ug[:,None]>np.cumsum(gp,axis=1)).sum(axis=1);gi=np.minimum(gi,candidates.shape[1]-1)
        goal=np.where(choose[:,None],candidates[np.arange(len(x)),gi],current_goal)
        if self.mode=='plain_rnn':goal=np.zeros_like(goal)
        used_pred=pred if forecast and self.mode!='plain_rnn' else np.zeros_like(pred)
        logits=h@p['A']+goal@p['GA']+used_pred@p['PA']+p['ab']
        ap=softmax(logits);action=(ua[:,None]>np.cumsum(ap,axis=1)).sum(axis=1);action=np.minimum(action,A-1)
        value=h@p['V']+p['vb'][0]
        return dict(x=x.copy(),h0=h0.copy(),h=h,pred=pred,used_pred=used_pred,goal=goal,gp=gp,gi=gi,
                    ap=ap,action=action,value=value,candidates=candidates.copy(),valid=valid.copy(),choose=choose.copy(),
                    fast=fast.copy(),forecast=forecast)

    def gradients(self,records,returns,advantages,entropy=.012,model_coef=.3):
        p=self.p;gr={k:np.zeros_like(v) for k,v in p.items()};dh_next=np.zeros_like(records[0]['h'])
        denom=max(1,sum(r['alive'].sum() for r in records));losses=np.zeros(4)
        for ti in reversed(range(len(records))):
            r=records[ti];live=r['alive'].astype(float);norm=live/denom
            ap=r['ap'];dl=ap.copy();dl[np.arange(len(live)),r['action']]-=1
            dl*= (advantages[ti]*norm)[:,None]
            logap=np.log(np.maximum(ap,1e-12));en=(ap*logap).sum(1,keepdims=True)
            dl+=entropy*ap*(logap-en)*norm[:,None]
            gr['A']+=r['h'].T@dl;gr['ab']+=dl.sum(0);gr['GA']+=r['goal'].T@dl;gr['PA']+=r['used_pred'].T@dl
            dh=dl@p['A'].T+dh_next
            dp=dl@p['PA'].T if r['forecast'] and self.mode!='plain_rnn' else np.zeros_like(r['pred'])
            err=r['pred'].reshape(-1,A,3)[np.arange(len(live)),r['action']]-r['target']
            dp=dp.reshape(-1,A,3);dp[np.arange(len(live)),r['action']]+=model_coef*err*norm[:,None]/3
            dp=dp.reshape(-1,P)*(1-r['pred']**2)
            gr['M']+=r['h'].T@dp;gr['mb']+=dp.sum(0)
            dh+=dp@p['M'].T+np.einsum('np,nhp->nh',dp,r['fast'])
            dv=.5*(r['value']-returns[ti])*norm
            gr['V']+=r['h'].T@dv;gr['vb'][0]+=dv.sum();dh+=dv[:,None]*p['V']
            if self.mode!='plain_rnn':
                gp=r['gp'];dg=gp.copy();dg[np.arange(len(live)),r['gi']]-=1
                gn=norm*r['choose'];dg*=(advantages[ti]*gn)[:,None]
                loggp=np.log(np.maximum(gp,1e-12));dg+=entropy*gp*(loggp-(gp*loggp).sum(1,keepdims=True))*gn[:,None]
                dq=np.einsum('nk,nkg->ng',dg,r['candidates']);gr['Q']+=r['h'].T@dq;dh+=dq@p['Q'].T
            dz=dh*(1-r['h']**2)*live[:,None]
            gr['W']+=r['x'].T@dz;gr['R']+=r['h0'].T@dz;gr['b']+=dz.sum(0);dh_next=dz@p['R'].T
            losses += [float((-np.log(np.maximum(ap[np.arange(len(live)),r['action']],1e-12))*advantages[ti]*norm).sum()),
                       float((.25*(r['value']-returns[ti])**2*norm).sum()),float((.5*model_coef*(err**2).mean(1)*norm).sum()),float((-en[:,0]*norm).sum())]
        return gr,losses

    def update(self,grad,lr=.001):
        self.updates+=1;norm=np.sqrt(sum((v*v).sum() for v in grad.values()));scale=min(1.,1./max(norm,1e-12))
        for k,g in grad.items():
            g=g*scale;self.adam_m[k]=.9*self.adam_m[k]+.1*g;self.adam_v[k]=.999*self.adam_v[k]+.001*g*g
            m=self.adam_m[k]/(1-.9**self.updates);v=self.adam_v[k]/(1-.999**self.updates)
            self.p[k]-=lr*m/(np.sqrt(v)+1e-8)
        return float(norm)

    def state(self):return {'mode':self.mode,'p':self.p,'adam_m':self.adam_m,'adam_v':self.adam_v,'updates':self.updates}

    @classmethod
    def from_state(cls,state):
        n=cls(0,state['mode']);n.p=state['p'];n.adam_m=state['adam_m'];n.adam_v=state['adam_v'];n.updates=state['updates'];return n
