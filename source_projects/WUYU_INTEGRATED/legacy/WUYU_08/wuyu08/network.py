"""NumPy neural controllers with explicit recurrent/iterative backpropagation.

Flat and graph networks share a differentiable subgoal-attention motor decoder.
The runtime contains no teacher, path search, argmax goal navigation, or symbolic
motor selection. All primitive action logits are learned.
"""
import copy
import numpy as np
from .memory import softmax
from .features import F,G,N

def ce_grad(p,y,scale):
    d=p.copy();d[np.arange(len(y)),y]-=1
    return d*scale[:,None]

class Network:
    def __init__(self,seed,kind='flat',rounds=3):
        self.kind=kind;self.rounds=rounds;self.hidden=30 if kind=='flat' else 40
        r=np.random.default_rng(seed);h=self.hidden
        def w(a,b):return r.normal(0,1/np.sqrt(a),(a,b))
        if kind=='flat':
            self.p=dict(W=w(N*F+G+N*N,h),R=.6*np.linalg.qr(r.normal(size=(h,h)))[0],b=np.zeros(h),P=w(h,12)*.1,pb=np.zeros(12))
        else:
            self.p=dict(Wn=w(F,h),Wg=w(G,h),R=w(h,h)*.2,S=w(h,h)*.2,
                        M=w(h,h)*.2,B=w(h,h)*.2,b=np.zeros(h),rb=np.zeros(h),P=w(h,2)*.1,pb=np.zeros(2))
        self.p.update(D=w(2*F+G,32),db=np.zeros(32),A=w(32,7)*.1,ab=np.zeros(7))
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v=copy.deepcopy(self.m);self.updates=0
    @property
    def count(self):return sum(v.size for v in self.p.values())
    def zero(self,b):return np.zeros((b,self.hidden)) if self.kind=='flat' else np.zeros((b,N,self.hidden))
    def forward(self,x,h0,rounds=None,cut_edges=False):
        p=self.p;n,g,e,l=[x[k] for k in ('node','glob','edge','local')]
        if cut_edges:e=np.zeros_like(e)
        rec=dict(x=x,h0=h0,edge=e)
        if self.kind=='flat':
            z=np.concatenate([n.reshape(len(n),-1),g,e.reshape(len(n),-1)],1)
            h=np.tanh(z@p['W']+h0@p['R']+p['b']);logits=h@p['P']+p['pb']
            gl,tl=logits[:,:6],logits[:,6:];rec['z']=z
        else:
            emb=n@p['Wn']+g[:,None,:]@p['Wg']+h0@p['R']+p['b']
            hs=[np.tanh(emb)];messages=[]
            for _ in range(self.rounds if rounds is None else rounds):
                z=hs[-1];msg=e@z;back=e.transpose(0,2,1)@z
                hs.append(np.tanh(emb+z@p['S']+msg@p['M']+back@p['B']+p['rb']))
                messages.append((msg,back))
            h=hs[-1];logits=h@p['P']+p['pb'];gl,tl=logits[:,:,0],logits[:,:,1]
            rec.update(emb=emb,hs=hs,messages=messages)
        gp,tp=softmax(gl),softmax(tl)
        context=np.concatenate([(tp[:,:,None]*n).sum(1),(l[:,:,None]*n).sum(1),g],1)
        motor=np.tanh(context@p['D']+p['db']);ap=softmax(motor@p['A']+p['ab'])
        rec.update(h=h,gp=gp,tp=tp,context=context,motor=motor,ap=ap)
        return rec
    def loss(self,records,labels):
        denom=max(1,sum(r['alive'].sum() for r in records));loss=0.
        for r,y in zip(records,labels):
            for key,col,wt in [('ap',0,1.),('gp',1,.3),('tp',2,.6)]:
                loss-=wt*np.sum(np.log(np.maximum(r[key][np.arange(len(y)),y[:,col]],1e-12))*r['alive'])/denom
        return float(loss)
    def gradients(self,records,labels):
        p=self.p;gr={k:np.zeros_like(v) for k,v in p.items()};dh_next=np.zeros_like(records[0]['h'])
        denom=max(1,sum(r['alive'].sum() for r in records))
        for r,y in zip(records[::-1],labels[::-1]):
            mask=r['alive'].astype(float);norm=mask/denom;n=r['x']['node'];g=r['x']['glob'];e=r['edge']
            da=ce_grad(r['ap'],y[:,0],norm)
            gr['A']+=r['motor'].T@da;gr['ab']+=da.sum(0)
            dd=(da@p['A'].T)*(1-r['motor']**2)
            gr['D']+=r['context'].T@dd;gr['db']+=dd.sum(0)
            dc=dd@p['D'].T
            dt=(dc[:,:F,None].transpose(0,2,1)*n).sum(2)
            dt=r['tp']*(dt-(dt*r['tp']).sum(1,keepdims=True))
            dt+=ce_grad(r['tp'],y[:,2],.6*norm);dg=ce_grad(r['gp'],y[:,1],.3*norm)
            if self.kind=='flat':
                dl=np.concatenate([dg,dt],1);gr['P']+=r['h'].T@dl;gr['pb']+=dl.sum(0)
                dh=dl@p['P'].T+dh_next;dz=dh*(1-r['h']**2)*mask[:,None]
                gr['W']+=r['z'].T@dz;gr['R']+=r['h0'].T@dz;gr['b']+=dz.sum(0);dh_next=dz@p['R'].T
            else:
                dl=np.stack([dg,dt],2);gr['P']+=np.einsum('bnh,bnk->hk',r['h'],dl);gr['pb']+=dl.sum((0,1))
                dh=dl@p['P'].T+dh_next;de=np.zeros_like(dh)
                hs=r['hs']
                for k in range(len(hs)-1,0,-1):
                    dz=dh*(1-hs[k]**2)*mask[:,None,None];de+=dz
                    z=hs[k-1];msg,back=r['messages'][k-1]
                    gr['S']+=z.reshape(-1,self.hidden).T@dz.reshape(-1,self.hidden)
                    gr['M']+=msg.reshape(-1,self.hidden).T@dz.reshape(-1,self.hidden)
                    gr['B']+=back.reshape(-1,self.hidden).T@dz.reshape(-1,self.hidden)
                    gr['rb']+=dz.sum((0,1))
                    dh=dz@p['S'].T+e.transpose(0,2,1)@(dz@p['M'].T)+e@(dz@p['B'].T)
                de+=dh*(1-hs[0]**2)*mask[:,None,None]
                gr['Wn']+=n.reshape(-1,F).T@de.reshape(-1,self.hidden)
                gr['Wg']+=g.T@de.sum(1);gr['R']+=r['h0'].reshape(-1,self.hidden).T@de.reshape(-1,self.hidden)
                gr['b']+=de.sum((0,1));dh_next=de@p['R'].T
        return gr
    def update(self,grad,lr=.001):
        self.updates+=1;norm=np.sqrt(sum((g*g).sum() for g in grad.values()));scale=min(1,2/max(norm,1e-12))
        for k,g in grad.items():
            g=g*scale;self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            self.p[k]-=lr*(self.m[k]/(1-.9**self.updates))/(np.sqrt(self.v[k]/(1-.999**self.updates))+1e-8)
    def state(self):return copy.deepcopy(self.__dict__)
    @classmethod
    def from_state(cls,s):
        x=cls.__new__(cls);x.__dict__.update(copy.deepcopy(s));return x
