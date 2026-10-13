"""Exploratory 2x2 intervention, identical developmental state per comparison."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,copy,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'inputs'))
from wuyu05.core import Agent,World,step,save,MAX_RATE,sigmoid
from wuyu05.control import encode


class Mixed(Agent):
    def rates(self,f,err,re):
        dynamic=MAX_RATE*sigmoid(np.einsum('oi,ni->no',self.learned_theta,f))
        fixed=np.broadcast_to(MAX_RATE*sigmoid(self.fixed_theta),dynamic.shape)
        return np.column_stack([dynamic[:,0] if self.use_fresh else fixed[:,0],dynamic[:,1] if self.use_replay else fixed[:,1]])

    def feedback(self,x,h0,p,answer,reward,valid,state,action,after):
        y=np.where(reward>0,answer,1-answer);sx,sh,sy,sv=self.memory.sample(self.net)
        self.forward_micro+=self.memory.burn*4;rp=self.net.forward(sx,sh)[0];self.forward_micro+=4
        f,err,re=self.features(p,y,valid,rp,sy,sv);rates=self.rates(f,err,re)
        g=self.net.gradients(x,h0,y,valid);r=self.net.gradients(sx,sh,sy,sv)
        dot=sum((g[k]*r[k]).reshape(self.n,-1).sum(1) for k in g)
        ng=np.sqrt(sum((v*v).reshape(self.n,-1).sum(1) for v in g.values()));nr=np.sqrt(sum((v*v).reshape(self.n,-1).sum(1) for v in r.values()))
        self.last_cos=np.where(valid & sv,dot/np.maximum(ng*nr,1e-15),np.nan)
        self.learn(x,h0,y,valid,rates[:,0]);self.learn(sx,sh,sy,sv,rates[:,1])
        self.fresh_nonzero+=(valid & (rates[:,0]>0));self.replay_nonzero+=(sv & (rates[:,1]>0))
        self.ctrl.update(state,action,reward,after);self.memory.add(x,y,valid)
        self.fast=np.where(valid,self.fast+(err-self.fast)/8,self.fast);self.slow=np.where(valid,self.slow+(err-self.slow)/64,self.slow)
        self.previous_error=np.where(valid,err,self.previous_error);self.previous_reward=reward.copy();self.previous_answer=np.where(valid,2*answer-1,0);self.t+=1
        return rates,f,sv


def forced_step(w,a,action):
    ctx,_,_,phase=w.conditions();before=w.charge.copy();state=encode(before,ctx);cue,valid=w.acquire(action)
    x=a.inputs(ctx,before,action,cue,valid);h0=a.net.h.copy();p=a.net.observe(x);a.forward_micro+=4
    ans=(p>=.5).astype(np.int8);reward=w.finish(ans,valid);rates,_,_=a.feedback(x,h0,p,ans,reward,valid,state,action,encode(w.charge,ctx))
    return dict(action=action,reward=reward,valid=valid,rate_current=rates[:,0],rate_replay=rates[:,1])


if __name__=='__main__':
    started=time.perf_counter();out=ROOT/'results';selection=json.loads((ROOT/'inputs/wuyu05_selection.json').read_text())
    seeds=np.arange(853000,853032);records={};report={};rng=np.random.default_rng(960050);ix=rng.integers(0,32,(4000,32))
    def est(x):return {'mean':float(x.mean()),'ci95':np.quantile(x[ix].mean(1),[.025,.975]).tolist()}
    for origin in ['fixed','learned']:
        theta=np.asarray(selection[origin]['theta']);a=Agent(seeds,origin,np.broadcast_to(theta,(32,)+theta.shape).copy());w=World(seeds,'prefix',True)
        while w.t<w.total:step(w,a)
        save(out/f'w05_prefix_{origin}.npz',w,a)
        for branch in ['return','change','noise']:
            script=None
            for coupling in ['closed','yoked']:
                group=f'{origin}_{branch}_{coupling}';group_data={};report[group]={}
                for tag,lf,lr in [('FF',False,False),('LF',True,False),('FL',False,True),('LL',True,True)]:
                    b=copy.deepcopy(a);b.__class__=Mixed;b.learned_theta=np.asarray(selection['learned']['theta']);b.fixed_theta=np.asarray(selection['fixed']['theta']);b.use_fresh=lf;b.use_replay=lr
                    v=World(seeds,branch,True);v.charge=w.charge.copy();rows=[]
                    for t in range(512):
                        rec=step(v,b) if coupling=='closed' else forced_step(v,b,script[t])
                        rows.append({k:rec[k] for k in ['action','reward','valid','rate_current','rate_replay']} | {'cosine':b.last_cos.copy()})
                    d={k:np.stack([r[k] for r in rows]) for k in rows[0]};group_data[tag]=d
                    if coupling=='closed' and tag=='FF':script=d['action'].copy()
                    for k,values in d.items():records[group+'_'+tag+'__'+k]=values
                    m={}
                    for window in [128,512]:
                        m[f'reward_{window}']=d['reward'][:window].mean(0)
                        m[f'accuracy_{window}']=(d['reward'][:window]>0).sum(0)/np.maximum(1,d['valid'][:window].sum(0))
                    m['gradient_conflict_fraction']=np.nansum(d['cosine']<0,axis=0)/np.maximum(1,np.isfinite(d['cosine']).sum(0))
                    report[group][tag]={k:est(val) for k,val in m.items()}
                    report[group][tag]['individuals']={k:val.tolist() for k,val in m.items()}
                report[group]['contrasts']={}
                for first,last in [('LF','FF'),('FL','FF'),('LL','FF'),('LL','LF')]:
                    q=group_data[first];z=group_data[last]
                    report[group]['contrasts'][first+'-'+last]={f'reward_{window}':est((q['reward'][:window]-z['reward'][:window]).mean(0)) for window in [128,512]}
                print(json.dumps({'diagnostic':group,'early_LL_minus_FF':report[group]['contrasts']['LL-FF']['reward_128']}),flush=True)
    np.savez_compressed(out/'w05_diagnostic_raw.npz',seeds=seeds,**records)
    (out/'w05_diagnostic.json').write_text(json.dumps({'groups':report,'seconds':time.perf_counter()-started,'note':'Exploratory re-use of WUYU_05 formal seeds. Fixed external actions/cues/targets in yoked panel; answer-dependent internal feedback still differs.'},indent=2))
    print('W05_DIAGNOSTIC_COMPLETE',flush=True)
