"""Independent raw-array summaries + world-accounting and runtime audits."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import numpy as np
from wuyu05.core import load,save,step

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
MODES=('fixed','heuristic','learned','matched_mean','shuffled','frozen')
KINDS=('return','change','noise')


def interval(a):
    a=np.asarray(a,float);rng=np.random.default_rng(85005)
    samples=a[rng.integers(0,len(a),(4000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),'n':len(a)}


def window(data,kind,selection):
    reward=data[kind+'__reward'];valid=data[kind+'__valid'];phase=data[kind+'__phase'];time=np.arange(len(reward))[:,None]
    d1=(phase==0).sum(axis=0)
    if selection=='first':mask=time<128
    elif selection=='late':mask=(time>=d1-128)&(time<d1)
    elif selection=='recovery':mask=time>=len(reward)-128
    elif selection=='all':mask=np.ones_like(valid)
    else:raise ValueError(selection)
    mask=np.broadcast_to(mask,valid.shape)
    return {'reward':(reward*mask).sum(axis=0)/mask.sum(axis=0),
            'accuracy':((reward>0)&mask).sum(axis=0)/np.maximum(1,(valid&mask).sum(axis=0))}


def verify_world(data,kind,start_charge):
    seeds=data['seeds'];prefix='prefix' if kind=='prefix' else kind
    actions=data[prefix+'__action'];length,n=actions.shape;charge=start_charge.copy()
    target=[];noise=[];d1=[];d2=[];polarity=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed)+(8100000 if kind=='prefix' else 8200000))
        target.append(rng.integers(0,2,length,dtype=np.int8));noise.append(rng.random((length,3)))
        sched=np.random.default_rng(int(seed)+8300000)
        d1.append(sched.integers(512,769));d2.append(0 if kind=='prefix' else sched.integers(256,385))
        polarity.append(np.random.default_rng(int(seed)+300000).integers(0,2))
    target=np.stack(target,axis=1);noise=np.stack(noise,axis=1);d1=np.array(d1);d2=np.array(d2);polarity=np.array(polarity)
    rows=np.arange(n)
    for t,act in enumerate(actions):
        assert np.all((act>=0)&(act<5))
        phase=(t>=d1).astype(int) if kind=='prefix' else np.where(t<d1,0,np.where(t<d1+d2,1,2))
        ctx=phase if kind=='prefix' else (phase==1).astype(int)
        valid=act==0
        for j in range(2):
            used=(act==j+1)&(charge[:,j]>0);valid|=used;charge[used,j]-=1;charge[act==j+3,j]=4
        reverse=(phase==0)&(kind=='change');degraded=(phase==0)&(kind=='noise')
        reliability=np.where(degraded,.6,np.array([.6,.9,.9])[np.clip(act,0,2)])
        cue=np.where(valid,target[t]^polarity^ctx^reverse^(noise[t,rows,np.clip(act,0,2)]>=reliability),-1)
        answer=data[prefix+'__answer'][t]
        reward=np.where(valid,np.where(answer==target[t],1.,-1.),-.2)
        for key,expected in [('phase',phase),('context',ctx),('valid',valid),('cue',cue),('reward',reward)]:
            np.testing.assert_array_equal(data[prefix+'__'+key][t],expected)
    return charge


def verify_features(data,kind,state,theta):
    fast,slow,prev=(x.copy() for x in state)
    p=data[kind+'__probability'];ans=data[kind+'__answer'];reward=data[kind+'__reward'];valid=data[kind+'__valid'];features=data[kind+'__features']
    labels=np.where(reward>0,ans,1-ans)
    for t in range(len(p)):
        err=np.where(valid[t],abs(p[t]-labels[t]),prev);pc=np.clip(p[t],1e-12,1-1e-12)
        entropy=-(pc*np.log(pc)+(1-pc)*np.log(1-pc))/np.log(2)
        expected=np.stack([np.ones(len(fast)),2*err-1,2*fast-1,2*slow-1,4*(fast-slow),2*entropy-1,2*(err-prev)],axis=1)
        np.testing.assert_allclose(features[t][:,[0,1,2,3,4,5,7]],np.clip(expected,-1,1),rtol=0,atol=1e-14)
        z=features[t]@theta.T;rates=.12/(1+np.exp(-np.clip(z,-30,30)))
        np.testing.assert_allclose(data[kind+'__rate_current'][t],rates[:,0],rtol=0,atol=1e-14)
        np.testing.assert_allclose(data[kind+'__rate_replay'][t],rates[:,1],rtol=0,atol=1e-14)
        fast=np.where(valid[t],fast+(err-fast)/8,fast);slow=np.where(valid[t],slow+(err-slow)/64,slow);prev=np.where(valid[t],err,prev)
    return fast,slow,prev


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['status']=='completed'
    selection=json.loads((OUT/'selection.json').read_text());raw={};summary={'methods':{},'comparisons':{},'gates':{}}
    for mode in MODES:
        with np.load(OUT/('raw_'+mode+'.npz'),allow_pickle=False) as f:raw[mode]={k:f[k] for k in f.files}
        d=raw[mode];n=len(d['seeds']);charge=verify_world(d,'prefix',np.zeros((n,2),dtype=int))
        scores=[];branches={}
        for kind in KINDS:
            verify_world(d,kind,charge)
            scores.append(d[kind+'__reward'].mean(axis=0))
            branches[kind]={where:{key:interval(values) for key,values in window(d,kind,where).items()} for where in ('first','late','recovery','all')}
            for name in ('rate_current','rate_replay'):
                assert np.all((d[kind+'__'+name]>=0)&(d[kind+'__'+name]<=.12))
            if mode=='frozen':assert np.all(d[kind+'__rate_current']==0) and np.all(d[kind+'__rate_replay']==0)
        np.testing.assert_allclose(d['branch_rewards'],np.stack(scores),rtol=0,atol=1e-13)
        fitness=np.mean(scores,axis=0);np.testing.assert_allclose(fitness,d['fitness'],rtol=0,atol=1e-13)
        summary['methods'][mode]={'reward':interval(fitness),'branches':branches,
             'old_b':interval(d['final_b'].mean(axis=0)),'prefix_b':interval(d['prefix_b']),
             'mean_current_rate':float(np.concatenate([d[k+'__rate_current'][d[k+'__valid']] for k in KINDS]).mean()),
             'mean_replay_rate':float(np.concatenate([d[k+'__rate_replay'][d[k+'__replay_valid']] for k in KINDS]).mean()),
             'valid_observations_mean':float(sum(d[k+'__valid'].sum(axis=0) for k in KINDS).mean()),
             'action_counts_mean':[float(sum((d[k+'__action']==j).sum(axis=0) for k in KINDS).mean()) for j in range(5)]}
    learned=raw['learned'];fixed=raw['fixed']
    for mode in MODES:
        if mode!='learned':summary['comparisons']['learned_minus_'+mode]=interval(learned['fitness']-raw[mode]['fitness'])
    branch_diff={k:interval(learned[k+'__reward'].mean(axis=0)-fixed[k+'__reward'].mean(axis=0)) for k in KINDS}
    old_diff=interval((learned['final_b']-fixed['final_b']).mean(axis=0))
    summary['comparisons']['branches_learned_minus_fixed']=branch_diff
    summary['comparisons']['old_b_learned_minus_fixed']=old_diff
    summary['gates']={'G1_net_reward':summary['comparisons']['learned_minus_fixed']['low']>0,
        'G2_each_branch':all(v['low']>-.02 for v in branch_diff.values()),'G3_old_b':old_diff['low']>-.03,
        'G4_history_correspondence':all(summary['comparisons']['learned_minus_'+m]['low']>0 for m in ('matched_mean','shuffled'))}
    state=tuple(np.full(32,.5) for _ in range(3));theta=np.array(selection['learned']['theta'])
    state=verify_features(learned,'prefix',state,theta)
    for kind in KINDS:verify_features(learned,kind,state,theta)
    world,agent=load(OUT/'checkpoint_mid.npz');start=world.t;assert start==400
    for t in range(start,start+128):
        rec=step(world,agent)
        for key,value in rec.items():np.testing.assert_array_equal(value,learned['change__'+key][t])
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'roundtrip.npz';save(p,world,agent);w,a=load(p)
        for t in range(start+128,start+256):
            first,second=step(world,agent),step(w,a)
            for key,value in first.items():
                np.testing.assert_array_equal(value,second[key]);np.testing.assert_array_equal(value,learned['change__'+key][t])
        p=Path(directory)/'first.npz';q=Path(directory)/'second.npz';save(p,world,agent);save(q,w,a)
        with np.load(p,allow_pickle=False) as first,np.load(q,allow_pickle=False) as second:
            assert first.files==second.files
            for key in first.files:np.testing.assert_array_equal(first[key],second[key])
    while world.t<world.total:step(world,agent)
    with np.load(OUT/'weights_learned_change.npz',allow_pickle=False) as data:
        for key in data.files:np.testing.assert_array_equal(getattr(agent.net,key),data[key])
    frozen=json.loads((OUT/'frozen_sources.json').read_text())
    for path,expected in frozen.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
    audit={'raw_world_accounting':True,'reward_metrics_independently_recomputed':True,
           'past_statistics_and_learned_rates_recomputed':True,'restored_raw_events_exact':256,
           'full_serialized_roundtrip_state_equal':True,'restored_final_network_equal':True,
           'frozen_numerical_sources_unchanged':True,'existing_mechanism_tests_passed':8}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'comparisons':summary['comparisons'],'gates':summary['gates'],'audit':audit},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
