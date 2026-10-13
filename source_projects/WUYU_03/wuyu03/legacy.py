import copy
import importlib.util
from pathlib import Path
import time
import numpy as np
from .control import Control,encode
from .core import events,Exogenous
from .stats import interval


def world_class(root):
    spec=importlib.util.spec_from_file_location('legacy_world',Path(root)/'inputs/legacy_world_source.py')
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod.World


def run(root,out):
    start=time.perf_counter();World=world_class(root)
    with np.load(Path(root)/'inputs/legacy.npz') as d:data={k:d[k].copy() for k in d.files}
    n=len(data['q']);preferred=np.r_[np.zeros(n//2,int),np.ones(n//2,int)]
    bases={}
    for mode in ['q','replay','dyna']:
        c=Control(n,mode=mode,seed=71303)
        c.q=data['q'].copy()
        for t in range(len(data['action'])):
            c.remember(data['state'][t],data['action'][t],data['reward'][t],data['after'][t])
        # Cache last 512; retain full learned model independently.
        c.bs[:,:512]=data['state'][-512:].T
        c.ba[:,:512]=data['action'][-512:].T
        c.br[:,:512]=data['reward'][-512:].T
        c.bn[:,:512]=data['after'][-512:].T
        c.count=512
        bases[mode]=c
    baseev=events(np.arange(710000,710000+n//2),2056)
    ev=Exogenous(*(np.concatenate([getattr(baseev,k),getattr(baseev,k)],axis=1)
                    for k in ['target','noise','explore','random_action','tie']))
    raw={};summaries={};rows=np.arange(n)
    state_ties=np.random.default_rng(71403).random((25,n,5))
    def reads(c):
        return np.array([c.choose(np.full(n,s),np.ones(n),np.zeros(n,int),state_ties[s],True)
                         for s in range(25)])
    for mode in bases:
        for support in [False,True]:
            for probe in [False,True]:
                tag=f'{mode}__{int(support)}__{int(probe)}'
                c=copy.deepcopy(bases[mode]);w=World(n)
                w.support[rows,preferred]=support
                before=reads(c)
                if probe:
                    for t in range(8):
                        w.charge.fill(0);s=encode(w.charge)
                        a=preferred+1
                        r,cue=w.step(a,ev.target[t],ev.noise[t])
                        c.update(s,a,r,encode(w.charge))
                w.charge.fill(0)
                post=reads(c)
                raw[tag+'__choices_before']=before;raw[tag+'__choices_after']=post
                choices_changed=(before[1:]!=post[1:]).mean(axis=0)
                for phase in ['support_phase','withdrawal']:
                    if phase=='withdrawal':w.support.fill(False)
                    arrays={k:[] for k in ['action','reward','cue','charge','probe']}
                    offset=8 if phase=='support_phase' else 1032
                    for t in range(offset,offset+1024):
                        charge=w.charge.copy();s=encode(charge)
                        a=c.choose(s,ev.explore[t],ev.random_action[t],ev.tie[t])
                        r,cue=w.step(a,ev.target[t],ev.noise[t])
                        c.update(s,a,r,encode(w.charge))
                        for k,v in {'action':a,'reward':r,'cue':cue,'charge':charge,
                                    'probe':(a==preferred+1)&(charge[rows,preferred]==0)}.items():arrays[k].append(v)
                    arrays={k:np.asarray(v) for k,v in arrays.items()}
                    for k,v in arrays.items():raw[tag+'__'+phase+'__'+k]=v
                    maint=(arrays['action'][-256:]==preferred[None,:]+3).mean(axis=0)
                    values={'maintenance':maint,'reward':arrays['reward'][-256:].mean(axis=0),
                            'zero_charge_probes':arrays['probe'].sum(axis=0)}
                    for k,v in values.items():raw[tag+'__'+phase+'__metric_'+k]=v
                    summaries[tag+'__'+phase]={k:interval(v,True) for k,v in values.items()}
                summaries[tag+'__choice_change']=interval(choices_changed,True)
                summaries[tag+'__updates_per_individual']={'real':c.real_updates,'extra':c.extra_updates,
                    'probe_events':8 if probe else 0,'natural_events':2048}
                print('A completed',tag,flush=True)
    def metric(tag):return raw[tag+'__support_phase__metric_maintenance']
    reduction_auto=metric('q__0__0')-metric('q__1__0')
    reduction_probe=metric('q__0__1')-metric('q__1__1')
    comparisons={'q_support_reduction_autonomous':interval(reduction_auto,True),
                 'q_support_reduction_equal_probe':interval(reduction_probe,True),
                 'probe_effect_on_reduction':interval(reduction_probe-reduction_auto,True)}
    for m in ['replay','dyna']:
        for p in [0,1]:
            comparisons[f'{m}_minus_q_maintenance_support_probe{p}']=interval(metric(f'{m}__1__{p}')-metric(f'q__1__{p}'),True)
    np.savez_compressed(Path(out)/'legacy_raw.npz',**raw)
    return {'conditions':summaries,'comparisons':comparisons,'seconds':time.perf_counter()-start,
            'independent_pairs':n//2,'individual_trajectories_per_condition':n}
