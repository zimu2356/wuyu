"""Run the frozen WUYU_04 protocol; artifacts are written under results/."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from wuyu04 import diagnostic as d
from wuyu04 import latent as l
from wuyu04.common import hashes,interval,write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def diagnostic():
    selection={}; validation_raw={}
    for mode in ('sequential','gated'):
        candidates=[]
        for rate in (.01,.03,.05):
            _,raw,cost=d.run(np.arange(750000,750008),mode,rate,3072,True)
            score=float(raw['evaluations'].mean())
            validation_raw[f'{mode}_{rate}']=raw['evaluations']
            candidates.append(dict(rate=rate,score=score,seconds=cost['seconds']))
            print('validation',mode,rate,round(score,5),flush=True)
        best=max(candidates,key=lambda v:v['score'])
        selection[mode]=dict(chosen_rate=best['rate'],candidates=candidates)
    write_json(OUT/'selection.json',selection)
    np.savez_compressed(OUT/'validation_raw.npz',**validation_raw)
    seeds=np.arange(751000,751032); raws={}; summaries={}
    for mode in ('joint','sequential','dense','gated','repeat','stale','burnin'):
        rate=selection['gated' if mode=='gated' else 'sequential']['chosen_rate']
        agent,raw,cost=d.run(seeds,mode,rate)
        for key,value in raw.items():raws[f'{mode}__{key}']=value
        # Active numeric storage counts net arrays, replay arrays and retained history.
        nbytes=sum(l.array_bytes(net) for net in agent.nets)
        if agent.replay is not None:
            nbytes+=l.array_bytes(agent.replay)+sum(x.nbytes for x in agent.replay.history)
        cost['numeric_bytes_per_individual']=nbytes/len(seeds)
        costs=cost
        summaries[mode]=dict(rate=rate,cost=costs,
            after_a=[interval(x) for x in raw['evaluations'][0]],
            final=[interval(x) for x in raw['evaluations'][-1]],
            context_flip=interval(raw['context_flips']),mapping=[interval(x) for x in raw['mapping']])
        checkpoint={}
        for j,net in enumerate(agent.nets):
            for name,value in vars(net).items():
                if isinstance(value,np.ndarray):checkpoint[f'net{j}__{name}']=value
        # Diagnostic weights only; these are explicitly NOT full runtime checkpoints.
        np.savez_compressed(OUT/f'diagnostic_weights_{mode}.npz',**checkpoint)
        print('diagnostic',mode,'final',[round(v['mean'],4) for v in summaries[mode]['final']],
              'seconds',round(cost['seconds'],2),flush=True)
    counts,evals=d.count_baseline(seeds)
    raws['count__evaluations']=evals;raws['count__counts']=counts
    summaries['count']=dict(after_a=[interval(x) for x in evals[0]],final=[interval(x) for x in evals[-1]],
                            cost=dict(learned_counts=8,numeric_bytes_per_individual=counts.nbytes/len(seeds)))
    diff=raws['burnin__evaluations'][-1]-raws['stale__evaluations'][-1]
    comparisons={'burnin_minus_stale':[interval(x) for x in diff],
                 'gated_minus_sequential':[interval(x) for x in raws['gated__evaluations'][-1]-raws['sequential__evaluations'][-1]]}
    gates={'A1_joint':all(x['low']>.8 for x in summaries['joint']['final']),
           'A2_gated':all(x['low']>.8 for x in summaries['gated']['final']),
           'A3_burnin':comparisons['burnin_minus_stale'][0]['low']>.02 and comparisons['burnin_minus_stale'][1]['low']>-.02}
    np.savez_compressed(OUT/'diagnostic_raw.npz',seeds=seeds,**raws)
    write_json(OUT/'diagnostic.json',dict(methods=summaries,comparisons=comparisons,gates=gates))


def hidden():
    seeds=np.arange(752000,752032); raw_all={}; summaries={}; checkpoints={}
    for mode in l.MODES:
        started=time.perf_counter(); world=l.World(seeds); agent=l.Agent(len(seeds),mode)
        for t in range(2048):l.step(world,agent)
        apprentice_mapping=l.mapping_accuracy(world,agent)
        before_updates=agent.updates.copy();before_values=agent.prediction_values.copy()
        history={}
        for t in range(9*512):
            out=l.step(world,agent)
            for key,value in out.items():history.setdefault(key,[]).append(value)
            if t==3*512+127:
                l.save(OUT/f'checkpoint_{mode}_mid.npz',world,agent)
        history={key:np.array(value) for key,value in history.items()}
        l.save(OUT/f'checkpoint_{mode}_final.npz',world,agent)
        final_mapping=l.mapping_accuracy(world,agent)
        for key,value in history.items():raw_all[f'{mode}__{key}']=value
        raw_all[f'{mode}__apprentice_mapping']=apprentice_mapping
        raw_all[f'{mode}__final_mapping']=final_mapping
        accuracy=history['correct'].reshape(9,512,len(seeds))
        reward=history['net_reward'].reshape(9,512,len(seeds))
        queries=history['query'].reshape(9,512,len(seeds))
        old=accuracy[[2,4,5],:32].mean(axis=(0,1))
        novel=accuracy[3,-128:].mean(axis=0)
        frozen=bool(np.array_equal(agent.w[:,:2],agent.apprentice_bank)) if mode!='single' else False
        s=dict(phase_accuracy=[interval(x) for x in accuracy.mean(axis=1)],
               first32=[interval(x) for x in accuracy[:,:32].mean(axis=1)],
               last128=[interval(x) for x in accuracy[:,-128:].mean(axis=1)],
               old_return_first32=interval(old),new_rule_last128=interval(novel),
               net_reward=interval(reward.mean(axis=(0,1))),
               probe_fraction=interval(queries.mean(axis=(0,1))),
               phase_probe_fraction=[interval(x) for x in queries.mean(axis=1)],
               apprentice_mapping=[interval(x) for x in apprentice_mapping],
               final_mapping=[interval(x) for x in final_mapping],frozen_exact=frozen,
               cost=dict(parameters=int(agent.w[0].size),ordinary_events_per_individual=4608,
                         probes_per_individual=agent.probes.tolist(),
                         update_calls_per_individual=(agent.updates-before_updates).tolist(),
                         sigmoid_scalar_evaluations_per_individual=(agent.prediction_values-before_values).tolist(),
                         agent_numeric_bytes_per_individual=l.array_bytes(agent)/len(seeds),
                         world_numeric_bytes_per_individual=l.array_bytes(world)/len(seeds),
                         seconds=time.perf_counter()-started))
        summaries[mode]=s
        print('hidden',mode,'old',round(s['old_return_first32']['mean'],4),
              'new',round(s['new_rule_last128']['mean'],4),'net',round(s['net_reward']['mean'],4),flush=True)
    ac=raw_all['active__correct'].reshape(9,512,len(seeds))
    pc=raw_all['passive__correct'].reshape(9,512,len(seeds))
    difference=(ac.astype(float)-pc.astype(float))
    switch=interval(difference[[1,2,3,4,5,6],:32].mean(axis=(0,1)))
    netdiff=interval((raw_all['active__net_reward']-raw_all['passive__net_reward']).mean(axis=0))
    gates={}
    for mode in ('passive','active'):
        gates[f'B1_{mode}_old_return']=summaries[mode]['old_return_first32']['low']>.8
        gates[f'B2_{mode}_new_rule']=summaries[mode]['new_rule_last128']['low']>.8 and summaries[mode]['frozen_exact']
    gates['B3_active_switch']=switch['low']>.02
    gates['B3_active_net']=netdiff['low']>0
    # Complete runtime continuation from a checkpoint in first C exposure.
    wa,aa=l.load(OUT/'checkpoint_active_mid.npz');wb,ab=l.load(OUT/'checkpoint_active_mid.npz')
    # Compare restored run with saved original raw continuation, not just two loads.
    start=3*512+128
    for i in range(128):
        out=l.step(wa,aa)
        for key,val in out.items():np.testing.assert_array_equal(val,raw_all['active__'+key][start+i])
    # Then compare a further save/load to the live branch, including full weights/RNG.
    l.save(OUT/'checkpoint_restore_audit.npz',wa,aa)
    wb,ab=l.load(OUT/'checkpoint_restore_audit.npz')
    for i in range(128):
        one=l.step(wa,aa);two=l.step(wb,ab)
        for key in one:np.testing.assert_array_equal(one[key],two[key])
    for one,two in ((wa,wb),(aa,ab)):
        for key,val in vars(one).items():
            if isinstance(val,np.ndarray):np.testing.assert_array_equal(val,getattr(two,key))
    assert wa.event_rng.bit_generator.state==wb.event_rng.bit_generator.state
    assert wa.probe_rng.bit_generator.state==wb.probe_rng.bit_generator.state
    gates['exact_runtime_restore']=True
    np.savez_compressed(OUT/'hidden_raw.npz',seeds=seeds,**raw_all)
    write_json(OUT/'hidden.json',dict(methods=summaries,comparisons=dict(active_minus_passive_switch_first32=switch,
                          active_minus_passive_net=netdiff),gates=gates))


def main():
    OUT.mkdir(exist_ok=True);initial=hashes(ROOT)
    write_json(OUT/'frozen_sources.json',initial)
    started=time.perf_counter()
    diagnostic(); hidden()
    final=hashes(ROOT)
    unchanged=all(final.get(k)==v for k,v in initial.items())
    manifest=dict(seconds=time.perf_counter()-started,python=platform.python_version(),numpy=np.__version__,
                  initial_source_hashes=initial,final_source_hashes=final,initial_sources_unchanged=unchanged)
    write_json(OUT/'manifest.json',manifest)
    if not unchanged:raise RuntimeError('an initially frozen source changed during experiment')
    print('complete',round(manifest['seconds'],2),'seconds; source hashes unchanged',unchanged,flush=True)


if __name__=='__main__':main()
