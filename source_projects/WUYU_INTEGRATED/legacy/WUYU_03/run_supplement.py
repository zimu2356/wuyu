import argparse
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from wuyu03 import core,reservoir
from wuyu03.stats import interval,source_hashes,digest
from run_experiment import PHASES,pack_outputs,metrics,dump,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/supplement');args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise SystemExit('Refusing overwrite')
    out.mkdir(parents=True)
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_hashes(ROOT),
              'supplement_sha256':digest(ROOT/'SUPPLEMENT_01.md'),'seeds':[740000,740031]}
    dump(out/'manifest.json',manifest);start=time.perf_counter()
    seeds=np.arange(740000,740032);ev=core.events(seeds,12288)
    raw={};summary={};costs={}
    for mode in ['ring','reservoir']:
        a=core.Agent(seeds,'replay',.05);w=core.World(seeds)
        if mode=='reservoir':a.memory=reservoir.ReservoirMemory(len(seeds),seed=174303)
        for stage,(name,ctx,reverse) in enumerate(PHASES):
            items=[]
            for t in range(stage*2048,(stage+1)*2048):
                nxt=PHASES[stage+1][1] if t==(stage+1)*2048-1 and stage<5 else ctx
                items.append(core.step(w,a,ev,t,ctx,reverse,nxt))
            data=pack_outputs(items);tag=f'{mode}__{stage}'
            for k,v in data.items():raw[tag+'__'+k]=v
            entry={'tail':{k:interval(v) for k,v in metrics(data,512).items()},
                   'old_A_cache_fraction':interval(a.memory.x[:,:,0,0].mean(axis=1))}
            for ctxt in [0,1]:
                result=core.evaluate(w,a,seeds,ctxt,reverse=(stage==3 and ctxt==0),
                                     offset=1000000+stage*10000+ctxt*1000)
                entry['eval'+str(ctxt)]={k:interval(v) for k,v in result.items()}
                for k,v in result.items():raw[tag+f'__eval{ctxt}__'+k]=v
            summary[tag]=entry
            print('Supplement',mode,name,'A',round(entry['eval0']['accuracy']['mean'],4),
                  'B',round(entry['eval1']['accuracy']['mean'],4),flush=True)
        if mode=='reservoir':reservoir.save(out/'checkpoint_reservoir.npz',w,a)
        else:core.save_checkpoint(out/'checkpoint_ring.npz',w,a)
        costs[mode]={'network_updates':interval(a.net.updates),'gradient_calls_per_individual':24576,
                     'events_per_individual':12288,'q_backups_per_individual':61440}
    def delta(stage,ctx):
        return interval(raw[f'reservoir__{stage}__eval{ctx}__accuracy']-raw[f'ring__{stage}__eval{ctx}__accuracy'])
    comparisons={'retained_A_after_B':delta(1,0),'new_B_after_B':delta(1,1),'reversed_A':delta(3,0)}
    gates={'S1_retention':comparisons['retained_A_after_B']['low']>.03,
           'S2_new_B_cost_within_2pp':comparisons['new_B_after_B']['low']>-.02,
           'S3_reversal_cost_within_2pp':comparisons['reversed_A']['low']>-.02}
    np.savez_compressed(out/'raw.npz',**raw)
    dump(out/'summary.json',{'stages':summary,'comparisons':comparisons,'gates':gates,'costs':costs})
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                    source_unchanged=source_hashes(ROOT)==manifest['source_sha256'])
    dump(out/'manifest.json',manifest)
    print(json.dumps({'comparisons':comparisons,'gates':gates},indent=2),flush=True)


if __name__=='__main__':main()
