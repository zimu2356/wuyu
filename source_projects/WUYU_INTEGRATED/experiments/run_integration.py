"""Reproduce the fixed protocol. CPU only; no external service calls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wuyu_core.runtime import Session,Individual,load_network,MODES
from wuyu_core.world import World
from wuyu_core.state import save_raw


def run_initialization(init):
    out=ROOT/'results';out.mkdir(exist_ok=True);start=time.perf_counter()
    net=load_network(init);metrics=[]
    for mode in MODES:
        for seed in range(910000,910016):
            s=Session(World([seed],768,shocks=True),Individual(net,seed,init,mode))
            records=[]
            for _ in range(768):
                q=s.step();z=q['outcome'];o=q['obs'][0]
                req=int(o[9:12].argmax())-1
                blocked=q['action']==5 and o[6:9].argmax()==2 and req>=0 and o[12+req]==0 and q['alive']
                records.append(dict(action=q['action'],alive=q['alive'],body=z['body'][0],product=int(z['product'][0]),
                                    blocked=blocked,prediction=q['prediction'],target=q['target'],mse=q['prediction_mse'],
                                    probability=q['probability'],shock=z['shock'][0],bias=q['bias'],bias_applied=q['bias_applied']))
            raw={k:np.array([r[k] for r in records]) for k in records[0]}
            save_raw(out/f'life_{init}_{seed}_{mode}.npz',**raw)
            alive=raw['alive'];steps=max(1,int(alive.sum()))
            row=dict(initialization=init,seed=seed,mode=mode,survival=float(alive.mean()),
                     final_alive=bool(s.world.alive[0]),deliveries=int((raw['product']<4).sum()),
                     body_area=float((raw['body'].sum(1)*alive).mean()),blocked=int(raw['blocked'].sum()),
                     prediction_mse=float((raw['mse']*alive).sum()/steps),updates=s.individual.predictor.updates,
                     replay_updates=s.individual.predictor.replay_updates)
            metrics.append(row)
            if seed==910000:s.save(out/f'individual_{init}_{mode}_768.npz')
        print(json.dumps(dict(initialization=init,mode=mode,completed=16,seconds=round(time.perf_counter()-start,1))),flush=True)
    (out/f'metrics_{init}.json').write_text(json.dumps(metrics,indent=2))
    return dict(initialization=init,seconds=time.perf_counter()-start,runs=len(metrics))


def analyse():
    out=ROOT/'results';rows=[]
    for i in range(3):rows+=json.loads((out/f'metrics_{i}.json').read_text())
    lookup={(r['mode'],r['initialization'],r['seed']):r for r in rows}
    summary=dict(protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
                 worlds=16,trained_initializations=3,events_per_run=768,total_runs=len(rows),
                 means={},per_initialization={},paired={},shadow_actions_identical=True)
    keys=['survival','final_alive','deliveries','body_area','blocked','prediction_mse','updates','replay_updates']
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode]
        summary['means'][mode]={k:float(np.mean([r[k] for r in selected])) for k in keys}
        summary['per_initialization'][mode]=[{k:float(np.mean([r[k] for r in selected if r['initialization']==i])) for k in keys} for i in range(3)]
    rng=np.random.default_rng(600133);draws=rng.integers(0,16,(2000,16))
    for mode in MODES[1:]:
        result={}
        for key in ('survival','deliveries','body_area','blocked'):
            delta=np.array([np.mean([lookup[mode,i,seed][key]-lookup['prior',i,seed][key] for i in range(3)]) for seed in range(910000,910016)])
            ci=np.quantile(delta[draws].mean(1),[.025,.975])
            result[key]=dict(mean=float(delta.mean()),ci95=ci.tolist(),scope='world-cluster paired bootstrap; conditional on three trained models')
        summary['paired'][mode]=result
    for i in range(3):
        for seed in range(910000,910016):
            with np.load(out/f'life_{i}_{seed}_prior.npz') as a,np.load(out/f'life_{i}_{seed}_shadow.npz') as b:
                if not np.array_equal(a['action'],b['action']):summary['shadow_actions_identical']=False
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--analyse-only',action='store_true');a=p.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if not a.analyse_only:
        # Capture protocol and actual implementation before any formal observation.
        files=[ROOT/'PROTOCOL.md',Path(__file__),*sorted((ROOT/'wuyu_core').glob('*.py'))]
        freeze={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        (out/'frozen_before_formal.json').write_text(json.dumps(freeze,indent=2))
        start=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:timing=list(pool.map(run_initialization,range(3)))
        (out/'timing.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-start,workers=a.workers,runs=timing),indent=2))
    result=analyse();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
