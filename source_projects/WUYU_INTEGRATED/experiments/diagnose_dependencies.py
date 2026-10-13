"""Exploratory dependency diagnostic; identical observed apprenticeship history.

The offline teacher generates a real 384-step history in the world. Runtime
receives only observations/actions/outcomes, never codes, products or labels.
This tests what subjects can do after supplied experience, not autonomous birth.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wuyu_core.legacy import module
from wuyu_core.runtime import Session,Individual,load_network
from wuyu_core.world import World
from wuyu_core.state import save_raw


def main():
    offline=module(8);out=ROOT/'results';rows=[]
    for init in range(3):
        seeds=np.arange(940000,940008);history=offline.common_history(seeds,801+init)
        net=load_network(init)
        for j,seed in enumerate(seeds):
            base=Individual(net,int(seed),init,'replay')
            for t,obs in enumerate(history['obs']):
                base.decide(obs[j:j+1],forced=int(history['action'][t,j]))
                nxt=history['obs'][t+1] if t+1<len(history['obs']) else history['last_obs']
                base.feedback(nxt[j:j+1])
            world=World.from_state(copy.deepcopy(history['world'])).one(j)
            for condition in ('independent','dependency','long_chain'):
                for need in (0,1):
                    for mode in ('prior','replay'):
                        w=World.from_state(copy.deepcopy(world.state()));a=Individual.from_state(base.state());a.mode=mode
                        w.shocks=False;w.training_varied=False;w.regime_override=0 if condition=='independent' else 1
                        if condition=='long_chain':
                            req=w.requirements();req[w.products<2]=1;req[w.products==3]=0;req[w.products==2]=-1;w.req_override=req
                        w.inventory[:]=0;w.stage[:]=0;w.last[:]=4;a.memory.learn_enabled=False
                        s=Session(w,a)
                        for _ in range(6):s.step(greedy=True,forced=1)
                        w.body[:]=[[.25,.9] if need==0 else [.9,.25]];w.alive[:]=True
                        records=[]
                        for _ in range(40):
                            r=s.step(greedy=True);o=r['obs'][0];req=int(o[9:12].argmax())-1
                            blocked=r['action']==5 and o[6:9].argmax()==2 and req>=0 and o[12+req]==0
                            records.append(dict(action=r['action'],product=int(r['outcome']['product'][0]),blocked=blocked,body=r['outcome']['body'][0]))
                        first=next((r['product'] for r in records if r['product']<2),-1)
                        rows.append(dict(init=init,seed=int(seed),condition=condition,need=need,mode=mode,first_success=first==need,blocked=int(sum(r['blocked'] for r in records)),first_product=first))
                        save_raw(out/f'probe_{init}_{seed}_{condition}_{need}_{mode}.npz',**{k:np.array([r[k] for r in records]) for k in records[0]})
    summary={}
    for mode in ('prior','replay'):
        summary[mode]={}
        for condition in ('independent','dependency','long_chain'):
            selected=[r for r in rows if r['mode']==mode and r['condition']==condition]
            summary[mode][condition]=dict(n=len(selected),first_success=float(np.mean([r['first_success'] for r in selected])),blocked=float(np.mean([r['blocked'] for r in selected])))
    result=dict(label='Exploratory; 8 new worlds x 3 fixed models x 2 body interventions; teacher-generated prehistory; memory frozen during probes, predictor adapts in replay condition',summary=summary,records=rows)
    (out/'dependency_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
