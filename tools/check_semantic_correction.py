import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified.runtime import source_locks,sha
from wuyu_unified.workshop_bridge import WorkshopBridge,target as old_target
from wuyu_unified.whole import WholeAgent,target
from wuyu_unified.paths import source_module

def life(job):
    seed,init,case,out=job;out=Path(out)
    old=WorkshopBridge.create(seed,'combined',1536,init)
    new=WholeAgent.create(seed,'A',1536,init)
    new.native.world=source_module('integrated').World.from_state(deepcopy(old.native.world.state()))
    if case=='stable':
        for b in (old,new):b.native.world.regime_override=0;b.native.world.shocks=False
    rows=np.zeros(1536,dtype=[('action_old','i1'),('action_new','i1'),('body_old','f8',(2,)),('body_new','f8',(2,)),
          ('alive_old','?'),('alive_new','?'),('spurious_stage','f8'),('burden_old','f8'),('burden_new','f8')])
    for t in range(1536):
        a=old.step();b=new.step();rows[t]['action_old']=a['action'];rows[t]['action_new']=b['action']
        for name,agent,e in [('old',old,a),('new',new,b)]:
            alive=bool(agent.native.world.alive[0]);body=e['after'][14:16]
            rows[t]['body_'+name]=body;rows[t]['alive_'+name]=alive
            rows[t]['burden_'+name]=float(np.sum((1-body)**2)) if alive else 2.
        rows[t]['spurious_stage']=old_target(a['before'],a['after'])[4]-target(a['before'],a['after'])[4]
    name=f'{seed}_i{init}_{case}';np.savez_compressed(out/(name+'.npz'),trace=rows)
    return dict(seed=seed,initialization=init,case=case,old_survived=bool(rows['alive_old'][-1]),
                new_survived=bool(rows['alive_new'][-1]),old_burden=float(rows['burden_old'].mean()),
                new_burden=float(rows['burden_new'].mean()),changed_actions=int(np.sum(rows['action_old']!=rows['action_new'])),
                spurious_progress_ticks=int(np.sum(rows['spurious_stage']!=0)))

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    seeds=list(range(1300800,1300808));inits=[0,1];cases=['stable','developmental']
    protocol=dict(seeds=seeds,initializations=inits,cases=cases,horizon=1536,sources=source_locks('whole'),
                  driver_sha256=sha(__file__),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  scope='Separate engineering effect of semantic correction. Both predictors corrected together; no action adapter.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with ProcessPoolExecutor(3) as pool:rows=list(pool.map(life,[(s,i,c,str(out)) for s in seeds for i in inits for c in cases]))
    result=dict(lives=len(rows)*2,paired=rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in cases:
        x=[r for r in rows if r['case']==c]
        print(c,json.dumps(dict(pairs=len(x),old_survivors=sum(r['old_survived'] for r in x),new_survivors=sum(r['new_survived'] for r in x),
               old_burden=np.mean([r['old_burden'] for r in x]),new_burden=np.mean([r['new_burden'] for r in x]),
               changed_actions=sum(r['changed_actions'] for r in x),spurious_progress=sum(r['spurious_progress_ticks'] for r in x))))
