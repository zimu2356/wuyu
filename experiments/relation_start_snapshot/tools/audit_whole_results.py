"""Independent bookkeeping plus exact world replay from recorded actions.

No training is performed in this audit. Every life is checked, including deaths.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified.runtime import sha
from wuyu_unified.whole_world import WholeWorld
from wuyu_unified.whole import target

def check_life(job):
    root,name,split,horizon=job;root=Path(root)
    s=json.loads((root/'summaries'/name).read_text());path=root/'traces'/(Path(name).stem+'.npz')
    if sha(path)!=s['trace_sha256']:raise AssertionError('trace SHA')
    tr=np.load(path)['trace'];assert len(tr)==horizon
    world=WholeWorld(s['seed'],horizon,s['case'],split)
    for i,row in enumerate(tr):
        before=world.observation()[0].copy()
        assert np.array_equal(row['before'],before),(name,i,'before')
        out=world.step(np.array([row['action']]))
        after=world.observation()[0]
        assert np.array_equal(row['after'],after),(name,i,'after')
        assert np.array_equal(row['target'],target(before,after)),(name,i,'target')
        assert np.array_equal(row['shock'],out['shock'][0])
        assert bool(row['alive'])==bool(world.alive[0])
        assert abs(float(row['probability'].sum())-1)<2e-6
    actual=np.sum((1-tr['after'][:,14:16])**2,axis=1);actual[~tr['alive']]=2.
    assert np.array_equal(actual,tr['burden'])
    assert float(actual.mean())==s['total_burden']
    assert float(actual[1024:].mean())==s['challenge_burden']
    assert int(tr['actor_updates'][-1])==s['actor_updates']
    dead=np.flatnonzero(~tr['alive'])
    if len(dead):
        assert not np.any(tr['alive'][dead[0]:]);assert np.all(tr['actor_updates'][dead[0]:]==tr['actor_updates'][dead[0]])
    return dict(name=name,steps=len(tr),passed=True)

def audit(directory,workers=4):
    root=Path(directory);p=json.loads((root/'protocol.json').read_text())
    names=sorted(q.name for q in (root/'summaries').glob('*.json'))
    expected=len(p['seeds'])*len(p['initializations'])*4*len(p['cases'])
    assert len(names)==expected
    tuples=[json.loads((root/'summaries'/n).read_text()) for n in names]
    actual={(r['seed'],r['initialization'],r['mode'],r['case']) for r in tuples}
    declared={(s,i,m,c) for s in p['seeds'] for i in p['initializations'] for m in p['modes'] for c in p['cases']}
    assert actual==declared
    with ProcessPoolExecutor(workers) as pool:
        checks=list(pool.map(check_life,[(str(root),n,p['split'],p['horizon']) for n in names]))
    result=dict(passed=True,lives=len(checks),physical_steps=sum(r['steps'] for r in checks),
                checks=['all declared lives, no exclusions','all raw trajectory SHA256',
                        'every pre/post observation matches independent world replay',
                        'both target body and same-device stage semantics',
                        'physical shock ledger','absorbing deaths and stopped actor updates',
                        'body-loss summaries recomputed from raw observations','valid action distributions'])
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(json.dumps(audit(a.directory,a.workers),indent=2))
