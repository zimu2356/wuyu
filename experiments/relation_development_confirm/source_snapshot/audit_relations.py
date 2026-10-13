"""Replay every recorded physical step and every observable movement update."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified.relation_world import RelationWorld
from wuyu_unified.relations import MovementRelations
from wuyu_unified.whole import target
from wuyu_unified.runtime import sha
from run_relation_experiment import summarize

def check_life(job):
    directory,name,p=job;root=Path(directory)
    s=json.loads((root/'summaries'/name).read_text());path=root/'traces'/(Path(name).stem+'.npz')
    assert sha(path)==s['trace_sha256']
    with np.load(path,allow_pickle=False) as f:t=f['trace']
    assert len(t)==p['horizon'];assert hashlib.sha256(t.tobytes()).hexdigest()==s['raw_array_sha256']
    world=RelationWorld(s['seed'],p['horizon'],s['case'],p['split'])
    model=MovementRelations(**p['relation_configuration'])
    for i,row in enumerate(t):
        before=world.observation()[0].copy();assert np.array_equal(before,row['before']),(name,i,'before')
        np.testing.assert_array_equal(model.success(i).astype('f4'),row['movement_success'])
        out=world.step(np.array([row['action']]));after=world.observation()[0]
        assert np.array_equal(after,row['after']),(name,i,'after')
        assert np.array_equal(target(before,after),row['target'])
        assert np.array_equal(out['shock'][0],row['shock'])
        assert bool(out['structural_denied'])==bool(row['structural_denied'])
        assert bool(out['movement_noise'])==bool(row['movement_noise'])
        assert bool(world.alive[0])==bool(row['alive'])
        ev=model.update(before,int(row['action']),after,i)
        assert np.array_equal(model.count,row['relation_count'])
        assert row['movement_brier']==(-1. if ev is None else ev['brier'])
        assert abs(float(row['probability'].sum())-1)<2e-6
        assert 0<float(row['prior_tau'])<=1.
    burden=np.sum((1-t['after'][:,14:16])**2,axis=1);burden[~t['alive']]=2.
    assert np.array_equal(burden,t['burden'])
    calculated=summarize(t,world)
    for key,value in calculated.items():
        assert value==s[key],(name,key,value,s[key])
    dead=np.flatnonzero(~t['alive'])
    if len(dead):
        assert not t['alive'][dead[0]:].any()
        assert np.all(t['actor_updates'][dead[0]:]==t['actor_updates'][dead[0]])
    return dict(name=name,steps=len(t),passed=True)

def audit(directory,workers):
    root=Path(directory);p=json.loads((root/'protocol.json').read_text())
    names=sorted(q.name for q in (root/'summaries').glob('*.json'))
    expected={(s,i,m,c) for s in p['seeds'] for i in p['initializations'] for m in p['modes'] for c in p['cases']}
    records=[json.loads((root/'summaries'/n).read_text()) for n in names]
    assert len(records)==len(expected)
    assert {(s['seed'],s['initialization'],s['mode'],s['case']) for s in records}==expected
    with ProcessPoolExecutor(workers) as pool:checks=list(pool.map(check_life,[(str(root),n,p) for n in names]))
    result=dict(passed=True,lives=len(checks),physical_steps=sum(c['steps'] for c in checks),
      checks=['all declared lives','file and raw-array SHA256','all pre/post observations and physical shocks',
              'absorbing death and stopped actor updates','observable movement belief updates and counts',
              'all published life summary fields independently recomputed'])
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory');a.add_argument('--workers',type=int,default=6)
    args=a.parse_args();print(json.dumps(audit(args.directory,args.workers),indent=2))

