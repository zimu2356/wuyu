"""Audit complete integration acceptance outputs and representative physics."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wuyu_unified.runtime import Session,sha
from wuyu_unified.paths import source_module
from wuyu_unified import codec

def main():
    out=ROOT/'validation/integrated_effects'
    freeze=json.loads((out/'FROZEN_BEFORE_RUN.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in freeze['sources'].items())
    assert sha(out/'PROTOCOL.json')==freeze['protocol_sha256']
    rows=json.loads((out/'ALL_METRICS.json').read_text());protocol=json.loads((out/'PROTOCOL.json').read_text())
    expected={(s,c,m) for s in protocol['seeds'] for c in protocol['cases'] for m in protocol['methods']}
    assert expected=={(r['seed'],r['case'],r['method']) for r in rows} and len(rows)==len(expected)
    source_module('integrated');World=importlib.import_module('wuyu_core.world').World
    checked=0;physics=0;checkpoints=0
    for r in rows:
        stem=f"{r['seed']}_{r['case']}_{r['method']}";p=out/'raw'/(stem+'.npy.gz')
        assert sha(p)==r['raw_sha256']
        with gzip.open(p,'rb') as f:events=np.load(f,allow_pickle=False)
        assert len(events)==1536 and np.array_equal(events['tick'],np.arange(1536))
        assert np.all((events['action']>=0)&(events['action']<=6))
        alive=events['alive'];assert np.all(alive[1:]<=alive[:-1])
        computed=np.where(alive,np.sum((1-events['body'])**2,axis=1),2.)
        assert np.array_equal(computed,events['burden'])
        assert float(alive.mean())==r['alive_fraction'] and float(computed.mean())==r['burden']
        assert int((events['product']<4).sum())==r['deliveries']
        seen=np.cumsum(np.r_[True,alive[:-1]])
        if r['method']=='replay':assert np.all(events['evidence_updates']==0)
        else:assert np.array_equal(events['evidence_updates'],seen)
        if r['method'] in ('combined','combined_no_memory','replay'):
            assert np.array_equal(events['neural_updates'],2*seen-1)
            assert np.array_equal(events['replay_updates'],seen-1)
        else:assert np.all(events['neural_updates']==0) and np.all(events['replay_updates']==0)
        if r['method']=='combined_no_memory':assert np.all(events['memory_updates']==0)
        if r['seed']==920000:
            w=World([r['seed']],1536,shocks=r['case']=='developmental')
            if r['case']!='developmental':w.regime_override=0
            for t,e in enumerate(events):
                if r['case']=='recipe_change' and t==768:w.codes[0,int(w.changed[0])]=w.replacement[0]
                after=w.step(np.array([int(e['action'])]))
                assert np.array_equal(after['body'][0],e['body'])
                assert bool(after['alive'][0])==bool(e['alive']) and int(after['product'][0])==int(e['product'])
            physics+=1536
            s=Session.load(out/'checkpoints'/(stem+'.wuyu'));assert s.event_index==1536
            assert codec.digest(s.state(False))==r['final_state_sha256']
            if s.profile=='workshop_bridge':
                for action,model in enumerate(s.backend.models):
                    for sample in model.samples:
                        tick=sample['tick'];assert events[tick]['action']==action
                        assert sample['event_id']==f"bridge:{r['seed']}:{tick}"
            checkpoints+=1
        checked+=len(events)
    result=dict(passed=True,runs=len(rows),raw_ticks_checked=checked,representative_physics_ticks=physics,
                checkpoints_verified=checkpoints,sources_unchanged=True,protocol_unchanged=True,
                all_seeds_conditions_retained=True,body_resets=0,postdeath_revivals=0)
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
