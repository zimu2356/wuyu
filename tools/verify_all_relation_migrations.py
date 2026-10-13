"""Exercise both supported Whole 1.1 origins and all five R destinations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wuyu_unified import codec
from wuyu_unified.runtime import Session
from migrate_relation_checkpoint import migrate


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/restore_snapshot_links.py')],check=True)
    subprocess.run([sys.executable,'-B','-m','wuyu_unified','run','--profile','whole',
        '--mode','A','--seed','1770102','--horizon','160','--steps','73','--out',str(out/'old_A')],
        cwd=ROOT/'experiments/relation_start_snapshot',check=True,stdout=subprocess.DEVNULL)
    origins={'A':out/'old_A/state.wuyu','C':ROOT/'experiments/relation_migration/old_C_t1097.wuyu'}
    rows=[]
    for origin,path in origins.items():
        before=codec.load(path)
        for mode in ('R00','R10','R01','R11','RT'):
            s=migrate(path,mode)
            assert codec.digest(before['backend']['native'])==codec.digest(s.backend.state()['native'])
            if origin=='C':
                assert len(s.backend.actor.buffer)==len(before['backend']['actor']['buffer'])==9
                for group in ('p','m','v'):
                    for key,value in before['backend']['actor'][group].items():
                        actual=getattr(s.backend.actor,group)[key]
                        actual=actual[:,:8] if key=='O' else actual[:8] if key=='bo' else actual
                        np.testing.assert_array_equal(actual,value)
                np.testing.assert_array_equal(s.backend.actor.h,before['backend']['actor']['h'])
            target=out/f'{origin}_to_{mode}.wuyu';s.save(target);r=Session.load(target)
            events=s.run(64);other=r.run(64)
            assert len(events)==64
            assert codec.digest(events)==codec.digest(other)
            assert codec.digest(s.state(False))==codec.digest(r.state(False))
            row=dict(origin=origin,destination=mode,passed=True,
                native_state_preserved=True,exact_resume_events=64,
                old_actor_optimizer_and_pending_buffer_preserved=(origin=='C'),
                final_state_sha256=codec.digest(s.state(False)),lineage=s.lineage)
            if origin=='C' and mode=='R00':
                with np.load(ROOT/'experiments/relation_migration/old_C_expected_64.npz') as e:
                    np.testing.assert_array_equal([q['action'] for q in events],e['actions'])
                    np.testing.assert_array_equal([q['after'][14:16] for q in events],e['body'])
                    np.testing.assert_allclose([q['decision']['final_probability'] for q in events],e['probability'],rtol=0,atol=1e-13)
                row['original_1_1_future_actions_and_body_equal']=True
            rows.append(row)
    result=dict(passed=True,migrations=len(rows),rows=rows,
        scope='Exact state preservation and runnable continuation; changes between destination modes are intended and not promised to be behaviorally identical.')
    (out/'MIGRATIONS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,migrations=len(rows))))


if __name__=='__main__':main()
