"""Verify explicit migration against a recorded frozen Whole 1.1 continuation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified import codec
from wuyu_unified.runtime import Session,source_locks
from migrate_relation_checkpoint import migrate

ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'experiments/relation_migration'
    old=directory/'old_C_t1097.wuyu'
    before=codec.load(old)
    checks={}
    for mode in ('R00','R10'):
        session=migrate(old,mode)
        path=directory/f'final_{mode}_t1097.wuyu'
        if path.exists():
            raise FileExistsError(path)
        session.save(path)
        restored=Session.load(path)
        assert codec.digest(session.state())==codec.digest(restored.state())
        assert codec.digest(before['backend']['native'])==codec.digest(session.backend.state()['native'])
        buffer_size=len(session.backend.actor.buffer)
        events=session.run(64)
        resumed=restored.run(64)
        assert codec.digest(events)==codec.digest(resumed)
        entry=dict(checkpoint=path.relative_to(ROOT).as_posix(),source_tick=before['event_index'],
            unfinished_buffer=buffer_size,native_body_inventory_skills_rng_equal=True,
            save_resume_64_events_bit_exact=True,migration=session.lineage)
        if mode=='R00':
            with np.load(directory/'old_C_expected_64.npz') as expected:
                actions=np.array([e['action'] for e in events])
                body=np.array([e['after'][14:16] for e in events])
                probability=np.array([e['decision']['final_probability'] for e in events])
                np.testing.assert_array_equal(actions,expected['actions'])
                np.testing.assert_array_equal(body,expected['body'])
                np.testing.assert_allclose(probability,expected['probability'],rtol=0,atol=1e-13)
            entry['old_1_1_next_64_actions_body_equal']=True
            entry['old_1_1_probability_atol']=1e-13
        checks[mode]=entry
    result=dict(passed=True,checks=checks,current_source_locks=source_locks('whole'),
        historical_demo_note='Earlier new_R00_t1097.wuyu and from_C_R00.wuyu predate the final registry description. Use final_R00/final_R10 for the frozen 1.2 mainline.')
    (directory/'final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
