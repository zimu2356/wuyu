"""Reconstruct damaged bytes only when the frozen expected SHA and state match.

This never changes a scientific outcome, summary, seed, configuration or stop
rule. Mismatching deterministic replays are rejected, not substituted.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
from pathlib import Path
import shutil
import sys
from run_whole_experiment import run_life
from wuyu_unified.runtime import sha

if __name__=='__main__':
    root=Path(sys.argv[1]);p=json.loads((root/'protocol.json').read_text())
    out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
    for n in ('traces','summaries','diagnostics','checkpoints','damaged_bytes'):(out/n).mkdir()
    records=[]
    for name in sys.argv[3:]:
        original=json.loads((root/'summaries'/(name+'.json')).read_text());path=root/'traces'/(name+'.npz')
        damaged_sha=sha(path);damaged_size=path.stat().st_size
        if damaged_sha==original['trace_sha256']:raise ValueError('refuse to reconstruct already intact trace')
        run_life((str(out),original['seed'],original['initialization'],original['mode'],original['case'],p))
        regenerated=json.loads((out/'summaries'/(name+'.json')).read_text())
        for key in original:
            if key=='elapsed_seconds':continue
            if original[key]!=regenerated[key]:raise AssertionError('reconstruction mismatch: '+name+' '+key)
        healthy=out/'traces'/(name+'.npz')
        assert sha(healthy)==original['trace_sha256']
        shutil.copy2(path,out/'damaged_bytes'/(name+'.damaged'))
        temp=path.with_suffix('.recovered');shutil.copy2(healthy,temp);os.replace(temp,path)
        records.append(dict(name=name,damaged_size=damaged_size,damaged_sha256=damaged_sha,
                            restored_sha256=sha(path),exact_original_bytes=True,
                            all_scientific_summary_fields_and_final_state_match=True))
    (out/'recovery.json').write_text(json.dumps(dict(scope='Integrity repair only; no new statistical sample.',records=records),indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)
