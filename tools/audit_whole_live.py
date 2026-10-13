"""Read-only physics audit alongside a finite experiment; no feedback to agents."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from audit_whole_results import check_life

if __name__=='__main__':
    root=Path(sys.argv[1]);p=json.loads((root/'protocol.json').read_text())
    expected=len(p['seeds'])*len(p['initializations'])*len(p['modes'])*len(p['cases'])
    checks=root/'audit_checks';checks.mkdir(exist_ok=False);queued=set();complete=[]
    with ProcessPoolExecutor(2) as pool:
        futures={}
        while len(complete)<expected:
            for f in sorted((root/'summaries').glob('*.json')):
                if f.name in queued:continue
                try:json.loads(f.read_text())
                except json.JSONDecodeError:continue
                queued.add(f.name)
                futures[pool.submit(check_life,(str(root),f.name,p['split'],p['horizon']))]=f.name
            for future in list(futures):
                if not future.done():continue
                value=future.result();complete.append(value)
                (checks/futures.pop(future)).write_text(json.dumps(value)+'\n')
            if len(complete)%32==0:print(json.dumps(dict(audited=len(complete),expected=expected)),flush=True)
            if len(complete)<expected:time.sleep(3)
    assert len(queued)==expected
    result=dict(passed=True,lives=len(complete),physical_steps=sum(x['steps'] for x in complete),
                checks=['all declared lives, no exclusions','all raw trajectory SHA256',
                        'every pre/post observation matches independent world replay',
                        'both target body and same-device stage semantics','physical shock ledger',
                        'absorbing deaths and stopped actor updates','body-loss summaries recomputed from raw observations',
                        'valid action distributions'],execution='read-only concurrent audit; no agent feedback')
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
