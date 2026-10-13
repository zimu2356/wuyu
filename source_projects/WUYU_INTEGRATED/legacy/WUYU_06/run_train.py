import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,time
from wuyu06.experiment import train

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    config=json.loads((ROOT/'TRAINING_CONFIG.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
    files=[ROOT/'PROTOCOL.md',ROOT/'TRAINING_CONFIG.json',ROOT/'run_train.py',*sorted((ROOT/'wuyu06').glob('*.py'))]
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'frozen_before_training.json').write_text(json.dumps(frozen,indent=2));start=time.perf_counter();all_runs=[]
    for mode in ['full','plain_rnn']:
        for initialization in range(3):
            existing=out/f'training_{mode}_{initialization}.json'
            result=json.loads(existing.read_text()) if existing.exists() else train(config,mode,initialization,out)
            all_runs.append(result)
    (out/'training_summary.json').write_text(json.dumps({'runs':all_runs,'seconds':time.perf_counter()-start},indent=2))
    print('TRAINING_COMPLETE',flush=True)
