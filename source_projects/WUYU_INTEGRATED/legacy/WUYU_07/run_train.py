import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,hashlib
from pathlib import Path
from wuyu07.experiment import train_rnn
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
    files=[ROOT/'PROTOCOL.md',ROOT/'CONFIG.json',ROOT/'run_train.py',*sorted((ROOT/'wuyu07').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'frozen_before_training.json').write_text(json.dumps(hashes,indent=2))
    runs=[]
    for i in range(3):
        f=out/f'training_rnn_{i}.json';runs.append(json.loads(f.read_text()) if f.exists() else train_rnn(cfg,i,out))
    (out/'training_summary.json').write_text(json.dumps(runs,indent=2))
    print('TRAINING_COMPLETE',flush=True)
