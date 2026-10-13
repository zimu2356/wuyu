import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
from wuyu07.student import train_student
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True);runs=[]
    for i in range(3):
        f=out/f'training_student_{i}.json';runs.append(json.loads(f.read_text()) if f.exists() else train_student(cfg,i,out))
    (out/'student_training_summary.json').write_text(json.dumps(runs,indent=2));print('STUDENT_COMPLETE',flush=True)
