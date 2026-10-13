import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,subprocess,sys,time,hashlib
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';start=time.perf_counter()
    paths=list((ROOT/'wuyu08').glob('*.py'))+[ROOT/'CONFIG.json',ROOT/'PROTOCOL.md',ROOT/'EVALUATION_DETAILS.md',ROOT/'run_train.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    f=out/'frozen_before_training.json'
    if f.exists():assert json.loads(f.read_text())==hashes
    else:f.write_text(json.dumps(hashes,indent=2))
    jobs=[(cell,i) for i in range(5) for cell in ['flat_regular','flat_varied','graph_regular','graph_varied']]
    def run(job):
        cell,i=job;dest=out/f'training_{cell}_{i}.json'
        if dest.exists():return dict(cell=cell,initialization=i,reused=True)
        with open(out/f'train_{cell}_{i}.log','w') as log:
            result=subprocess.run([sys.executable,'run_train.py','--cell',cell,'--init',str(i)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{cell}_{i} exited {result.returncode}')
        return dict(cell=cell,initialization=i,selected=json.loads(dest.read_text())['selected'])
    with ThreadPoolExecutor(max_workers=cfg['training_workers']) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(dict(**f.result(),wall_seconds=time.perf_counter()-start)),flush=True)
    (out/'training_wall_time.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,workers=cfg['training_workers'])))
    print('TRAINING_COMPLETE',flush=True)
