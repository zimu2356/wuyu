import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib
from pathlib import Path
import numpy as np
from wuyu07.experiment import rollout
from wuyu07.rnn import Network
from wuyu07.state import read_tree
ROOT=Path(__file__).resolve().parent
MODES=['full','no_skill','frozen','fixed_value','plain_rnn','neural_student','neural_student_frozen']

if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True)
    files=[ROOT/'PROTOCOL.md',ROOT/'STUDENT_ADDENDUM.md',ROOT/'REFINEMENT_PROTOCOL.md',ROOT/'EVALUATION_DETAILS.md',ROOT/'ENGINEERING_LOG.md',ROOT/'CONFIG.json',ROOT/'run_eval.py',ROOT/'run_probes.py',*sorted((ROOT/'wuyu07').glob('*.py'))]
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    f=out/'frozen_before_formal.json'
    if f.exists() and json.loads(f.read_text())!=frozen:raise RuntimeError('Formal protocol/source changed; use a separate study directory')
    f.write_text(json.dumps(frozen,indent=2));runs=[]
    for mode in MODES:
        for i,seed in enumerate(cfg['initializations']):
            name=f'{mode}_{i}';cached=out/f'metrics_{name}.json'
            if cached.exists():runs.append(json.loads(cached.read_text()));continue
            net=None
            if mode=='plain_rnn' or mode.startswith('neural_student'):
                family='rnn' if mode=='plain_rnn' else 'student_final'
                info=json.loads((out/f'training_{family}_{i}.json').read_text());net=Network.from_state(read_tree(out/info['selected']['file']))
            points=cfg['checkpoints'] if mode in ('full','neural_student') else [384]
            m,raw,w,a,seconds=rollout(740000+np.arange(cfg['formal_worlds']),seed,mode,cfg['lifetime'],net,out,name,cfg,points)
            item=dict(mode=mode,initialization=i,seconds=seconds,metrics={k:v.tolist() for k,v in m.items()})
            cached.write_text(json.dumps(item,indent=2));runs.append(item)
            print(json.dumps(dict(mode=mode,init=i,seconds=seconds,mean={k:float(v.mean()) for k,v in m.items()})),flush=True)
    (out/'main_runs.json').write_text(json.dumps(runs,indent=2));print('FORMAL_COMPLETE',flush=True)
