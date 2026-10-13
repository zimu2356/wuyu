"""Resume the delivered developing agent and run a finite continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
from wuyu05.core import load,step,save

parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=128)
args=parser.parse_args();root=Path(__file__).resolve().parent
w,a=load(root/'results/checkpoint_mid.npz');start=w.t;records=[]
for _ in range(min(args.steps,w.total-w.t)):records.append(step(w,a))
save(root/'results/demo_checkpoint.npz',w,a)
result={'start':start,'end':w.t,'events':len(records),'individuals':a.n,
        'mean_reward':float(np.mean([r['reward'] for r in records])),
        'mean_current_rate':float(np.mean([r['rate_current'] for r in records])),
        'mean_replay_rate':float(np.mean([r['rate_replay'] for r in records])),
        'note':'Finite continuation of the saved experiment, not a background daemon.'}
(root/'results/demo.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
