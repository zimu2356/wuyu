"""Post-hoc falsification: can a past-only action mean beat the new predictor?"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=ROOT/'results';records=[]
    for i in range(3):
        for seed in range(910000,910016):
            with np.load(p/f'life_{i}_{seed}_prior.npz') as a,np.load(p/f'life_{i}_{seed}_shadow.npz') as b:
                mean=np.zeros((7,6));count=np.zeros(7);err=[];changes={}
                for act,y,alive in zip(a['action'],a['target'],a['alive']):
                    if not alive:continue
                    err.append(np.mean((mean[act]-y)**2));count[act]+=1;mean[act]+=(y-mean[act])/count[act]
                for mode in ('online','replay'):
                    with np.load(p/f'life_{i}_{seed}_{mode}.npz') as c:changes[mode]=int(np.sum(c['action']!=a['action']))
                records.append(dict(init=i,seed=seed,action_mean_mse=float(np.mean(err)),shadow_mse=float(b['mse'][b['alive']].mean()),zero_mse=float(np.mean(a['target'][a['alive']]**2)),changed_actions=changes))
    result=dict(label='Post-hoc diagnostic on identical prior/shadow histories; not an additional prespecified success criterion',mean={k:float(np.mean([r[k] for r in records])) for k in ('action_mean_mse','shadow_mse','zero_mse')},changed_actions={k:int(sum(r['changed_actions'][k] for r in records)) for k in ('online','replay')},records=records)
    (p/'prediction_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['mean']))


if __name__=='__main__':main()
