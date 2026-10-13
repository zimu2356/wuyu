"""Post-hoc trajectory comparisons; does not alter acceptance definitions."""
from pathlib import Path
import gzip
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'validation/integrated_effects';d=json.loads((out/'SUMMARY.json').read_text())
    rows=json.loads((out/'ALL_METRICS.json').read_text())
    result=dict(scope='post-hoc mechanism diagnosis, all saved lives, no tuning',matching_actions={},developmental_deaths={},combined_update_means={})
    for case in ('stationary','recipe_change','developmental'):
        for a,b in [('online','update'),('evidence','revision')]:
            matches=[];different=0
            for seed in range(920000,920032):
                with gzip.open(out/'raw'/f'{seed}_{case}_{a}.npy.gz','rb') as f:x=np.load(f,allow_pickle=False)
                with gzip.open(out/'raw'/f'{seed}_{case}_{b}.npy.gz','rb') as f:y=np.load(f,allow_pickle=False)
                n=int((x['action']!=y['action']).sum());matches.append(n==0);different+=n
            result['matching_actions'][case+' '+a+' vs '+b]=dict(exact_lives=sum(matches),lives=32,different_ticks=different)
        result['combined_update_means'][case]={k:d['means'][case]['combined'][k] for k in ('evidence_updates','neural_updates','replay_updates','memory_updates','choice_changes','revisions')}
    for mode in ('prior','online','combined'):
        result['developmental_deaths'][mode]=[r['seed'] for r in rows if r['method']==mode and r['case']=='developmental' and not r['final_alive']]
    (out/'POSTHOC_MECHANISM_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'comparisons':len(result['matching_actions'])}))
if __name__=='__main__':main()
