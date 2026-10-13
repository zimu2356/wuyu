"""Post-formal exploratory description of fixed trajectories, no retraining.

Introduced after observing initialization-0 longer-chain failures. Reuses the same
formal worlds; strata are not independent confirmation. Observer truth is used only
to classify knowledge adequacy, never supplied to the acting controller.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
from wuyu08.state import read_tree
from wuyu08.memory import softmax
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    rows=[];example=None
    for i in range(5):
        q=read_tree(ROOT/f'results/probe_graph_varied_{i}_long_chain.npz')
        p=read_tree(ROOT/f'results/probe_graph_varied_{i}_long_chain_planner.npz')
        raw=q['raw'];obs=raw['obs'];st=obs[:,:,6:9].argmax(2);req=obs[:,:,9:12].argmax(2)-1;inv=obs[:,:,12:14]
        held=np.take_along_axis(inv,np.maximum(req,0)[:,:,None],axis=2)[:,:,0]
        blocked=(raw['action']==5)&(st==2)&(req>=0)&(held==0)&raw['alive']
        for j,m in enumerate(q['initial']['memories']):
            known=(m['effect_count']>0)&(m['effect'].argmax(1)==q['start']['products'][j])
            motor=[]
            for site in range(6):
                if m['valid'][site]:button=m['motor'][site].argmax(1)
                else:button=np.array([softmax(m['transition'][site,k])[:,k+1].argmax() for k in range(2)])
                motor.append(bool(np.array_equal(button,q['start']['codes'][j,site])))
            adequate=bool(known.all() and all(motor) and np.array_equal(m['requirement'],q['start']['req_override'][j]))
            failed=not bool(q['metrics']['first_success'][j]);row=dict(initialization=i,world=int(q['start']['seeds'][j]),need=int(q['need'][j]),
                neural_success=bool(q['metrics']['first_success'][j]),planner_success=bool(p['metrics']['first_success'][j]),
                known_product_count=int(known.sum()),correct_motor_sites=int(sum(motor)),all_six_sites_adequate=adequate,
                blocked_take_actions=int(blocked[:,j].sum()),no_body=bool(q['metrics']['no_body'][j]))
            rows.append(row)
            if example is None and adequate and failed and row['planner_success']:
                example=dict(**row,neural_actions=raw['action'][:,j].tolist(),neural_positions=raw['position'][:,j].tolist(),
                    neural_targets=raw['target'][:,j].tolist(),neural_products=raw['product'][:,j].tolist(),
                    planner_actions=p['raw']['action'][:,j].tolist(),planner_positions=p['raw']['position'][:,j].tolist(),planner_products=p['raw']['product'][:,j].tolist(),
                    products=q['start']['products'][j].tolist(),requirements=q['start']['req_override'][j].tolist())
    failures=[r for r in rows if not r['neural_success']];complete=[r for r in rows if r['all_six_sites_adequate']]
    summary=dict(label='Post-formal exploratory analysis of existing traces; not new independent experiments.',total_branches=len(rows),
        failed_branches=len(failures),failed_with_blocked_take=sum(r['blocked_take_actions']>0 for r in failures),
        median_blocked_take_among_failures=float(np.median([r['blocked_take_actions'] for r in failures])),
        adequate_branches=len(complete),adequate_unique_worlds=len(set(r['world'] for r in complete)),
        adequate_neural_success=float(np.mean([r['neural_success'] for r in complete])) if complete else None,
        adequate_planner_success=float(np.mean([r['planner_success'] for r in complete])) if complete else None,
        example=example,rows=rows)
    (ROOT/'results/failure_diagnosis.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','example')},indent=2))
