import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json
from pathlib import Path
import numpy as np
from run_probes import student_one,matched_world,trace_header
from wuyu07.experiment import restore,step
from wuyu07.state import write_tree
from analyze import stats
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    out=ROOT/'results';data={};traces=[]
    for init in range(3):
        w,students=restore(out/f'checkpoint_neural_student_{init}_384.npz')
        for j,seed in enumerate(w.seeds):
            individual=student_one(students,j)
            for phase in [0,1]:
                for severity in ['moderate','large']:
                    for mode in ['neural','neural_greedy','planner_same_memory']:
                        key=f'{mode}_phase{phase}_{severity}';data.setdefault(key,[[],[],[]]);row=[]
                        for need in [0,1]:
                            body=[.85,.85] if severity=='moderate' else [.9,.9];body[need]=.65 if severity=='moderate' else .25
                            c=matched_world(w.one(j),int(seed%6),phase,40,body)
                            a=copy.deepcopy(individual) if mode.startswith('neural') else [copy.deepcopy(individual.memories[0])]
                            if not mode.startswith('neural'):a[0].mode='full';a[0].exploration=0
                            trace=trace_header(c,f'{key}_{init}_{j}_{need}');success=0;cost=40
                            for t in range(40):
                                if not c.alive[0]:break
                                if mode=='neural_greedy':
                                    r=a.act(c.observation());r['action']=r['ap'].argmax(1);feedback=c.step(r['action']);a.feedback(r,c.observation(),feedback['reward'])
                                else:r=step(c,a)
                                trace['actions'].append(int(r['action'][0]));trace['products'].append(int(c.last[0]));trace['bodies'].append(c.body[0].tolist())
                                if c.last[0]<2:
                                    success=int(c.last[0]==need);cost=t+1 if success else 40;break
                            trace.update(target_need=need,success=success,cost=cost,budget=40);traces.append(trace);row.append(success)
                        data[key][init].append(row)
        print('diagnostic',init,flush=True)
    arrays={k:np.array(v) for k,v in data.items()};summary={k:stats(v) for k,v in arrays.items()}
    write_tree(out/'student_diagnostic_raw.npz',dict(arrays=arrays,traces=traces))
    (out/'student_diagnostic.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
