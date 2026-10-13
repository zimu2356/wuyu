import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,copy
from pathlib import Path
import numpy as np
from wuyu07.world import World
from wuyu07.student import Student
from wuyu07.rnn import Network
from wuyu07.experiment import rollout
from wuyu07.state import read_tree,write_tree
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';all_runs=[]
    for i,seed in enumerate(cfg['initializations']):
        dest=out/f'training_student_final_{i}.json'
        if dest.exists():all_runs.append(json.loads(dest.read_text()));continue
        prior=json.loads((out/f'training_student_{i}.json').read_text());net=Network.from_state(read_tree(out/prior['selected']['file']))
        selection=copy.deepcopy(prior['validation']);logs=[];start=time.perf_counter();w=None;mix_rng=np.random.default_rng(seed+780007);active=0
        for update in range(512):
            if w is None or w.t>=w.total:
                birth=update//48;seeds=780000+i*1000000+birth*32+np.arange(32);w=World(seeds,1536);a=Student(net,seeds,seed)
            records=[];labels=[]
            for _ in range(32):
                obs=w.observation();r=a.act(obs);target=np.array([m.act(o,explore=False)['action'] for m,o in zip(a.memories,obs)])
                if update<128:r['action']=np.where(mix_rng.random(32)<.3,target,r['action'])
                outc=w.step(r['action']);a.feedback(r,w.observation(),outc['reward']);records.append(r);labels.append(target);active+=int(r['alive'].sum())
            net.update(net.imitation_gradients(records,labels),lr=.001)
            if (update+1)%64==0:
                acc=float(np.mean([np.mean(r['ap'].argmax(1)==y) for r,y in zip(records,labels)]))
                logs.append(dict(update=update+1,accuracy=acc,alive=float(w.alive.mean()),active_examples=active,seconds=time.perf_counter()-start))
                print(json.dumps(dict(refine=i,**logs[-1])),flush=True)
            if (update+1)%256==0:
                m,_,_,_,_=rollout(730000+np.arange(16),seed,'neural_student',net=net)
                filename=f'model_student_refined_{i}_{update+1}.npz';write_tree(out/filename,net.state())
                sel=dict(update=update+1,file=filename,survival=float(m['survival'].mean()),body_area=float(m['body_area'].mean()));selection.append(sel)
                print(json.dumps(dict(refine_validation=i,**sel)),flush=True)
        result=dict(initialization=i,prior_selected=prior['selected'],logs=logs,validation=selection,
            selected=max(selection,key=lambda x:(x['survival'],x['body_area'])),seconds=time.perf_counter()-start,active_examples=active)
        dest.write_text(json.dumps(result,indent=2));all_runs.append(result)
    (out/'student_final_training_summary.json').write_text(json.dumps(all_runs,indent=2));print('REFINEMENT_COMPLETE',flush=True)
