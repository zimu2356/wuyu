import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,copy,time,hashlib
import numpy as np
from wuyu06.world import World
from wuyu06.agent import Agent
from wuyu06.network import Network
from wuyu06.state import read_tree
from wuyu06.experiment import metrics,transition
from run_eval import MODES,summarize

ROOT=Path(__file__).resolve().parent

class StressWorld(World):
    def __init__(self,seeds,kind):super().__init__(seeds,1024,True);self.kind=kind
    def observation(self):
        obs=super().observation()
        if self.kind=='paid_observation':
            mask=(self.phase()>0) & ~self.inspect
            cue=np.minimum((self.random[self.t,:,0]*3).astype(int),2);obs[mask,4:7]=np.eye(3)[cue[mask]]
        return obs
    def step(self,action):
        before=self.body.copy();was=self.alive.copy();p=self.phase();out=super().step(action)
        if self.kind=='maintenance_drain':
            self.body[:,0]=np.maximum(0,self.body[:,0]-.006*was*(p>0));self.alive &= self.body[:,0]>.001
            delta=self.body-before
            out.update(reward=np.where(was,delta.sum(1)+.02*self.alive-(was & ~self.alive),0),
                       target=np.column_stack([delta/.36,(out['target'][:,2])]),delta=delta,alive=self.alive.copy(),body=self.body.copy())
        return out


if __name__=='__main__':
    out=ROOT/'results';config=json.loads((ROOT/'TRAINING_CONFIG.json').read_text());start=time.perf_counter();all_results={}
    (out/'stress_frozen.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'SUPPLEMENT_01.md',Path(__file__)]},indent=2))
    for kind in ['paid_observation','maintenance_drain']:
        data={}
        for i in range(3):
            for mode in MODES:
                model_kind='plain_rnn' if mode=='plain_rnn' else 'full';selected=json.loads((out/f'training_{model_kind}_{i}.json').read_text())['selected']['checkpoint']
                net=Network.from_state(read_tree(out/selected));seeds=np.arange(981000,981032);w=StressWorld(seeds,kind);a=Agent(net,seeds,mode);records=[]
                while w.t<w.total:
                    r=transition(w,a);v=r['world'];rows=np.arange(a.n)
                    records.append(dict(alive=r['alive'],action=r['action'].astype(np.int8),body=v['body'],reward=r['reward'],
                      pred=r['pred'].reshape(a.n,7,3)[rows,r['action']],target=r['target'],phase=v['phase'].astype(np.int8),goal_achieved=r['goal_achieved']))
                raw={k:np.stack([r[k] for r in records]) for k in records[0]};m=metrics(raw);name=f'{mode}_{i}';data[name]={k:v.tolist() for k,v in m.items()}
                np.savez_compressed(out/f'stress_{kind}_{name}.npz',seeds=seeds,**raw)
                print(json.dumps({'stress':kind,'method':name,'survival':float(m['survival'].mean())}),flush=True)
        all_results[kind]={'summary':summarize(data,config),'individual_metrics':data}
    all_results['seconds']=time.perf_counter()-start;(out/'stress_summary.json').write_text(json.dumps(all_results,indent=2));print('STRESS_COMPLETE',flush=True)
