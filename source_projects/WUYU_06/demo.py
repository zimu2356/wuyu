"""Resume a real individual batch, printing one representative individual's life."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json
from pathlib import Path
from wuyu06.state import load,save
from wuyu06.experiment import transition
from wuyu06.world import ACTIONS

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(Path(__file__).parent/'results/checkpoint_full_0_128.npz'));p.add_argument('--steps',type=int,default=48);p.add_argument('--save');p.add_argument('--individual',type=int,default=0);args=p.parse_args()
    w,a=load(args.checkpoint);i=args.individual
    for _ in range(min(args.steps,w.total-w.t)):
        r=transition(w,a)
        print(json.dumps({'t':w.t,'alive':bool(w.alive[i]),'position':int(w.pos[i]),'energy':round(float(w.body[i,0]),4),'integrity':round(float(w.body[i,1]),4),
                          'action':ACTIONS[r['action'][i]],'goal':r['goal'][i].round(3).tolist(),'goal_source_tick':int(r['goal_origin'][i]),
                          'predicted_body_change':(r['pred'].reshape(a.n,7,3)[i,r['action'][i],:2]*.36).round(4).tolist()},ensure_ascii=False))
    if args.save:save(args.save,w,a)
