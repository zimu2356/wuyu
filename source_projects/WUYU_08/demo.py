"""Resume the predeclared graph/varied initialization-0 actual continuous life."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json
from pathlib import Path
from wuyu08.experiment import restore,neural_step,checkpoint
from wuyu08.world import ACTIONS
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'results/checkpoint_graph_varied_0_384.npz'));p.add_argument('--steps',type=int,default=48);p.add_argument('--save');a=p.parse_args()
    w,individual=restore(a.checkpoint)
    for i in range(a.steps):
        if w.t>=w.total:
            # Extend only after completing the original physical phase schedule.
            w.regime_override=w.phase();w.total=w.t+(a.steps-i)
            print('Beyond the 1536-event boundary: exploratory continuation, not a formal result.')
        r=neural_step(w,individual)
        print(json.dumps(dict(t=w.t,individual=int(w.seeds[0]),action=ACTIONS[int(r['action'][0])],body=w.body[0].round(5).tolist(),position=int(w.pos[0]),product=int(w.last[0]),alive=bool(w.alive[0]),goal=int(r['gp'][0].argmax()),subgoal=int(r['tp'][0].argmax())),ensure_ascii=False))
    if a.save:checkpoint(a.save,w,individual);print('Saved',a.save)
