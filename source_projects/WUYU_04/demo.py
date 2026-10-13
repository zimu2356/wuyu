"""Continue an actual saved finite-world run, including world RNG and belief."""
import argparse
import json
from pathlib import Path
import numpy as np
from wuyu04.latent import load,save,step


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='results/checkpoint_active_mid.npz')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--save',default=None)
    args=parser.parse_args()
    w,a=load(args.checkpoint);start=w.t;outs=[]
    for _ in range(args.steps):
        try:outs.append(step(w,a))
        except StopIteration:break
    if args.save:save(args.save,w,a)
    result=dict(checkpoint=args.checkpoint,start_event=start,end_event=w.t,mode=a.mode,
                executed_events=len(outs),finite_world_finished=w.t==6656)
    if outs:result.update(accuracy=float(np.mean([o['correct'] for o in outs])),
                         probes=int(np.sum([o['query'] for o in outs])),
                         net_reward=float(np.mean([o['net_reward'] for o in outs])))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
