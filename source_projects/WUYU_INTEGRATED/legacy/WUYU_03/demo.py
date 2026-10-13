"""Continue saved individuals in a fresh stream; never overwrite a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from wuyu03 import core,reservoir


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='results/main/checkpoint_online_stage5.npz')
    parser.add_argument('--context',choices=['A','B'],default='A')
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--individual',type=int,default=0)
    parser.add_argument('--seed',type=int,default=900000)
    parser.add_argument('--save',default=None)
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    if args.save and Path(args.save).exists():parser.error('Refusing to overwrite saved state')
    with np.load(args.checkpoint,allow_pickle=False) as d:meta=json.loads(str(d['meta']))
    is_reservoir='reservoir_rng' in meta
    w,a=(reservoir.load if is_reservoir else core.load_checkpoint)(args.checkpoint)
    if not 0<=args.individual<a.net.n:parser.error('individual out of range')
    ev=core.events(np.arange(args.seed,args.seed+a.net.n),args.steps)
    rows=[]
    for t in range(args.steps):
        x=core.step(w,a,ev,t,int(args.context=='B'),args.reverse,
                    frozen=(a.mode=='frozen'))
        i=args.individual
        rows.append({'event':t,'action':int(x['action'][i]),'cue':int(x['cue'][i]),
                     'answer':int(x['answer'][i]) if x['valid'][i] else None,
                     'probability_of_one':float(x['probability'][i]),
                     'reward':float(x['reward'][i])})
    valid=[r for r in rows if r['answer'] is not None]
    print(json.dumps({'individual':args.individual,'context':args.context,'reverse':args.reverse,
                      'events':len(rows),'valid_answers':len(valid),
                      'accuracy':sum(r['reward']>0 for r in valid)/max(1,len(valid)),
                      'reward':sum(r['reward'] for r in rows)/len(rows),'first_12':rows[:12]},
                     indent=2,ensure_ascii=False))
    if args.save:
        (reservoir.save if is_reservoir else core.save_checkpoint)(args.save,w,a)
        print('Saved continuation:',args.save)


if __name__=='__main__':main()
