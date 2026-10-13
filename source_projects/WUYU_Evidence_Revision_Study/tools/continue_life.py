"""Resume a complete world/agent checkpoint without resetting state or RNG."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import World,Agent,digest
from storage import read_gz_json,write_json,sha,source_hashes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--state');ap.add_argument('--seed',type=int)
    ap.add_argument('--case',default='stable');ap.add_argument('--method',default='BE');ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();c=json.loads(Path(args.config).read_text());root=Path(__file__).resolve().parents[1]
    if Path(args.out).exists(): raise FileExistsError(args.out)
    if args.steps<0: raise ValueError('steps must be nonnegative')
    if args.state:
        path=Path(args.state);s=read_gz_json(path) if path.suffix=='.gz' else json.loads(path.read_text())
        if 'config_sha' in s:
            assert s['config_sha']==sha(args.config);assert s['source_hashes']==source_hashes(root)
        else:
            start=json.loads((path.parent/'RUN_START.json').read_text());assert start['config_sha']==sha(args.config);assert start['sources']==source_hashes(root)
        w=World.restore(s.get('world',s.get('final_world')),c);a=Agent.restore(s.get('agent',s.get('final_agent')),c)
    else:
        if args.seed is None: raise ValueError('seed or state required')
        w=World(args.seed,args.case,c);a=Agent(args.seed,c,args.method,w.n_actions)
    events=[]
    for _ in range(args.steps):
        o=w.observation();action,d=a.decide(o);after=w.step(action)
        a.observe(o,action,after,f'continuation:{o["tick"]}',d['pred'][action])
        events.append(dict(before=o,action=action,after=after,prediction=d['pred'][action].tolist()))
    result=dict(config_sha=sha(args.config),source_hashes=source_hashes(root),world=w.snapshot(),agent=a.snapshot(),events=events)
    write_json(args.out,result);print(json.dumps(dict(tick=w.tick,state_sha=digest(dict(world=result['world'],agent=result['agent'])))))

if __name__=='__main__': main()
