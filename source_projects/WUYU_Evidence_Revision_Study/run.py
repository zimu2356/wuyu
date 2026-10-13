"""Run each seed as a paired block; completed block hashes survive interruptions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse, concurrent.futures, json, platform, time
from pathlib import Path
import numpy as np
from experiment import run_life
from storage import write_json, source_hashes, sha

def block(task):
    seed,c,out,development=task;out=Path(out);results=[]
    for case in c['cases']:
        for method in c['methods']: results.append(run_life(seed,case,method,c,out,development))
    receipt=dict(seed=seed,results=results)
    write_json(out/f'block_{seed}.json',receipt)
    return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--start',type=int,required=True);ap.add_argument('--count',type=int,required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--development',action='store_true');args=ap.parse_args()
    c=json.loads(Path(args.config).read_text());root=Path(__file__).parent;before=source_hashes(root)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'RUN_START.json',dict(config=c,config_sha=sha(args.config),sources=before,start=args.start,count=args.count,
               development=args.development,python=platform.python_version(),numpy=np.__version__,workers=args.workers))
    t=time.monotonic();receipts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(s,c,str(out),args.development) for s in range(args.start,args.start+args.count)]
        for receipt in pool.map(block,tasks):
            receipts.append(receipt);print(json.dumps(dict(completed=len(receipts),total=args.count,seed=receipt['seed'],seconds=round(time.monotonic()-t,1))),flush=True)
    assert before==source_hashes(root),'Scientific source changed during run'
    manifest={};summaries=[]
    for b in receipts:
        for r in b['results']: manifest.update(r['files']);summaries.append(r['summary'])
        manifest[f'block_{b["seed"]}.json']=sha(out/f'block_{b["seed"]}.json')
    write_json(out/'summaries.json',summaries);manifest['summaries.json']=sha(out/'summaries.json')
    write_json(out/'MANIFEST.json',manifest)
    write_json(out/'COMPLETE.json',dict(seeds=args.count,lives=len(summaries),steps=len(summaries)*c['life_ticks'],
              elapsed_seconds=time.monotonic()-t,manifest_sha=sha(out/'MANIFEST.json'),sources_unchanged=True))

if __name__=='__main__': main()
