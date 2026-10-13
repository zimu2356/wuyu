"""Create or continue one saved individual; each output directory is new.

This convenience interface is added after the formal freeze and does not alter
the frozen experiment. It reuses the frozen core and transition functions.
"""
import argparse
import gzip
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import Agent,World,digest
from experiment import execute
from run import sha,verify_freeze,write_json


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--seed',type=int);g.add_argument('--resume',type=Path)
    p.add_argument('--ticks',type=int,default=100);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.ticks<=0:p.error('--ticks must be positive')
    verify_freeze(ROOT/'FORMAL_FREEZE.json');config=json.loads((ROOT/'formal_protocol.json').read_text())
    if args.resume:
        saved=json.loads(args.resume.read_text())
        if saved['config_sha256']!=digest(config) or saved['core_sha256']!=sha(ROOT/'core.py'):raise RuntimeError('Snapshot belongs to a different implementation/configuration')
        w=World.restore(saved['world'],config);a=Agent.restore(saved['agent'],config)
    else:w=World(args.seed,config);a=Agent(args.seed,config)
    args.output.mkdir(parents=True,exist_ok=False)
    with open(args.output/'events.jsonl.gz','wb') as raw:
        with io.TextIOWrapper(gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0),encoding='utf-8') as stream:
            def emit(name,row):stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
            for _ in range(args.ticks):execute(w,a,w.seed,'life',emit)
    write_json(args.output/'state.json',dict(config_sha256=digest(config),core_sha256=sha(ROOT/'core.py'),world=w.snapshot(),agent=a.snapshot()))
    print(json.dumps(dict(seed=w.seed,tick=w.tick,state=str((args.output/'state.json').resolve()))))


if __name__=='__main__':main()
