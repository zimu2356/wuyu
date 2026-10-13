import argparse
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import time

from experiment import run_individual

ROOT=Path(__file__).resolve().parent
STREAMS=('events','checkpoints','interventions','predictions','final_states')


def utc_now():return datetime.now(timezone.utc).isoformat()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def verify_freeze(path):
    freeze=json.loads(Path(path).read_text())
    for name,value in freeze['files'].items():
        if sha(ROOT/name)!=value:raise RuntimeError('Frozen content changed: '+name)
    return freeze


def sources():
    paths=list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths+=[ROOT/name for name in ('development_protocol.json','DEVELOPMENT_FREEZE.json','GIVEN_AND_LEARNED.md')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def csv_write(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(cohort,output):
    verify_freeze(ROOT/'DEVELOPMENT_FREEZE.json')
    if cohort=='holdout':
        verify_freeze(ROOT/'FORMAL_FREEZE.json');config=json.loads((ROOT/'formal_protocol.json').read_text())
        seeds=config['holdout_seeds']
    else:
        config=json.loads((ROOT/'development_protocol.json').read_text());seeds=config['development_seeds']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    before=sources();start=time.perf_counter()
    manifest=dict(status='running',cohort=cohort,config=config,seeds=seeds,source_sha256=before,
                  started_utc=utc_now(),python=platform.python_version())
    write_json(output/'run_manifest.json',manifest)
    files={};handles=[];counts={k:0 for k in STREAMS}
    for name in STREAMS:
        raw=open(output/(name+'.jsonl.gz'),'wb');handles.append(raw)
        files[name]=io.TextIOWrapper(gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=5),encoding='utf-8',newline='\n')
    def emit(name,row):
        files[name].write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');counts[name]+=1
    probes=[];predictions=[];windows=[]
    try:
        for index,seed in enumerate(range(seeds['start'],seeds['start']+seeds['count'])):
            p,d,w=run_individual(seed,config,emit);probes.extend(p);predictions.extend(d);windows.extend(w)
            if (index+1)%8==0:print(f'{cohort}: {index+1}/{seeds["count"]}',flush=True)
    finally:
        for f in files.values():f.close()
        for h in handles:h.close()
    csv_write(output/'branches.csv',probes);csv_write(output/'prediction_probes.csv',predictions);csv_write(output/'life_windows.csv',windows)
    if sources()!=before:raise RuntimeError('Science source changed during run')
    if cohort=='holdout':verify_freeze(ROOT/'FORMAL_FREEZE.json')
    manifest.update(status='complete',completed_utc=utc_now(),elapsed_seconds=time.perf_counter()-start,
                    rows=counts,diagnostic_branches=len(probes),
                    output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.name!='run_manifest.json'})
    write_json(output/'run_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','cohort','diagnostic_branches','rows','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['development','holdout'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cohort,a.output)
