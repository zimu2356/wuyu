"""Reconstruct lost bytes with frozen code; preserve every surviving result.

This is evidence recovery, not new independent experimental replication.
No protocol, seed, condition, or inherited model is changed. Existing summary
values, raw-array hashes, compressed trace hashes and final-state hashes must
match exactly; wall-clock duration is the sole allowed difference.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)

def initialize(runtime):
    sys.path.insert(0,str(Path(runtime)/'tools'))
    sys.path.insert(0,str(runtime))

def recover(job):
    from run_relation_experiment import run_life
    output,original,seed,init,mode,case,p,was_original,prior_outputs=job
    output=Path(output);original=Path(original)
    label=f'{seed}_i{init}_{case}_{mode}'
    existing=original/'summaries'/(label+'.json')
    previous=json.loads(existing.read_text()) if existing.exists() else None
    regenerated=None
    required_files=[original/'traces'/(label+'.npz')]
    if seed==p['seeds'][0]:
        required_files.extend(original/'checkpoints'/(label+suffix+'.wuyu') for suffix in ('_t1024','_final'))
    for diagnostic in (previous or {}).get('diagnostics',[]):
        prefix=label+f"_t{diagnostic['tick']}"
        required_files.append(original/'diagnostics'/(prefix+'_origin.wuyu'))
        required_files.extend(original/'diagnostics'/(prefix+'_'+kind+'.npz') for kind in ('intact','related_old','matched_other','restored'))
    if previous is not None and all(f.exists() for f in required_files):
        for prior in prior_outputs:
            candidate=Path(prior)/'summaries'/(label+'.json')
            if candidate.exists():regenerated=json.loads(candidate.read_text());break
        if not was_original:regenerated=previous
    if regenerated is None:
        # Every attempt has a fresh directory; interrupted create-only
        # checkpoint fragments can never collide with new executions.
        run_life((str(output),seed,init,mode,case,p))
        regenerated=json.loads((output/'summaries'/(label+'.json')).read_text())
    if previous is not None:
        differences=[k for k in previous.keys()|regenerated.keys()
                     if k!='elapsed_seconds' and previous.get(k)!=regenerated.get(k)]
        if differences:raise AssertionError((label,'surviving summary mismatch',differences))
    trace=output/'traces'/(label+'.npz')
    if not trace.exists():trace=original/'traces'/(label+'.npz')
    assert digest(trace)==regenerated['trace_sha256']
    for meta in (output/'diagnostics').glob(label+'_*_meta.json'):
        old=original/'diagnostics'/meta.name
        if old.exists():assert json.loads(meta.read_text())==json.loads(old.read_text()),meta.name
    # Only publish the regenerated evidence after all surviving records match.
    for folder in ('traces','checkpoints','diagnostics'):
        pattern=label+'.npz' if folder=='traces' else label+'_*'
        for source in (output/folder).glob(pattern):
            target=original/folder/source.name
            if target.exists():
                assert source.read_bytes()==target.read_bytes(),str(target)
                source.unlink()
            else:os.replace(source,target)
    if previous is None:shutil.copy2(output/'summaries'/(label+'.json'),existing)
    return dict(label=label,recovery_method='frozen_execution',matched_surviving_summary=previous is not None and was_original,
                trace_sha256=regenerated['trace_sha256'],
                raw_array_sha256=regenerated['raw_array_sha256'],
                final_state_sha256=regenerated['final_state_sha256'],
                reconstruction_seconds=regenerated['elapsed_seconds'])

def setup_runtime(name,workspace):
    original=ROOT/'experiments'/name
    p=json.loads((original/'protocol.json').read_text())
    runtime=ROOT
    changed=[k for k,v in p['sources'].items() if digest(ROOT/k)!=v]
    if changed:
        runtime=workspace/'runtimes'/name
        (runtime/'wuyu_unified').mkdir(parents=True,exist_ok=True)
        (runtime/'tools').mkdir(exist_ok=True)
        for path in (ROOT/'wuyu_unified').glob('*.py'):
            frozen=original/'source_snapshot'/path.name
            shutil.copy2(frozen if frozen.exists() else path,runtime/'wuyu_unified'/path.name)
        for name in ('run_relation_experiment.py','run_whole_experiment.py','analyze_relations.py','audit_relations.py'):
            frozen=original/'source_snapshot'/name
            shutil.copy2(frozen if frozen.exists() else ROOT/'tools'/name,runtime/'tools'/name)
        for name in ('source_projects','reference_runtime_v1','provenance'):
            link=runtime/name
            if not link.exists():link.symlink_to(ROOT/name,target_is_directory=True)
    for key,value in p['sources'].items():assert digest(runtime/key)==value,(name,key)
    return runtime,p

def main():
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('--workers',type=int,default=8)
    args=a.parse_args();workspace=ROOT.parent/'wuyu_reconstruction'
    prior_outputs=[str(q) for q in workspace.glob(args.cohort+'*') if q.is_dir()]
    output=workspace/(args.cohort+'_attempt_'+str(time.time_ns()));original=ROOT/'experiments'/args.cohort
    for directory in (output,original):
        for folder in ('summaries','traces','checkpoints','diagnostics'):(directory/folder).mkdir(parents=True,exist_ok=True)
    runtime,p=setup_runtime(args.cohort,workspace)
    sys.path.insert(0,str(runtime))
    from wuyu_unified import codec
    check=dict(p);expected=check.pop('protocol_digest');assert codec.digest(check)==expected
    record_path=original/'RECOVERY_COMPLETE_2026-10-13.json'
    if record_path.exists():raise FileExistsError(record_path)
    baseline_path=original/'RECOVERY_BASELINE_2026-10-13.json'
    if not baseline_path.exists():
        write(baseline_path,dict(created_utc=now(),original_summary_names=sorted(q.name for q in (original/'summaries').glob('*.json'))))
    original_names=set(json.loads(baseline_path.read_text())['original_summary_names'])
    start=time.perf_counter();jobs=[];prior_count=0;copies=[];representatives={}
    for seed in p['seeds']:
        for init in p['initializations']:
            for case in p['cases']:
                for mode in p['modes']:
                    label=f'{seed}_i{init}_{case}_{mode}'
                    saved=original/'summaries'/(label+'.json')
                    was_original=saved.name in original_names
                    prior_count+=was_original
                    old=json.loads(saved.read_text()) if was_original else None
                    # A surviving SHA256 identifies the complete compressed
                    # bytes, not merely an equal aggregate outcome. Runs with
                    # checkpoint/diagnostic obligations are always executed.
                    key=old['trace_sha256'] if old else None
                    required=(seed==p['seeds'][0] or bool(old and old.get('diagnostics')))
                    if key in representatives and not required:
                        copies.append((label,representatives[key],old))
                    else:
                        jobs.append((str(output),str(original),seed,init,mode,case,p,was_original,prior_outputs))
                        if key is not None:representatives[key]=label
    write(original/'RECOVERY_START_2026-10-13.json',dict(started_utc=now(),protocol_digest=expected,
        lives=len(jobs)+len(copies),executions=len(jobs),hash_identified_copies=len(copies),existing_summaries=prior_count,source_lock_verified=True,
        note='Restoration from frozen code and seeds after transient workspace reset. Not independent new evidence.'))
    records=[]
    with ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(str(runtime),)) as pool:
        futures=[pool.submit(recover,job) for job in jobs]
        for future in as_completed(futures):
            try:records.append(future.result())
            except BaseException:
                error=traceback.format_exc()
                (original/'RECOVERY_ERROR_2026-10-13.txt').write_text(error)
                print(error,flush=True)
                for child in pool._processes.values():child.terminate()
                raise
            if len(records)%24==0 or len(records)==len(jobs):
                print(json.dumps(dict(cohort=args.cohort,done=len(records),total=len(jobs),seconds=time.perf_counter()-start)),flush=True)
    for label,representative,old in copies:
        source=original/'traces'/(representative+'.npz')
        assert digest(source)==old['trace_sha256']
        destination=original/'traces'/(label+'.npz')
        if destination.exists():assert digest(destination)==old['trace_sha256']
        else:shutil.copy2(source,destination)
        records.append(dict(label=label,recovery_method='identical_original_SHA256',
            representative=representative,matched_surviving_summary=False,
            trace_sha256=old['trace_sha256'],raw_array_sha256=old['raw_array_sha256'],
            final_state_reexecuted=False))
    complete=dict(passed=True,completed_utc=now(),lives=len(records),protocol_digest=expected,
        matched_surviving_summaries=sum(r['matched_surviving_summary'] for r in records),
        original_hash_identified_copies=len(copies),
        wall_seconds=time.perf_counter()-start,records=sorted(records,key=lambda r:r['label']))
    write(record_path,complete)
    # Original run timing, if retained, remains unchanged and separate.
    if not (original/'RUN_COMPLETE.json').exists():
        write(original/'RUN_COMPLETE.json',dict(lives=len(records),protocol_digest=expected,
            wall_seconds=complete['wall_seconds'],completed_utc=now(),evidence_origin='2026-10-13 reconstruction with original frozen protocol'))
    print(json.dumps({k:v for k,v in complete.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
