"""Wait for immutable run completion, then analyze and audit every formal life."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def wait_complete(name):
    path=ROOT/'experiments'/name/'RUN_COMPLETE.json'
    while not path.exists():time.sleep(10)
    return json.loads(path.read_text())

def command(args,log):
    start=time.perf_counter()
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{args[0]} failed; see {log.relative_to(ROOT)}')
    return dict(command=args,seconds=time.perf_counter()-start)

def main():
    output=ROOT/'experiments/relation_validation';records=[]
    (output/'ANALYSIS_PIPELINE_START.json').write_text(json.dumps(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    name='relation_holdout_main';wait_complete(name)
    directory=ROOT/'experiments'/name
    print('Main cohort complete: analyze and audit while auxiliary runs continue.',flush=True)
    records.append(command(['tools/analyze_relations.py','experiments/'+name],directory/'analysis_stdout.json'))
    records.append(command(['tools/audit_relations.py','experiments/'+name,'--workers','4'],directory/'audit_stdout.json'))
    # Wait for both simulations before assigning the full pool to the remaining
    # audits, keeping audit concurrency from extending the final live cohort.
    for name in ('relation_holdout_scenarios','relation_holdout_long'):wait_complete(name)
    for name in ('relation_holdout_scenarios','relation_holdout_long'):
        directory=ROOT/'experiments'/name
        records.append(command(['tools/analyze_relations.py','experiments/'+name],directory/'analysis_stdout.json'))
        records.append(command(['tools/audit_relations.py','experiments/'+name,'--workers','7'],directory/'audit_stdout.json'))
    for script in ('summarize_relation_evidence','plot_relation_results','build_relation_report'):
        records.append(command(['tools/'+script+'.py'],output/(script+'.log')))
    result=dict(passed=True,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stages=records)
    (output/'ANALYSIS_PIPELINE_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
