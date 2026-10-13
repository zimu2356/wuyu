"""Finish recovery and audit without changing the frozen scientific program."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
VALIDATION=ROOT/'experiments/relation_validation'

def run(arguments,log):
    with (VALIDATION/log).open('w') as stream:
        subprocess.run([sys.executable,'-B',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    marker=ROOT/'experiments/relation_holdout_main/RECOVERY_COMPLETE_2026-10-13.json'
    while not marker.exists():
        logfile=ROOT.parent/'wuyu_reconstruction_main.log'
        if logfile.exists() and 'Traceback (most recent call last)' in logfile.read_text():
            raise RuntimeError('Main reconstruction failed; inspect its log before any retry.')
        time.sleep(5)
    print('Main raw evidence restored. An independent process audits it while remaining cohorts are restored.',flush=True)
    for name in ('relation_holdout_scenarios','relation_holdout_long','relation_development','relation_development_confirm'):
        if not (ROOT/'experiments'/name/'RECOVERY_COMPLETE_2026-10-13.json').exists():
            run(['tools/recover_relation_evidence.py',name,'--workers','8'],'recovery_'+name+'.log')
        print(name+' restored.',flush=True)
    names=('relation_development','relation_development_confirm','relation_holdout_main','relation_holdout_scenarios','relation_holdout_long')
    for name in names:
        if name!='relation_holdout_main':
            run(['tools/audit_relations.py','experiments/'+name,'--workers','8'],'recovery_audit_'+name+'.log')
        else:
            while not (VALIDATION/'RECOVERY_MAIN_AUDIT_DONE_2026-10-13.json').exists():time.sleep(5)
        if name.startswith('relation_holdout'):
            run(['tools/analyze_relations.py','experiments/'+name],'recovery_analysis_'+name+'.log')
        print(name+' analyzed and audited.',flush=True)
    records=[json.loads((ROOT/'experiments'/name/'RECOVERY_COMPLETE_2026-10-13.json').read_text()) for name in names]
    result=dict(passed=True,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        note='Same frozen protocols and seeds; reconstructed missing bytes are not an added independent sample.',
        lives=sum(r['lives'] for r in records),
        exact_surviving_full_summaries=sum(r['matched_surviving_summaries'] for r in records),
        original_hash_identified_copies=sum(r['original_hash_identified_copies'] for r in records),
        cohorts=[{k:v for k,v in r.items() if k!='records'} for r in records])
    (VALIDATION/'RECOVERY_PIPELINE_COMPLETE_2026-10-13.json').write_text(json.dumps(result,indent=2)+'\n')
    for script in ('summarize_relation_evidence','plot_relation_results','build_relation_report'):
        run(['tools/'+script+'.py'],'recovery_'+script+'.log')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
