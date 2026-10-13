"""Require identical scientific outputs; wall clock and CPU timings excluded."""
import argparse
import json
from pathlib import Path
from run_study import sha256, write_json


FILES=('acquisition.jsonl.gz','initial_learners.jsonl.gz','test_traces.jsonl.gz',
       'knowledge.csv','episodes.csv','summary.json','data_audit.json')


def compare(first,second,output):
    items=[]
    for name in FILES:
        a,b=sha256(Path(first)/name),sha256(Path(second)/name)
        items.append(dict(file=name,first_sha256=a,second_sha256=b,identical=a==b))
    result=dict(status='passed' if all(r['identical'] for r in items) else 'failed',
                comparisons=items,excluded=['run_manifest.json (timestamps and CPU timing hashes)',
                                            'collector_timing.csv','timing_summary.json'])
    write_json(output,result)
    if result['status']!='passed':
        raise AssertionError('Scientific outputs differ')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('first',type=Path)
    parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compare(args.first,args.second,args.output)
