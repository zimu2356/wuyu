import argparse
import json
from pathlib import Path

from run import sha,write_json


def compare(left,right):
    left=Path(left);right=Path(right)
    manifest=json.loads((left/'run_manifest.json').read_text())
    names=list(manifest['output_sha256'])+['summary.json','individual_effects.csv','audit.json']
    matches={name:dict(left=sha(left/name),right=sha(right/name),identical=sha(left/name)==sha(right/name)) for name in names}
    if not all(x['identical'] for x in matches.values()):raise AssertionError('Reproduction differs')
    result=dict(status='passed',files=matches,excluded='run_manifest.json timestamps and elapsed time; scientific data, summaries and audit must be byte-identical')
    write_json(left/'reproduction.json',result);print(json.dumps(dict(status='passed',byte_identical_files=len(matches))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');a=p.parse_args();compare(a.left,a.right)
