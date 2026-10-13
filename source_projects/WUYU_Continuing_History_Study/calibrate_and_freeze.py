"""Irreversible selection rule; refuses to overwrite an existing formal freeze."""
import argparse
import json
import math
from pathlib import Path

from run import ROOT,sha,sources,utc_now,verify_freeze,write_json


def freeze(development):
    if (ROOT/'FORMAL_FREEZE.json').exists() or (ROOT/'formal_protocol.json').exists():raise RuntimeError('Formal protocol already exists')
    verify_freeze(ROOT/'DEVELOPMENT_FREEZE.json')
    development=Path(development);summary=json.loads((development/'summary.json').read_text())
    if json.loads((development/'audit.json').read_text())['status']!='passed':raise RuntimeError('Audit required')
    if 'OK' not in (ROOT/'validation/unit_tests.txt').read_text():raise RuntimeError('Unit validation required')
    c=json.loads((ROOT/'development_protocol.json').read_text())
    qualified=[t for t in c['primary_checkpoint_candidates'] if summary['checkpoints'][str(t)]['full_buffer_fraction']>=.9]
    selected=min(qualified) if qualified else 1200;sd=summary['checkpoints'][str(selected)]['selective_effect']['sd']
    widths={str(n):1.96*sd/math.sqrt(n) for n in c['holdout_count_candidates']}
    ns=[n for n in c['holdout_count_candidates'] if widths[str(n)]<=c['precision_target_halfwidth']]
    count=min(ns) if ns else max(c['holdout_count_candidates'])
    c.update(selected_primary_checkpoint=selected,holdout_seeds=dict(start=c['holdout_seed_start'],count=count),
             calibration=dict(formation_coverage_qualified=bool(qualified),full_buffer_fraction=summary['checkpoints'][str(selected)]['full_buffer_fraction'],
                              expected_halfwidths=widths,precision_target_expected_met=bool(ns),development_effect_sd=sd,
                              choices='Checkpoint by full-buffer coverage only; N by SD only. No parameter/architecture changes based on effect.'))
    write_json(ROOT/'formal_protocol.json',c)
    files=sources()
    for name in ('formal_protocol.json','ANALYSIS_DEFINITIONS.md','validation/unit_tests.txt'):
        files[name]=sha(ROOT/name)
    frozen=dict(frozen_utc=utc_now(),files=files,development_input_sha256={name:sha(development/name) for name in ('summary.json','audit.json','run_manifest.json')},
                primary_checkpoint=selected,holdout_count=count,note='Local timestamp and hashes establish within-package order; not an externally registered preregistration.')
    write_json(ROOT/'FORMAL_FREEZE.json',frozen);print(json.dumps(c['calibration'],indent=2));print(f'Frozen: checkpoint={selected}, N={count}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('development',type=Path);a=p.parse_args();freeze(a.development)
