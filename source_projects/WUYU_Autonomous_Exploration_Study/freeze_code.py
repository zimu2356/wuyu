"""Freeze executable scientific code after development checks, before holdout."""
import json
from run_study import ROOT, science_sources, utc_now, verify_freeze, write_json


if __name__ == '__main__':
    verify_freeze(ROOT/'PROTOCOL_FREEZE.json')
    target = ROOT/'CODE_FREEZE.json'
    if target.exists():
        raise FileExistsError('Do not overwrite a scientific code freeze')
    dev = json.loads((ROOT/'outputs/development/run_manifest.json').read_text())
    if dev['status'] != 'complete':
        raise RuntimeError('Development run incomplete')
    tests = json.loads((ROOT/'validation/test_result.json').read_text())
    if tests['exit_code'] != 0:
        raise RuntimeError('Correctness checks failed')
    write_json(target, dict(frozen_utc=utc_now(), before_holdout=True,
                           development_run_completed_utc=dev['completed_utc'],
                           tests=tests, files=science_sources(),
                           statement='No outcome-driven changes to parameters or algorithms after development run.'))
    print('Scientific source frozen before holdout.')
