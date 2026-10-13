"""Package unique data, frozen code and provenance without duplicate raw runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run import sha,verify_freeze,write_json


def main():
    verify_freeze(ROOT/'FORMAL_FREEZE.json')
    original=json.loads((ROOT/'outputs/holdout/run_manifest.json').read_text())
    repeated=json.loads((ROOT/'outputs/reproduction/run_manifest.json').read_text())
    assert original['output_sha256']==repeated['output_sha256']
    for name,value in original['output_sha256'].items():assert sha(ROOT/'outputs/holdout'/name)==value
    omitted=['outputs/reproduction/'+name for name in repeated['output_sha256']]
    write_json(ROOT/'PACKAGING.json',dict(format=1,omitted_identical_raw_copies=omitted,
               canonical_raw_directory='outputs/holdout',reproduction_evidence='outputs/holdout/reproduction.json',
               content='All development and holdout unique raw records, source, frozen protocols, analyses, audits, report and handoff. Duplicate reproduction raw bytes are represented by their matching original hashes; both run manifests are retained.',
               integrity_incident='See validation/RAW_LOG_INCIDENT.json; a truncated original was recovered from byte-exact reproduction. Corrupt duplicate is not presented as usable data.'))
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='checksums.json' and p.relative_to(ROOT).as_posix() not in omitted]
    files=sorted(files);checksums={p.relative_to(ROOT).as_posix():sha(p) for p in files};write_json(ROOT/'checksums.json',checksums)
    files.append(ROOT/'checksums.json');target=ROOT.parent/'WUYU_Continuing_History_Study_2026-10-12.zip'
    with zipfile.ZipFile(target,'w',allowZip64=True) as z:
        for p in sorted(files):
            relative=p.relative_to(ROOT).as_posix();info=zipfile.ZipInfo(ROOT.name+'/'+relative,date_time=(2026,10,12,12,0,0));info.external_attr=0o644<<16
            info.compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.png') else zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes(),compresslevel=6)
    with target.open('rb') as f:os.fsync(f.fileno())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,want in checksums.items():
            h=hashlib.sha256()
            with z.open(ROOT.name+'/'+name) as f:
                while data:=f.read(1024*1024):h.update(data)
            assert h.hexdigest()==want,name
        assert len(z.namelist())==len(files)
    verify_freeze(ROOT/'FORMAL_FREEZE.json')
    print(json.dumps(dict(status='passed',path=str(target),bytes=target.stat().st_size,members=len(files),sha256=sha(target),all_members_crc_and_sha256_verified=True),indent=2))


if __name__=='__main__':main()
