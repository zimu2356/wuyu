"""Verify archive membership/hashes and run the extracted source independently."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

DATE='2026-10-12'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def check_archive(job):
    path,record,expected,root=job
    assert digest(path)==record['sha256'] and path.stat().st_size==record['bytes']
    with zipfile.ZipFile(path) as z:
        names=z.namelist();assert len(names)==len(set(names))==record['members']
        for name in names:
            parts=PurePosixPath(name).parts
            assert parts[0]==root and '..' not in parts and not name.startswith('/')
            relative='/'.join(parts[1:])
            raw=z.read(name) # Also validates ZIP CRC for every member.
            if relative=='provenance/RELATION_DISTRIBUTION_MANIFEST.json':continue
            wanted=expected[relative]
            assert len(raw)==wanted['bytes']
            assert hashlib.sha256(raw).hexdigest()==wanted['sha256'],name
        return dict(archive=path.name,checked_members=len(names),all_member_hashes_and_crc_pass=True,names=names)

def main():
    directory=Path(sys.argv[1]).resolve()
    outer=json.loads((directory/f'WUYU_Relation_Archive_Manifest_{DATE}.json').read_text())
    source=directory/f'WUYU_Relation_Source_{DATE}.zip';root=outer['root']
    with zipfile.ZipFile(source) as z:
        manifest=json.loads(z.read(root+'/provenance/RELATION_DISTRIBUTION_MANIFEST.json'))
    expected=manifest['files']
    jobs=[(directory/r['name'],r,expected,root) for r in outer['archives']]
    with ThreadPoolExecutor(3) as pool:checks=list(pool.map(check_archive,jobs))
    allnames=[n for c in checks for n in c.pop('names')]
    assert len(allnames)==len(set(allnames))==len(expected)+1
    assert set(allnames)=={root+'/'+name for name in expected}|{root+'/provenance/RELATION_DISTRIBUTION_MANIFEST.json'}
    with tempfile.TemporaryDirectory(prefix='wuyu_relation_portable_',dir=str(directory.parent)) as temporary:
        with zipfile.ZipFile(source) as z:z.extractall(temporary)
        project=Path(temporary)/root
        environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
        environment.pop('PYTHONPATH',None)
        verify=subprocess.run([sys.executable,'-B','-m','wuyu_unified','verify-sources'],cwd=project,env=environment,text=True,capture_output=True,check=True)
        script='''from pathlib import Path
import json
import wuyu_unified
from wuyu_unified.runtime import Session
from wuyu_unified import codec
assert Path(wuyu_unified.__file__).resolve().is_relative_to(Path.cwd().resolve())
results={}
fresh=Session.create('whole',seed=888099,mode='R10',case='r_block',horizon=96,initialization=1)
assert len(fresh.run(16))==16 and fresh.event_index==16
results['fresh_R10']=dict(loaded_inherited_model_from_extracted_source=True,physical_steps=16)
for mode in ('R00','R10'):
    path=Path('experiments/relation_migration')/f'final_{mode}_t1097.wuyu'
    a=Session.load(path);b=Session.load(path)
    assert a.event_index==1097 and len(a.backend.actor.buffer)==9
    assert codec.digest(a.run(64))==codec.digest(b.run(64))
    results[mode]=dict(loaded_with_verified_current_sources=True,next_64_events_exact=True)
print(json.dumps(results))
'''
        continuation=subprocess.run([sys.executable,'-B','-c',script],cwd=project,env=environment,text=True,capture_output=True,check=True)
    result=dict(passed=True,archives=checks,nonoverlapping_members=len(allnames),
        full_manifest_coverage=True,source_only_portable_import_and_continuation=json.loads(continuation.stdout),
        inherited_source_check_stdout=verify.stdout,
        historical_raw_trace_note=manifest['historical_raw_omitted'])
    path=directory/f'WUYU_Relation_Verification_{DATE}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    sums=directory/f'WUYU_Relation_SHA256SUMS_{DATE}.txt'
    sums.write_text('\n'.join(f'{digest(p)}  {p.name}' for p in sorted(directory.iterdir()) if p.is_file() and p!=sums and not p.name.startswith('.'))+'\n')
    print(json.dumps(dict(passed=True,archives=len(checks),nonoverlapping_members=len(allnames),portable_source=True),indent=2))

if __name__=='__main__':main()
