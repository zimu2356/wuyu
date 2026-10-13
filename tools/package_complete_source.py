"""Build bounded, independent ZIPs containing all current source dependencies.

Keep historical and relation raw-data distributions separate. Every current
acceptance record is included. No existing delivery archive is modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from package_relations import category

ROOT=Path(__file__).resolve().parents[1]
DATE='2026-10-13'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def archive(job):
    output,paths=job
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in paths:z.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    assert output.stat().st_size<32_000_000,(output,output.stat().st_size)
    return dict(name=output.name,bytes=output.stat().st_size,sha256=sha(output),members=len(paths))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--previous-source',required=True,type=Path)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'validation/system_acceptance_2026-10-13'
    for name in ('ALL_TESTS.json','full/ACCEPTANCE.json','migrations/MIGRATIONS.json'):
        assert json.loads((evidence/name).read_text())['passed']
    with zipfile.ZipFile(args.previous_source) as z:
        previous={x.filename.split('/',1)[1]:x for x in z.infolist()}
    files=[];omitted=[];symlinks=[]
    for directory,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d!='__pycache__')
        for name in dirs:
            path=Path(directory)/name
            if path.is_symlink():symlinks.append(path.relative_to(ROOT).as_posix())
        for name in sorted(names):
            path=Path(directory)/name
            if name.startswith('.') or path.suffix in ('.pyc','.tmp','.recovered'):continue
            if path.is_symlink():symlinks.append(path.relative_to(ROOT).as_posix());continue
            if category(path)!='source':omitted.append(path);continue
            files.append(path)
    files.sort()
    # Old compressed sizes are estimates only; additions use an upper bound.
    # Reserve space for ZIP metadata and changed small documentation.
    chunks=[];chunk=[];budget=0
    for path in files:
        rel=path.relative_to(ROOT).as_posix();old=previous.get(rel)
        estimate=old.compress_size if old and old.file_size==path.stat().st_size else path.stat().st_size
        estimate+=len(rel)*2+300
        if chunk and budget+estimate>28_000_000:
            chunks.append(chunk);chunk=[];budget=0
        chunk.append(path);budget+=estimate
    if chunk:chunks.append(chunk)
    jobs=[(out/f'WUYU_Complete_Source_{i:02}_of_{len(chunks):02}_{DATE}.zip',paths) for i,paths in enumerate(chunks,1)]
    membership={p:output.name for output,paths in jobs for p in paths}
    entries={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p),part=membership[p]) for p in files}
    with ThreadPoolExecutor(max_workers=3) as pool:parts=list(pool.map(archive,jobs))
    manifest=dict(schema='wuyu-complete-source-parts/1',root=ROOT.name,parts=parts,files=entries,
        source_only=True,contains='All original and current source, required existing models, migration examples, study summaries, complete current engineering acceptance evidence.',
        separate_raw_evidence=dict(files=len(omitted),bytes=sum(p.stat().st_size for p in omitted)),
        source_symlink_aliases_omitted=symlinks,source_snapshot_links_restore='python3 tools/restore_snapshot_links.py',
        historical_manifests='Retained older release manifests describe their original releases. This manifest is authoritative for this complete-source delivery.')
    (out/f'WUYU_Complete_Manifest_{DATE}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(ROOT/'tools/merge_wuyu.py',out/'merge_wuyu.py')
    shutil.copyfile(ROOT/f'WUYU_Complete_Verification_{DATE}.md',out/f'WUYU_Complete_Verification_{DATE}.md')
    print(json.dumps(dict(parts=parts,files=len(files),total_bytes=sum(p['bytes'] for p in parts)),indent=2))


if __name__=='__main__':main()
