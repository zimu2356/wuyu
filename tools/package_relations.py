"""Complete source plus nonoverlapping bounded new-data packages.

Only the actual inherited large experiment traces are omitted; their earlier
distribution is retained. Every new relation trace and checkpoint is included.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DATE='2026-10-12'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def category(path):
    rel=path.relative_to(ROOT)
    if rel.parts[0]=='experiments' and not rel.parts[1].startswith('relation_') and path.suffix=='.npz':
        return 'historical_raw'
    if rel.parts[:2]==('experiments','relation_migration'):
        # Executable migration examples and frozen expected observations allow
        # meaningful source-only verification without fetching all data parts.
        return 'source'
    if rel.parts[0]=='experiments' and rel.parts[1].startswith('relation_') and path.suffix in ('.npz','.wuyu'):
        return 'new_raw'
    return 'source'

def archive(job):
    output,files=job
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in files:z.write(path,arcname=ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    if output.stat().st_size>=490_000_000:raise ValueError('Archive exceeds transfer budget: '+str(output))
    return dict(name=output.name,bytes=output.stat().st_size,sha256=sha(output),members=len(files))

def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    for name in ('relation_development','relation_development_confirm','relation_holdout_main','relation_holdout_scenarios','relation_holdout_long'):
        assert json.loads((ROOT/'experiments'/name/'audit.json').read_text())['passed']
    assert (ROOT/'WUYU_Relation_Report_2026-10-12.html').exists()
    files=[];links=[]
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d!='__pycache__')
        for name in dirs:
            path=Path(base)/name
            if path.is_symlink():links.append(path.relative_to(ROOT).as_posix())
        for name in sorted(names):
            if name.startswith('.'):continue
            path=Path(base)/name
            if path.is_symlink():links.append(path.relative_to(ROOT).as_posix());continue
            if path.suffix in ('.pyc','.tmp','.recovered'):continue
            if name=='RELATION_DISTRIBUTION_MANIFEST.json':continue
            files.append(path)
    files=sorted(files)
    old=[p for p in files if category(p)=='historical_raw']
    source=[p for p in files if category(p)=='source']
    raw=[p for p in files if category(p)=='new_raw']
    chunks=[];chunk=[];size=0
    for path in raw:
        if chunk and size+path.stat().st_size>275_000_000:
            chunks.append(chunk);chunk=[];size=0
        chunk.append(path);size+=path.stat().st_size
    if chunk:chunks.append(chunk)
    package={p:'Source' for p in source}
    for i,chunk in enumerate(chunks,1):
        for p in chunk:package[p]=f'Data_{i:02}'
    # Hash all delivered files. The manifest itself cannot contain its own hash;
    # its integrity is covered by the archive hash and outer checksum file.
    entries={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p),package=package[p]) for p in sorted(package)}
    manifest=dict(root=ROOT.name,files=entries,
        note='All original source and runtime models, all new relation development/formal traces and checkpoints. Historical raw traces were already delivered and are not rerun or duplicated.',
        historical_raw_omitted=dict(count=len(old),bytes=sum(p.stat().st_size for p in old),previous_source_library_id='libfile_fd9dcc4c937c8191869a018514a89575'),
        reference_symlinks_omitted=links,restore_reference_links='python tools/restore_snapshot_links.py',
        manifest_self_hash='Covered by Source archive SHA256 and outer SHA256SUMS; no recursive self-hash.')
    manifest_path=ROOT/'provenance/RELATION_DISTRIBUTION_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    source.append(manifest_path)
    jobs=[(output/f'WUYU_Relation_Source_{DATE}.zip',source)]
    jobs += [(output/f'WUYU_Relation_Data_{i:02}_{DATE}.zip',chunk) for i,chunk in enumerate(chunks,1)]
    with ThreadPoolExecutor(3) as pool:archives=list(pool.map(archive,jobs))
    for name in (f'WUYU_Relation_Report_{DATE}.md',f'WUYU_Relation_Report_{DATE}.html',f'WUYU_Relation_HANDOFF_{DATE}.md'):
        shutil.copy2(ROOT/name,output/name)
    report=output/f'WUYU_Relation_Report_{DATE}.md'
    text=report.read_text()
    for stem in ('results','scenarios','timeline'):
        name=f'WUYU_Relation_{stem.title()}_{DATE}.png'
        shutil.copy2(ROOT/'figures'/f'relation_{stem}.png',output/name)
        text=text.replace(f'(figures/relation_{stem}.png)',f'(sandbox:{output.resolve()}/{name})')
    report.write_text(text)
    outer=dict(archives=archives,delivered_members=len(package)+1,new_data_parts=len(chunks),
        source_members=len(source),new_data_members=len(raw),root=ROOT.name)
    (output/f'WUYU_Relation_Archive_Manifest_{DATE}.json').write_text(json.dumps(outer,indent=2)+'\n')
    checks=[f'{sha(p)}  {p.name}' for p in sorted(output.iterdir()) if p.is_file() and not p.name.startswith('.')]
    (output/f'WUYU_Relation_SHA256SUMS_{DATE}.txt').write_text('\n'.join(checks)+'\n')
    print(json.dumps(outer,indent=2),flush=True)

if __name__=='__main__':main()
