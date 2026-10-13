"""Non-overlapping self-contained source and bounded raw-data ZIPs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def is_raw(p):
    rel=p.relative_to(ROOT)
    return rel.parts[0]=='experiments' and p.suffix=='.npz'

def archive(job):
    out,files=job
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,arcname=ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    return dict(name=out.name,bytes=out.stat().st_size,sha256=sha(out),members=len(files))

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    assert json.loads((ROOT/'experiments/whole_holdout/audit.json').read_text())['passed']
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.recovered'))
    sources=[p for p in files if not is_raw(p)];raw=[p for p in files if is_raw(p)]
    chunks=[];chunk=[];size=0
    for p in raw:
        if chunk and size+p.stat().st_size>280_000_000:chunks.append(chunk);chunk=[];size=0
        chunk.append(p);size+=p.stat().st_size
    if chunk:chunks.append(chunk)
    entries={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p),package='Source' if p in sources else None) for p in files}
    for i,chunk in enumerate(chunks,1):
        for p in chunk:entries[str(p.relative_to(ROOT))]['package']=f'Data_{i:02}'
    manifest=dict(root=ROOT.name,files=entries,note='Complete source and all NEW experiment records. Historical bulk data remain in the previous seven data archives.')
    manifest_path=ROOT/'provenance/WHOLE_DISTRIBUTION_MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    sources.append(manifest_path)
    jobs=[(out/'WUYU_Whole_Source_2026-10-12.zip',sources)]+[(out/f'WUYU_Whole_Data_{i:02}_2026-10-12.zip',chunk) for i,chunk in enumerate(chunks,1)]
    with ThreadPoolExecutor(3) as pool:results=list(pool.map(archive,jobs))
    names=['WUYU_Whole_Report_2026-10-12.md','WUYU_Whole_Report_2026-10-12.html','WUYU_Whole_HANDOFF_2026-10-12.md']
    for name in names:shutil.copy2(ROOT/name,out/name)
    (out/'WUYU_Whole_Archive_Manifest_2026-10-12.json').write_text(json.dumps(dict(archives=results,uncompressed_members=len(files)+1,raw_data_parts=len(chunks)),indent=2)+'\n')
    checksum=[f"{sha(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file()]
    (out/'WUYU_Whole_SHA256SUMS_2026-10-12.txt').write_text('\n'.join(checksum)+'\n')
    print(json.dumps(results,indent=2),flush=True)
