import json,hashlib,zipfile,tempfile,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checksum(paths):return ''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in paths)
if __name__=='__main__':
    source=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('CHECKSUMS.sha256','packaging.log') and not p.name.endswith('.tmp')]
    chosen={json.loads(p.read_text())['selected']['file'] for p in (ROOT/'results').glob('training_*.json') if p.name!='training_wall_time.json'}
    light=[]
    for p in source:
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='engineering':continue
        if rel.parts[0]=='results':
            if p.suffix=='.npz' and p.name not in chosen and p.name!='checkpoint_graph_varied_0_384.npz':continue
            if p.name.startswith('train_') and p.suffix=='.log':continue
            if p.name=='probe_manifest.json':continue
        light.append(p)
    results=[]
    for name,paths in [('WUYU_08_run.zip',light),('WUYU_08_complete.zip',source)]:
        paths=sorted(paths);manifest=checksum(paths);dest=ROOT.parent/name
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in paths:z.write(p,'WUYU_08/'+p.relative_to(ROOT).as_posix())
            z.writestr('WUYU_08/CHECKSUMS.sha256',manifest)
        with zipfile.ZipFile(dest) as z:
            assert z.testzip() is None
            for line in manifest.splitlines():
                sha,rel=line.split('  ',1);assert hashlib.sha256(z.read('WUYU_08/'+rel)).hexdigest()==sha
        results.append(dict(file=str(dest),bytes=dest.stat().st_size,sha256=digest(dest),members=len(paths)+1))
    (ROOT/'CHECKSUMS.sha256').write_text(checksum(sorted(source)))
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(ROOT.parent/'WUYU_08_run.zip') as z:z.extractall(d)
        r=subprocess.run([sys.executable,'demo.py','--steps','3'],cwd=Path(d)/'WUYU_08',capture_output=True,text=True,timeout=60)
        assert r.returncode==0,r.stderr
        smoke=dict(exit_code=r.returncode,stdout=r.stdout)
    (ROOT.parent/'WUYU_08_package_smoke.json').write_text(json.dumps(smoke,indent=2))
    (ROOT.parent/'WUYU_08_packages.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
