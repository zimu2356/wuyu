import hashlib,json,zipfile,subprocess,tempfile,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def eligible(p):return p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp')


if __name__=='__main__':
    out=ROOT/'results';selected=set()
    for family in ['rnn','student_final']:
        for i in range(3):selected.add(json.loads((out/f'training_{family}_{i}.json').read_text())['selected']['file'])
    selected.add('checkpoint_neural_student_0_384.npz')
    files=sorted(p for p in ROOT.rglob('*') if eligible(p) and p.name!='CHECKSUMS.sha256')
    checks=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files)
    (ROOT/'CHECKSUMS.sha256').write_text(checks)
    def build(light):
        name='WUYU_07_run.zip' if light else 'WUYU_07_complete.zip';target=ROOT.parent/name
        count=0;included=[]
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(ROOT.rglob('*')):
                if not eligible(p):continue
                rel=p.relative_to(ROOT)
                if p.name=='CHECKSUMS.sha256':continue
                if light and 'results' in rel.parts and p.suffix=='.npz' and p.name not in selected:continue
                z.write(p,Path('WUYU_07')/rel);count+=1
                included.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(rel)+'\n')
            z.writestr('WUYU_07/CHECKSUMS.sha256',''.join(included));count+=1
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            for line in z.read('WUYU_07/CHECKSUMS.sha256').decode().splitlines():
                digest,name=line.split('  ',1);assert hashlib.sha256(z.read('WUYU_07/'+name)).hexdigest()==digest
        return dict(path=str(target),bytes=target.stat().st_size,members=count,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    light=build(True)
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(light['path']) as z:z.extractall(d)
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        r=subprocess.run([sys.executable,'demo.py','--steps','3'],cwd=Path(d)/'WUYU_07',env=env,text=True,capture_output=True,check=True)
        smoke=dict(extracted_light_package=True,exit_code=r.returncode,stdout=r.stdout)
    (out/'package_smoke.json').write_text(json.dumps(smoke,ensure_ascii=False,indent=2))
    # Include the already completed extraction test in both final packages and refresh manifests.
    files=sorted(p for p in ROOT.rglob('*') if eligible(p) and p.name!='CHECKSUMS.sha256')
    (ROOT/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
    light=build(True);complete=build(False)
    results=[light,complete];(ROOT.parent/'WUYU_07_packages.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
