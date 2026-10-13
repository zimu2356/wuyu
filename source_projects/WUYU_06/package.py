from pathlib import Path
import json,hashlib,zipfile

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

if __name__=='__main__':
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='CHECKSUMS.sha256')
    checksums={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'CHECKSUMS.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in checksums.items()))
    files.append(ROOT/'CHECKSUMS.sha256')
    selected=set()
    for mode in ['full','plain_rnn']:
        for i in range(3):selected.add(json.loads((OUT/f'training_{mode}_{i}.json').read_text())['selected']['checkpoint'])
    summaries={'summary.json','formal_metrics.json','probes.json','audit.json','delivery_validation.json','final_gates.json','frozen_before_training.json','stress_summary.json','w05_diagnostic.json','recovery.json','training_summary.json','tests.txt','demo_actual.jsonl'}
    light=[]
    for p in files:
        rel=p.relative_to(ROOT)
        if p.name=='CHECKSUMS.sha256':continue
        if rel.parts[0]!='results' or p.name in selected|summaries or p.name.startswith('training_') and p.suffix=='.json' or p.name=='checkpoint_full_0_128.npz':light.append(p)
    destinations=[(ROOT.parent/'WUYU_06_run.zip',light),(ROOT.parent/'WUYU_06_complete.zip',files)]
    result=[]
    for destination,members in destinations:
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in members:
                compression=zipfile.ZIP_STORED if p.suffix in ('.npz','.png','.partial') else zipfile.ZIP_DEFLATED
                z.write(p,'WUYU_06/'+p.relative_to(ROOT).as_posix(),compress_type=compression)
            if destination.name.endswith('_run.zip'):
                z.writestr('WUYU_06/CHECKSUMS.sha256',''.join(f'{checksums[p.relative_to(ROOT).as_posix()]}  {p.relative_to(ROOT).as_posix()}\n' for p in members))
                z.writestr('WUYU_06/LIGHT_BUNDLE.txt','This package includes runnable code, selected models, one real checkpoint and summaries. It omits raw experiment arrays and other checkpoints. See README.md for regeneration instructions.\n')
        with zipfile.ZipFile(destination) as z:
            assert z.testzip() is None
            count=len(z.namelist())
        result.append({'file':str(destination),'bytes':destination.stat().st_size,'members':count,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'crc_passed':True})
    (ROOT.parent/'WUYU_06_packages.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
