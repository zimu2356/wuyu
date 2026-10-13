"""Package only completed, audited runs. Verify every ZIP member and the final archive."""
import os,sys,json,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from storage import sha,write_json

ROOT=Path(__file__).resolve().parents[1]
def main():
    freeze=json.loads((ROOT/'FORMAL_FREEZE.json').read_text())
    for name,h in freeze['files'].items(): assert sha(ROOT/name)==h,(name,'frozen source changed')
    for folder in ('development','holdout','reproduction'):
        assert json.loads((ROOT/folder/'AUDIT.json').read_text())['passed']
    proof=json.loads((ROOT/'validation/REPRODUCTION_COMPARISON.json').read_text());assert proof['passed']
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name.endswith('PACKAGE_MANIFEST.json'): continue
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='reproduction' and (p.name.endswith('.npy.gz') or p.name.endswith('.json.gz') or p.name.startswith('block_')): continue
        files.append(p)
    results=[]
    for kind in ('Source','Full'):
        chosen=[]
        for p in files:
            rel=p.relative_to(ROOT)
            if kind=='Source' and rel.parts[0] in ('development','holdout','reproduction','pilot_stable') and p.name not in ('ANALYSIS.json','AUDIT.json','COMPLETE.json','RUN_START.json','MANIFEST.json','summaries.json'): continue
            if kind=='Source' and rel.parts[0]=='development_diagnostics_v2' and p.name!='DIAGNOSTICS_SUMMARY.json': continue
            if kind=='Source' and rel.parts[0]=='validation' and p.name.startswith('resume_'): continue
            chosen.append(p)
        entries={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in chosen}
        manifest=ROOT/(kind.upper()+'_PACKAGE_MANIFEST.json')
        write_json(manifest,dict(kind=kind,files=entries,duplicate_omissions='Reproduction raw trajectories/model states/block receipts omitted only after equality of every manifest entry; original complete records are in holdout.',
             source_package_scope='Full implementation, protocols, diagnostics summaries, validation and report. Full package additionally includes every unique raw development and holdout trajectory.'))
        chosen.append(manifest);destination=ROOT.parent/f'WUYU_Evidence_Revision_{kind}_2026-10-12.zip'
        with zipfile.ZipFile(destination,'w',allowZip64=True) as z:
            for p in chosen:
                rel=str(p.relative_to(ROOT));info=zipfile.ZipInfo(ROOT.name+'/'+rel,date_time=(2026,10,12,0,0,0));info.external_attr=0o644<<16
                info.compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.png','.zip') else zipfile.ZIP_DEFLATED
                z.writestr(info,p.read_bytes(),compresslevel=6 if info.compress_type==zipfile.ZIP_DEFLATED else None)
        with destination.open('rb') as f: os.fsync(f.fileno())
        expected={**{ROOT.name+'/'+name:v['sha256'] for name,v in entries.items()},ROOT.name+'/'+manifest.name:sha(manifest)}
        import hashlib
        with zipfile.ZipFile(destination) as z:
            assert z.testzip() is None;assert set(z.namelist())==set(expected)
            for name,h in expected.items(): assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        result=dict(kind=kind,path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),members=len(expected),all_members_verified=True)
        results.append(result);print(json.dumps(result),flush=True)
    write_json(ROOT.parent/'WUYU_Evidence_Revision_Package_Validation_2026-10-12.json',dict(passed=True,archives=results))

if __name__=='__main__': main()
