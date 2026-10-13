"""Package full sources plus nonoverlapping bounded historical-data ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
STEM=ROOT.name
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write_member(z,p,rel):
    info=zipfile.ZipInfo(STEM+'/'+rel,date_time=(2026,10,12,0,0,0));info.external_attr=0o100644<<16
    info.compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.zip') else zipfile.ZIP_DEFLATED
    with p.open('rb') as source,z.open(info,'w',force_zip64=True) as dest:
        for block in iter(lambda:source.read(1<<20),b''):dest.write(block)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['data','source','verify'],required=True);ap.add_argument('--out',type=Path,default=ROOT.parent/'wuyu_delivery');a=ap.parse_args();a.out.mkdir(exist_ok=True)
    imported=json.loads((ROOT/'provenance/SOURCE_IMPORT_MANIFEST.json').read_text())['files']
    selected=set(json.loads((ROOT/'provenance/SOURCE_PACKAGE_SELECTION.json').read_text()))
    data_manifest=ROOT/'provenance/DATA_PACKAGE_MANIFEST.json'
    if a.phase=='data':
        if data_manifest.exists():raise FileExistsError(data_manifest)
        remaining=sorted(set(imported)-selected);parts=[];part=None;z=None;mapping={}
        for n in remaining:
            p=ROOT/'source_projects'/n
            if z is None or (z.fp.tell()+p.stat().st_size>300_000_000 and part['files']):
                if z:
                    z.close();part['bytes']=(a.out/part['filename']).stat().st_size;part['sha256']=sha(a.out/part['filename']);print(json.dumps({k:v for k,v in part.items() if k!='files'}),flush=True)
                name=f'WUYU_Unified_Data_{len(parts)+1}_2026-10-12.zip';part=dict(filename=name,files=[]);parts.append(part)
                if (a.out/name).exists():raise FileExistsError(a.out/name)
                z=zipfile.ZipFile(a.out/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)
            write_member(z,p,'source_projects/'+n);part['files'].append(n);mapping[n]=part['filename']
        if z:
            z.close();part['bytes']=(a.out/part['filename']).stat().st_size;part['sha256']=sha(a.out/part['filename']);print(json.dumps({k:v for k,v in part.items() if k!='files'}),flush=True)
        data_manifest.write_text(json.dumps(dict(parts=parts,files=mapping),indent=2)+'\n')
    elif a.phase=='source':
        path=a.out/'WUYU_Unified_Source_2026-10-12.zip'
        files=[]
        for p in ROOT.rglob('*'):
            if not p.is_file():continue
            rel=str(p.relative_to(ROOT))
            if rel.startswith('source_projects/'):
                if rel.removeprefix('source_projects/') not in selected:continue
            elif '__pycache__' in p.parts or p.suffix=='.pyc':continue
            files.append((rel,p))
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for rel,p in sorted(files):write_member(z,p,rel)
        print(json.dumps(dict(filename=path.name,members=len(files),bytes=path.stat().st_size,sha256=sha(path))))
    else:
        seen=set();packages=[]
        for path in sorted(a.out.glob('WUYU_Unified_*.zip')):
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    assert info.filename not in seen,info.filename;seen.add(info.filename)
                    data=z.read(info)
                    prefix=STEM+'/source_projects/'
                    if info.filename.startswith(prefix):
                        n=info.filename.removeprefix(prefix);assert hashlib.sha256(data).hexdigest()==imported[n]['sha256'],n
            packages.append(dict(filename=path.name,bytes=path.stat().st_size,sha256=sha(path)))
        found={n.removeprefix(STEM+'/source_projects/') for n in seen if n.startswith(STEM+'/source_projects/')}
        assert found==set(imported),(len(found),len(imported))
        result=dict(passed=True,original_members=len(found),all_original_bytes_preserved=True,overlap=0,packages=packages)
        (a.out/'PACKAGE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
