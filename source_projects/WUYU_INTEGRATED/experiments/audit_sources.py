"""Validate original archive fingerprints when available and preserved members."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    recovered=ROOT.parent/'wuyu_recovered_sources';rows=[]
    for item in json.loads((ROOT/'provenance/source_archives.json').read_text()):
        archive=recovered/item['archive'];version=item['archive'].split('_')[1]
        if not archive.exists():
            rows.append(dict(version=version,archive_available=False,note='Original archive optional; consult packaged per-file manifest'))
            continue
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['sha256'],archive
        count=0
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir():continue
                parts=Path(member.filename).parts
                if parts[0]==f'WUYU_{version}':parts=parts[1:]
                p=ROOT/'legacy'/f'WUYU_{version}'/Path(*parts)
                assert p.read_bytes()==z.read(member),str(p)
                count+=1
        rows.append(dict(version=version,archive_available=True,original_archive_hash_matches=True,unchanged_members=count))
    manifest_path=ROOT/'provenance/legacy_file_hashes.json'
    actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'legacy').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for name,digest in actual.items():
            assert manifest.get(name)==digest,'Packaged source differs: '+name
    else:
        manifest=actual;manifest_path.write_text(json.dumps(manifest,indent=2))
    frozen=json.loads((ROOT/'results/frozen_before_formal.json').read_text())
    mismatches=[str(p) for p,h in frozen.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    result=dict(archives=rows,manifest_files=len(manifest),present_files_verified=len(actual),omitted_files=len(manifest)-len(actual),formal_source_mismatches=mismatches)
    (ROOT/'validation/source_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if mismatches:raise SystemExit('formal source mismatch')


if __name__=='__main__':main()
