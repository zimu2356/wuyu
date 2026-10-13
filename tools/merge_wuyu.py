"""Verify and combine the complete WUYU source ZIP parts (standard library)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description='校验并合并物欲完整源码分包；不需要第三方依赖。')
    p.add_argument('--parts-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path)
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if not args.verify_only and args.out is None:p.error('合并时需要 --out 指定一个尚不存在的目录')
    directory=args.parts_dir.resolve()
    manifest=json.loads((directory/'WUYU_Complete_Manifest_2026-10-13.json').read_text(encoding='utf-8'))
    root=manifest['root']
    if PurePosixPath(root).name!=root or root in ('.','..'):raise ValueError('invalid root')
    # Verify every archive before creating the output tree.
    for part in manifest['parts']:
        name=part['name']
        if Path(name).name!=name:raise ValueError('invalid part name')
        path=directory/name
        if path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:
            raise ValueError('分包缺失或校验失败：'+name)
    if not args.verify_only:args.out.mkdir(parents=True,exist_ok=False)
    seen=set()
    for part in manifest['parts']:
        with zipfile.ZipFile(directory/part['name']) as z:
            for info in z.infolist():
                posix=PurePosixPath(info.filename)
                if posix.is_absolute() or '..' in posix.parts or posix.parts[0]!=root:
                    raise ValueError('unsafe member path')
                if info.is_dir() or (info.external_attr>>16)&0o170000==0o120000:
                    raise ValueError('unexpected directory or symbolic link member')
                rel='/'.join(posix.parts[1:]);entry=manifest['files'].get(rel)
                if rel in seen or entry is None or entry['part']!=part['name']:
                    raise ValueError('unexpected or duplicate member: '+rel)
                if info.file_size!=entry['bytes']:raise ValueError('member size mismatch: '+rel)
                h=hashlib.sha256()
                if args.verify_only:
                    with z.open(info) as source:
                        for b in iter(lambda:source.read(1<<20),b''):h.update(b)
                else:
                    target=args.out.joinpath(*posix.parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(info) as source,target.open('xb') as dest:
                        for b in iter(lambda:source.read(1<<20),b''):
                            h.update(b);dest.write(b)
                if h.hexdigest()!=entry['sha256']:raise ValueError('member checksum mismatch: '+rel)
                seen.add(rel)
        print('已校验：'+part['name'],flush=True)
    if seen!=set(manifest['files']):raise ValueError('source members missing')
    print(json.dumps(dict(passed=True,parts=len(manifest['parts']),files=len(seen),
        output=None if args.verify_only else str((args.out/root).resolve())),ensure_ascii=False))


if __name__=='__main__':main()
