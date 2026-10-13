"""Create portable run/full bundles, without caches or duplicating input ZIPs."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
DEFAULTS={2:'results/checkpoint_matched.npz',3:'results/main/checkpoint_online_stage5.npz',
          4:'results/checkpoint_active_mid.npz',5:'results/checkpoint_mid.npz',
          6:'results/checkpoint_full_0_128.npz',7:'results/checkpoint_neural_student_0_384.npz',8:'results/checkpoint_graph_varied_0_384.npz'}


def main():
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    light=[]
    for p in files:
        rel=p.relative_to(ROOT)
        if rel.parts[0]!='legacy' or p.suffix.lower() in ('.py','.md','.json','.html','.txt','.sh','.toml') or p.name=='LICENSE':
            light.append(p);continue
        version=int(rel.parts[1].split('_')[1]);rest='/'.join(rel.parts[2:])
        if rest==DEFAULTS.get(version) or (version==8 and p.name.startswith('model_graph_varied_')):light.append(p)
    distribution={}
    for tag,selected in [('Run',light),('Full',files)]:
        path=ROOT.parent/f'WUYU_Integrated_{tag}.zip'
        contents=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in selected]
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in selected:z.write(p,'WUYU_INTEGRATED/'+str(p.relative_to(ROOT)))
            z.writestr('WUYU_INTEGRATED/DISTRIBUTION.json',json.dumps(dict(kind=tag,files=contents,omitted=[str(p.relative_to(ROOT)) for p in files if p not in selected]),indent=2))
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        distribution[tag]=dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),members=len(selected)+1)
    (ROOT.parent/'WUYU_Distribution_Checksums.json').write_text(json.dumps(distribution,indent=2));print(json.dumps(distribution,indent=2))


if __name__=='__main__':main()
