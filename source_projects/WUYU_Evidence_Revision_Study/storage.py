"""Deterministic, closed-and-synced records; no pickle or lossy state compression."""
import gzip, hashlib, json, os
from pathlib import Path
import numpy as np
from core import canonical

DTYPE=np.dtype([('action','u1'),('purpose','u1'),('revision','u1'),('streak','u1'),('base_choice','u1'),
                ('after','<f8',(3,)),('prediction','<f8',(3,)),('score','<f8'),('residual','<f8'),
                ('age','<u2'),('n','u1'),('mass','<f8'),('ess','<f8'),('version','<u4')])
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def write_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f: f.write(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def write_gz_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=5) as g: g.write(canonical(x).encode())
        f.flush();os.fsync(f.fileno())
def read_gz_json(path):
    with gzip.open(path,'rt') as f: return json.load(f)
def write_events(path,rows):
    with open(path,'wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=5) as g:
            np.lib.format.write_array(g,rows,allow_pickle=False)
        f.flush();os.fsync(f.fileno())
def read_events(path):
    with gzip.open(path,'rb') as f: return np.lib.format.read_array(f,allow_pickle=False)
def source_hashes(root):
    return {p.name:sha(p) for p in sorted(Path(root).glob('*.py'))}
