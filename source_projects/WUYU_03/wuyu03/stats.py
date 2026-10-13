import hashlib
from pathlib import Path
import numpy as np


def interval(values, pairs=False):
    x=np.asarray(values,dtype=float)
    if pairs:x=(x[:len(x)//2]+x[len(x)//2:])/2
    rng=np.random.default_rng(70303)
    means=x[rng.integers(0,len(x),(2000,len(x)))].mean(axis=1)
    return {'mean':float(x.mean()),'low':float(np.quantile(means,.025)),
            'high':float(np.quantile(means,.975)),'n_units':len(x)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    paths=[p for p in Path(root).rglob('*') if p.is_file() and
           (p.suffix=='.py' or p.name=='PROTOCOL.md' or 'inputs' in p.parts)]
    return {str(p.relative_to(root)):digest(p) for p in sorted(paths)}
