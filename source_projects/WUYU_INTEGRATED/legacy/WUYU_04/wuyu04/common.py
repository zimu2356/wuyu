import hashlib
import json
from pathlib import Path
import numpy as np


def interval(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(70404)
    means = a[rng.integers(0, len(a), (4000, len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), n=len(a))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and (p.suffix == '.py' or p.name == 'PROTOCOL.md')}
