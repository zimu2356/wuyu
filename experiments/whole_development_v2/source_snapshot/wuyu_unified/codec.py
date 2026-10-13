"""Complete typed state: arrays, tuple dictionary keys and RNG tuples, no pickle."""
import base64
from collections import deque
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np

def pack(value):
    if isinstance(value, np.ndarray):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"type": "ndarray", "data": base64.b64encode(stream.getvalue()).decode("ascii")}
    if isinstance(value, np.generic):
        return pack(value.item())
    if isinstance(value, dict):
        return {"type": "dict", "items": [[pack(k), pack(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, deque)):
        return {"type": type(value).__name__, "items": [pack(x) for x in value]}
    if isinstance(value, set):
        values = [pack(x) for x in value]
        return {"type": "set", "items": sorted(values, key=canonical)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")

def unpack(value):
    kind = value["type"]
    if kind == "scalar": return value["value"]
    if kind == "ndarray":
        return np.load(io.BytesIO(base64.b64decode(value["data"])), allow_pickle=False)
    if kind == "dict": return {unpack(k): unpack(v) for k, v in value["items"]}
    constructor = {"list": list, "tuple": tuple, "set": set, "deque": deque}.get(kind)
    if constructor is None: raise ValueError(f"Unsupported state node: {kind}")
    return constructor(unpack(x) for x in value["items"])

def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(pack(value)).encode()).hexdigest()

def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, deque)): return [jsonable(x) for x in value]
    if isinstance(value, set): return sorted(jsonable(x) for x in value)
    return value

def write_new(path, data):
    """Atomic create; never replace an existing checkpoint or result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".wuyu-", dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)

def save(path, state):
    tree = pack(state)
    payload = canonical(tree).encode()
    envelope = dict(format="wuyu-typed-checkpoint/1", sha256=hashlib.sha256(payload).hexdigest(), tree=tree)
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as f:
        f.write(canonical(envelope).encode())
    write_new(path, stream.getvalue())

def load(path):
    with gzip.open(path,"rt",encoding="utf-8") as f: envelope=json.load(f)
    if envelope.get("format") != "wuyu-typed-checkpoint/1": raise ValueError("Unsupported checkpoint format")
    payload=canonical(envelope["tree"]).encode()
    if hashlib.sha256(payload).hexdigest() != envelope["sha256"]: raise ValueError("Checkpoint checksum mismatch")
    return unpack(envelope["tree"])

