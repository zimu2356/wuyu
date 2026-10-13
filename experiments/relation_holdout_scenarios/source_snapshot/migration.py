"""Explicit imports of complete native states; incomplete snapshots are rejected."""
import gzip
import importlib
import json
from pathlib import Path
from .adapters import ContinuousBackend,IntegratedBackend,LegacyBackend
from .paths import project,source_module,config
from .runtime import Session,sha

def import_native(profile,path):
    path=Path(path)
    lineage=dict(origin="native checkpoint import",source_filename=path.name,source_sha256=sha(path),
                 scope="continuation, not an additional formal experimental individual")
    if profile in ("integrated",) or profile.isdigit():
        source_module("integrated")
        state=importlib.import_module("wuyu_core.state").read_tree(path)
        if profile=="integrated":
            backend=IntegratedBackend.restore(state)
        else:
            if state.get("schema")!="wuyu.legacy-envelope/1" or state.get("version")!=int(profile):
                raise ValueError("01–08 import requires the complete old unified envelope, including external streams")
            backend=LegacyBackend.restore(state)
    elif profile in ("history","revision"):
        opener=gzip.open if path.suffix==".gz" else open
        with opener(path,"rt",encoding="utf-8") as f:state=json.load(f)
        c=config(profile);m=source_module(profile);namespace="main"
        if profile=="history":
            if state.get("config_sha256")!=m.digest(c) or state.get("core_sha256")!=sha(project(profile)/"core.py"):
                raise ValueError("history input must be a verified complete continue_life state.json")
            namespace="life"
        else:
            namespace="continuation"
            expected_config=sha(project(profile)/"formal_protocol.json")
            if "config_sha" in state:
                source_manifest=state["source_hashes"];actual_config=state["config_sha"]
            else:
                metadata_path=path.parent/"RUN_START.json"
                if not metadata_path.is_file():raise ValueError("native formal state needs its RUN_START.json provenance")
                metadata=json.loads(metadata_path.read_text())
                source_manifest=metadata["sources"];actual_config=metadata["config_sha"]
            if actual_config!=expected_config:raise ValueError("native protocol hash mismatch")
            for rel,expected in source_manifest.items():
                q=(project(profile)/rel).resolve()
                if not q.is_relative_to(project(profile).resolve()) or not q.is_file() or sha(q)!=expected:
                    raise ValueError(f"native scientific source mismatch: {rel}")
        ws=state.get("world",state.get("final_world"));ags=state.get("agent",state.get("final_agent"))
        if ws is None or ags is None:raise ValueError("incomplete native world/agent state")
        backend=ContinuousBackend(profile,m.World.restore(ws,c),m.Agent.restore(ags,c),c,None,namespace)
    else:
        raise ValueError("This original study's analytic snapshots are not full resumable states; use unified checkpoints")
    return Session(profile,backend,lineage=lineage)

def migrate08(path,index=0,mode="prior"):
    source_module("integrated")
    native=importlib.import_module("wuyu_core.migration").migrate_08(path,index,mode)
    return Session("integrated",IntegratedBackend(native),lineage=native.lineage)
