import json
from pathlib import Path
from .paths import ROOT,SOURCES
from .runtime import sha

def verify_sources(full=False):
    manifest=json.loads((ROOT/"provenance/SOURCE_IMPORT_MANIFEST.json").read_text())
    selected={k:v for k,v in manifest["files"].items()
              if full or k.endswith((".py",".js")) or Path(k).name.lower().endswith("protocol.json")}
    missing=[];changed=[]
    for name,entry in selected.items():
        p=SOURCES/name
        if not p.is_file():missing.append(name)
        elif sha(p)!=entry["sha256"]:changed.append(name)
    return dict(scope="all imported source and evidence bytes" if full else "Python, JavaScript and JSON protocols",
                selected=len(selected),checked=len(selected)-len(missing),missing=missing,changed=changed,
                passed=not missing and not changed)

def data_status():
    manifest=json.loads((ROOT/"provenance/SOURCE_IMPORT_MANIFEST.json").read_text())
    missing=[k for k in manifest["files"] if not (SOURCES/k).is_file()]
    return dict(expected=len(manifest["files"]),present=len(manifest["files"])-len(missing),
                missing=len(missing),full_data_installed=not missing,
                note="Extract all data ZIPs beside the source ZIP for full historical evidence")

