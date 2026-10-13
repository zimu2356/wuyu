"""Assemble unchanged source projects and evidence from verified input ZIPs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = [*(f"WUYU_{i:02d}_complete.zip" for i in range(1, 9)),
            "WUYU_Integrated_Full.zip", "WUYU_Goal_Formation_Study.zip",
            "WUYU_Autonomous_Exploration_Study_2026-10-12.zip",
            "WUYU_Continuing_History_Study_2026-10-12.zip",
            "WUYU_Evidence_Revision_Source_2026-10-12.zip",
            *(f"WUYU_Evidence_Revision_Data_{i}_2026-10-12.zip" for i in range(1,4))]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    dest = ROOT / "source_projects"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "wuyu-source-import/1", "archives": [], "files": {}}
    for name in ARCHIVES:
        src = args.inputs / name
        if not src.is_file():
            raise FileNotFoundError(src)
        entry = {"name": name, "sha256": sha256(src), "bytes": src.stat().st_size, "members": 0}
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                rel = PurePosixPath(info.filename)
                if rel.is_absolute() or ".." in rel.parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unsafe ZIP member: {name}: {info.filename}")
                data = z.read(info)  # also validates each member CRC
                digest = hashlib.sha256(data).hexdigest()
                path = dest.joinpath(*rel.parts)
                key = str(rel)
                if key in manifest["files"]:
                    if manifest["files"][key]["sha256"] != digest:
                        raise ValueError(f"Conflicting source member: {key}")
                    manifest["files"][key]["archives"].append(name)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    manifest["files"][key] = {"sha256": digest, "bytes": len(data), "archives": [name]}
                entry["members"] += 1
        manifest["archives"].append(entry)
        print(json.dumps(entry), flush=True)
    out = ROOT / "provenance" / "SOURCE_IMPORT_MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "unique_members": len(manifest["files"]), "manifest": str(out)}))

if __name__ == "__main__":
    main()
