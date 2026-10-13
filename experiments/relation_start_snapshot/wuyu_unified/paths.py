from functools import lru_cache
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "source_projects"
PROJECTS = {
    **{f"{i:02}": f"WUYU_{i:02}" for i in range(1, 9)},
    "integrated": "WUYU_INTEGRATED",
    "goals": "WUYU_Goal_Formation_Study",
    "exploration": "WUYU_Autonomous_Exploration_Study",
    "history": "WUYU_Continuing_History_Study",
    "revision": "WUYU_Evidence_Revision_Study",
}

def project(profile):
    return SOURCES / PROJECTS[profile]

def config(profile):
    name = "formal_protocol.json" if profile in ("history", "revision") else "protocol.json"
    return json.loads((project(profile) / name).read_text())

@lru_cache(None)
def source_module(profile):
    if profile == "integrated":
        directory = str(project(profile))
        if directory not in sys.path:
            sys.path.insert(0, directory)
        return importlib.import_module("wuyu_core.runtime")
    relative = {"goals": "wuyu_goal_study/core.py", "exploration": "study.py",
                "history": "core.py", "revision": "core.py"}[profile]
    if profile == "exploration":
        directory = str(project(profile))
        if directory not in sys.path:
            sys.path.insert(0, directory)
    name = "_wuyu_frozen_" + profile
    spec = importlib.util.spec_from_file_location(name, project(profile) / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

