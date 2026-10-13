"""Run every original correctness suite in its own import namespace/process."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = [*(f"WUYU_{i:02}" for i in range(1,9)), "WUYU_INTEGRATED",
            "WUYU_Goal_Formation_Study", "WUYU_Autonomous_Exploration_Study",
            "WUYU_Continuing_History_Study", "WUYU_Evidence_Revision_Study"]

def run(name):
    out = ROOT / "validation" / "original_tests"
    out.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-B", "-m", "unittest"]
    args += ["-v", "test_mechanisms"] if name == "WUYU_Evidence_Revision_Study" else ["discover", "-s", "tests", "-v"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    start = time.monotonic()
    result = subprocess.run(args, cwd=ROOT / "source_projects" / name, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    (out / f"{name}.log").write_text(result.stdout)
    counts = re.findall(r"Ran (\d+) tests? in", result.stdout)
    return dict(project=name, command=args, returncode=result.returncode,
                tests=int(counts[-1]) if counts else None, seconds=round(time.monotonic()-start,3),
                log=f"validation/original_tests/{name}.log")

def main():
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run, name) for name in PROJECTS]):
            row = f.result(); rows.append(row); print(json.dumps(row),flush=True)
    rows.sort(key=lambda x: PROJECTS.index(x["project"]))
    result = dict(suites=rows, all_passed=all(x["returncode"]==0 for x in rows),
                  tests=sum(x["tests"] or 0 for x in rows), scope="original correctness suites; not retraining or new scientific evidence")
    (ROOT / "validation" / "ORIGINAL_TESTS.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["all_passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
