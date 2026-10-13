"""Original study commands run only in an isolated, new project copy."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .paths import ROOT,SOURCES,PROJECTS,project

COMMANDS={
 "01":[["run_experiment.py","--reuse-models"],["supplement.py"]],
 "02":[["run_experiment.py","--output","reproduced_results"]],
 "03":[["run_experiment.py","--output","reproduced_main"],["run_supplement.py","--output","reproduced_supplement"]],
 "04":[["run_experiment.py"]],
 "05":[["run_experiment.py"]],
 "06":[["run_eval.py"],["run_probes.py"],["run_stress.py"],["audit_results.py"]],
 "07":[["run_eval.py"],["run_probes.py"],["analyze.py"],["audit.py"]],
 "08":[["run_eval.py"],["analyze.py"],["audit.py"]],
 "integrated":[["experiments/run_integration.py","--workers","2"],["experiments/diagnose_prediction.py"],["experiments/diagnose_dependencies.py"]],
 "goals":[["run_study.py","--output","reproduced_results"]],
 "exploration":[["run_study.py","--cohort","holdout","--output","outputs/reproduced_results"],
                ["analyze.py","outputs/reproduced_results"],["audit_data.py","outputs/reproduced_results"]],
 "history":[["run.py","--cohort","holdout","--output","outputs/reproduced_results"],
            ["analyze.py","outputs/reproduced_results"],["audit.py","outputs/reproduced_results"]],
 "revision":[["run.py","--config","formal_protocol.json","--out","reproduced_results","--start","71000","--count","128","--workers","4"],
             ["analysis.py","--config","formal_protocol.json","--data","reproduced_results"],
             ["audit.py","--config","formal_protocol.json","--data","reproduced_results"]],
}

def plan(profile):
    if profile not in COMMANDS:raise ValueError("no original scientific experiment for this integration profile")
    return dict(profile=profile,source=str(project(profile).relative_to(ROOT)),commands=COMMANDS[profile],
                scope="re-evaluation with supplied selected models where applicable; not a claim of retraining",
                policy="create isolated copy; never run a historical writer inside source_projects")

def execute(profile,output):
    description=plan(profile);output=Path(output).resolve()
    if output==SOURCES.resolve() or output.is_relative_to(SOURCES.resolve()):
        raise ValueError("reproduction output must be outside immutable source projects")
    output.mkdir(parents=True,exist_ok=False)
    working=output/PROJECTS[profile]
    shutil.copytree(project(profile),working,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
    rows=[]
    (output/"REPRODUCTION_PLAN.json").write_text(json.dumps(description,ensure_ascii=False,indent=2)+"\n")
    for i,args in enumerate(COMMANDS[profile]):
        command=[sys.executable,"-B",*args]
        with (output/f"command_{i:02}.log").open("w") as log:
            result=subprocess.run(command,cwd=working,env=env,stdout=log,stderr=subprocess.STDOUT)
        rows.append(dict(command=args,returncode=result.returncode))
        (output/"REPRODUCTION_STATUS.json").write_text(json.dumps(rows,indent=2)+"\n")
        if result.returncode:raise RuntimeError(f"original command failed; see {output/f'command_{i:02}.log'}")
    return dict(output=str(output),commands=rows)

