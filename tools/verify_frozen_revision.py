"""Full-trajectory equality against 30 archived formal lives, no new inference."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wuyu_unified.paths import project,source_module,config
from wuyu_unified.runtime import Session

def main():
    source=project("revision")/"holdout";c=config("revision");m=source_module("revision");results=[]
    for case in c["cases"]:
        for method in c["methods"]:
            stem=f"71000_{case}_{method}"
            with gzip.open(source/(stem+".npy.gz"),"rb") as f:expected=np.load(f,allow_pickle=False)
            with gzip.open(source/(stem+".json.gz"),"rt") as f:meta=json.load(f)
            s=Session.create("revision",seed=71000,case=case,mode=method);a=s.backend.agent
            actual=np.zeros_like(expected)
            for tick in range(len(actual)):
                before=[(x.mass,x.mass**2/max(x.mass2,1e-30),tick-a.last_use[i]) for i,x in enumerate(a.models)]
                q=s.step();action=q["action"];mass,ess,age=before[action];d=q["decision"];model=a.models[action];after=q["after"]
                actual[tick]=(action,d["purpose"],q["learning"]["revision"],model.streak,d["base_choice"],
                              [*after["body"],after["fatigue"]],d["pred"][action],float(d["score"][action]),
                              q["learning"]["residual"],age,min(a.visits[action]-1,c["model_window"]),mass,ess,model.version-1)
            fields={k:bool(np.array_equal(actual[k],expected[k])) for k in actual.dtype.names}
            final=m.digest(dict(world=s.backend.world.snapshot(),agent=a.snapshot()))
            final_ok=final==meta["summary"]["final_state_sha"]
            result=dict(seed=71000,case=case,method=method,ticks=len(actual),all_event_fields_exact=all(fields.values()),fields=fields,final_state_exact=final_ok)
            results.append(result);print(json.dumps(result),flush=True)
            if not all(fields.values()) or not final_ok:raise AssertionError(result)
    report=dict(passed=True,lives=len(results),ticks=sum(r["ticks"] for r in results),results=results,
                scope="All five methods and six scenarios, one archived seed; not an independent scientific replication of all 128 seeds")
    (ROOT/"validation/FROZEN_REVISION_REGRESSION.json").write_text(json.dumps(report,indent=2)+"\n")
if __name__=="__main__":main()
