"""Fixed engineering acceptance experiment, not a new developmental claim.

Freeze the protocol and source hashes BEFORE any cohort observations. Run all
conditions without selecting winners or tuning parameters from these results.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wuyu_unified.runtime import Session,source_locks
from wuyu_unified import codec

PROTOCOL=dict(version=1,scope="integration acceptance; exploratory paired effects, no parameter optimization",
 seeds=list(range(920000,920032)),initialization_rule="(seed-920000)%4; conditional on four supplied trained policies",
 horizon=1536,cases=["stationary","recipe_change","developmental"],
 methods=["prior","replay","online","evidence","update","revision","combined","combined_no_memory"],
 worlds={"stationary":"Original world, phase held at 0, no shocks; no resource dependencies",
         "recipe_change":"Same as stationary; at tick 768 replace the two hidden button codes at the original preselected changed site; no notification to agent",
         "developmental":"Original 1536-tick four-phase world with dependencies, recipe changes and six bodily shocks"},
 comparison="Within seed, same trained initialization, initial state and coupled private action draw; histories may subsequently diverge",
 learning="One real primitive action at a time; only observed feedback trains heads. Continuous body and inventory; WAIT has physical cost. No teacher/planner/oracle at runtime.",
 combined="Fixed 0.5 neural/0.5 evidence predicted utility, clipped to [-1,1], plus E; U updates evidence head; neural head trains online and prefix replay",
 combined_no_memory="Disable associative transition/effect/motor learning from birth; observation bookkeeping continues. Diagnostic ablation, not a new autonomous mode.",
 budget="1536 primitive actions per run, INCLUDING post-death absorbing ticks; extra compute counted, not matched and not charged to body",
 endpoints={"alive_fraction":"Fraction alive AFTER action over the full declared lifetime",
            "burden":"Mean sum((1-body)^2); an absorbing dead state scores 2",
            "late_burden":"Same over last 384 ticks; includes every dead individual",
            "late_deliveries":"Actual successful takes (body resources or materials) over last 384 ticks",
            "final_alive":"Alive at the declared horizon"},
 acceptance="Combined vs prior, in EACH case: paired-seed descriptive bootstrap95 lower(alive_fraction delta)>-0.02 AND upper(burden delta)<0.02. This is a declared engineering nondegradation screen, not confirmatory general superiority. Failure to clear is failure to establish compatibility, not proof of equivalence or harm.",
 bootstrap_seed=927401,bootstrap_resamples=10000,
 limitations=["32 seeds; four selected pretrained policies, each used for eight worlds, not a full seed-by-initialization cross",
 "New feature transfer and fixed mixture are engineering choices; no theoretical guarantee of beneficial composition",
 "Stationary and recipe-change controls have no dependencies; developmental case mixes changes and bodily shocks",
 "No new desires, learned value standards, social relations, self-model, subjective experience or long-chain generalization tested"])

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def run_seed(args):
    seed,out=args;out=Path(out);rows=[];start=time.perf_counter()
    for case in PROTOCOL["cases"]:
        for method in PROTOCOL["methods"]:
            mode="shadow" if method=="prior" else "combined" if method=="combined_no_memory" else method
            profile="integrated" if method=="replay" else "workshop_bridge"
            s=Session.create(profile,seed=seed,initialization=(seed-920000)%4,mode=mode,horizon=1536)
            native=s.backend.native
            if case!="developmental":native.world.regime_override=0;native.world.shocks=False
            if method=="combined_no_memory":native.individual.memory.learn_enabled=False
            dtype=[("tick","i4"),("action","i1"),("native_choice","i1"),("body","f8",(2,)),
                   ("alive","?"),("product","i1"),("blocked","?"),("revision","?"),
                   ("head_mse","f8"),("bias_used","?"),("counterfactual_choice_changed","?"),
                   ("neural_updates","i4"),("replay_updates","i4"),("evidence_updates","i4"),
                   ("memory_updates","i4"),("burden","f8")]
            trace=np.zeros(1536,dtype=dtype)
            for t in range(1536):
                if case=="recipe_change" and t==768:
                    w=native.world;w.codes[0,int(w.changed[0])]=w.replacement[0]
                q=s.step();n=q["payload"] if method=="replay" else q["native_payload"]
                before=n["obs"][0];outcome=n["outcome"];after=outcome["body"][0];alive=bool(outcome["alive"][0])
                req=int(before[9:12].argmax())-1
                blocked=bool(n["alive"] and n["action"]==5 and before[6:9].argmax()==2 and req>=0 and before[12+req]==0)
                if method=="replay":
                    nc=-1;revised=False;mse=n["prediction_mse"];used=n["bias_applied"];changed=False;updates=0
                else:
                    nc=q["decision"]["native_choice"];revised=q["learning"]["revision"]
                    mse=float(np.mean((q["prediction"]-q["learning"]["target"])**2))
                    used=q["decision"]["use"];changed=used and n["action"]!=nc;updates=sum(s.backend.visits)
                trace[t]=(t,n["action"],nc,after,alive,int(outcome["product"][0]),blocked,revised,mse,used,changed,
                          native.individual.predictor.updates,native.individual.predictor.replay_updates,
                          updates,native.individual.memory.updates,float(np.sum((1-after)**2)) if alive else 2.)
            filename=f"{seed}_{case}_{method}"
            with gzip.GzipFile(filename=str(out/"raw"/(filename+".npy.gz")),mode="wb",mtime=0) as f:
                np.save(f,trace,allow_pickle=False)
            if seed==920000:s.save(out/"checkpoints"/(filename+".wuyu"))
            alive=trace["alive"];valid=np.r_[True,alive[:-1]]
            rows.append(dict(seed=seed,initialization=(seed-920000)%4,case=case,method=method,
                alive_fraction=float(alive.mean()),final_alive=bool(alive[-1]),burden=float(trace["burden"].mean()),
                late_burden=float(trace["burden"][-384:].mean()),late_deliveries=int((trace["product"][-384:]<4).sum()),
                deliveries=int((trace["product"]<4).sum()),blocked=int(trace["blocked"].sum()),
                revisions=int(trace["revision"].sum()),choice_changes=int(trace["counterfactual_choice_changed"].sum()),
                evidence_updates=int(trace["evidence_updates"][-1]),neural_updates=int(trace["neural_updates"][-1]),
                replay_updates=int(trace["replay_updates"][-1]),memory_updates=int(trace["memory_updates"][-1]),
                observed_head_mse=float(trace["head_mse"][valid].mean()),
                raw_sha256=hashlib.sha256((out/"raw"/(filename+".npy.gz")).read_bytes()).hexdigest(),
                final_state_sha256=codec.digest(s.state(with_locks=False))))
    write_json(out/"metrics"/f"{seed}.json",rows)
    return dict(seed=seed,runs=len(rows),seconds=round(time.perf_counter()-start,2))

def analyse(out):
    rows=[]
    for seed in PROTOCOL["seeds"]:rows+=json.loads((out/"metrics"/f"{seed}.json").read_text())
    lookup={(r["seed"],r["case"],r["method"]):r for r in rows}
    keys=["alive_fraction","final_alive","burden","late_burden","late_deliveries","deliveries","blocked","revisions","choice_changes","evidence_updates","neural_updates","replay_updates","memory_updates"]
    rng=np.random.default_rng(PROTOCOL["bootstrap_seed"]);draws=rng.integers(0,32,(10000,32))
    means={};paired={};gates={}
    for case in PROTOCOL["cases"]:
        means[case]={};paired[case]={}
        for method in PROTOCOL["methods"]:
            rr=[lookup[seed,case,method] for seed in PROTOCOL["seeds"]]
            means[case][method]={k:float(np.mean([r[k] for r in rr])) for k in keys}
            paired[case][method]={}
            for key in ("alive_fraction","burden","late_burden","late_deliveries"):
                delta=np.array([float(lookup[seed,case,method][key])-float(lookup[seed,case,"prior"][key]) for seed in PROTOCOL["seeds"]])
                ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
                paired[case][method][key]=dict(mean=float(delta.mean()),ci95=ci,better=int((delta<0 if "burden" in key else delta>0).sum()),worse=int((delta>0 if "burden" in key else delta<0).sum()),equal=int((delta==0).sum()))
        c=paired[case]["combined"]
        gates[case]=dict(survival=c["alive_fraction"]["ci95"][0]>-.02,body_burden=c["burden"]["ci95"][1]<.02)
        gates[case]["cleared"]=all(gates[case].values())
    result=dict(runs=len(rows),ticks=len(rows)*1536,independent_seed_units=32,means=means,paired_vs_prior=paired,
                combined_acceptance=gates,all_cases_cleared=all(x["cleared"] for x in gates.values()),
                note="All contrasts exploratory/descriptive and conditional on supplied policies. Head prediction errors differ in scale for replay and bridge and are not pooled.")
    write_json(out/"SUMMARY.json",result);write_json(out/"ALL_METRICS.json",rows)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=ROOT/"validation/integrated_effects");p.add_argument("--workers",type=int,default=4);p.add_argument("--analyse-only",action="store_true");a=p.parse_args()
    out=a.out.resolve()
    if not a.analyse_only:
        out.mkdir(parents=True,exist_ok=False)
        for name in ("raw","metrics","checkpoints"):(out/name).mkdir()
        write_json(out/"PROTOCOL.json",PROTOCOL)
        locks=source_locks("workshop_bridge");locks[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(out/"FROZEN_BEFORE_RUN.json",dict(sources=locks,protocol_sha256=hashlib.sha256((out/"PROTOCOL.json").read_bytes()).hexdigest()))
        start=time.perf_counter();timing=[]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs=[pool.submit(run_seed,(seed,str(out))) for seed in PROTOCOL["seeds"]]
            for f in as_completed(jobs):
                result=f.result();timing.append(result);print(json.dumps(result),flush=True)
        write_json(out/"TIMING.json",dict(seconds=time.perf_counter()-start,workers=a.workers,details=timing))
    result=analyse(out);print(json.dumps(dict(runs=result["runs"],ticks=result["ticks"],combined_acceptance=result["combined_acceptance"])),flush=True)
if __name__=="__main__":main()
