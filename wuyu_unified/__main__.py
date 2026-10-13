import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import argparse
import json
from pathlib import Path
from . import codec
from .paths import ROOT,SOURCES
from .registry import DESCRIPTIONS,describe
from .runtime import Session

def emit(value):print(json.dumps(codec.jsonable(value),ensure_ascii=False,indent=2,allow_nan=False))

def new_output(path):
    path=Path(path).resolve()
    if path==SOURCES.resolve() or path.is_relative_to(SOURCES.resolve()):
        raise ValueError("output must be outside immutable source_projects")
    path.mkdir(parents=True,exist_ok=False)
    return path

def run_to_output(session,steps,output):
    if steps<0:raise ValueError("steps must be nonnegative")
    output=new_output(output);rows=[];count=0
    with (output/"events.jsonl").open("x",encoding="utf-8") as trace:
        for _ in range(steps):
            try:event=session.step()
            except StopIteration:break
            trace.write(codec.canonical(codec.jsonable(event))+"\n")
            count+=1
            rows.append(dict(event_index=event["event_index"],action=codec.jsonable(event.get("action")),clock=codec.jsonable(event.get("clock_after"))))
            rows=rows[-3:]
    session.save(output/"state.wuyu")
    summary=dict(profile=session.profile,executed_events=count,total_events=session.event_index,
                 last=rows,checkpoint=str(output/"state.wuyu"),lineage=session.lineage,
                 note="Explicit finite execution; step meanings depend on profile; no combined scientific score")
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description="物欲 1–8 与后续实验的统一运行和证据入口")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("list")
    status=sub.add_parser("data-status")
    audit=sub.add_parser("verify-sources");audit.add_argument("--full",action="store_true")
    inspect=sub.add_parser("inspect");inspect.add_argument("checkpoint")
    run=sub.add_parser("run");run.add_argument("--profile",choices=DESCRIPTIONS,default="whole")
    run.add_argument("--seed",type=int,default=990001);run.add_argument("--mode");run.add_argument("--case")
    run.add_argument("--horizon",type=int);run.add_argument("--initialization",type=int,choices=range(5),default=0)
    resume=sub.add_parser("resume");resume.add_argument("checkpoint")
    fork=sub.add_parser("fork");fork.add_argument("checkpoint")
    fork.add_argument("--intervention",choices=("erase_model","freeze_learning","suppress_work_state"));fork.add_argument("--action",type=int)
    evaluation=sub.add_parser("evaluate-history");evaluation.add_argument("checkpoint")
    evaluation.add_argument("--case",default="quiet");evaluation.add_argument("--variant",default="learned")
    native=sub.add_parser("import-native");native.add_argument("--profile",choices=DESCRIPTIONS,required=True);native.add_argument("checkpoint")
    migration=sub.add_parser("migrate08");migration.add_argument("checkpoint")
    migration.add_argument("--individual",type=int,default=0);migration.add_argument("--mode",choices=("prior","online","replay","shadow"),default="prior")
    whole_migration=sub.add_parser('migrate-whole');whole_migration.add_argument('checkpoint')
    whole_migration.add_argument('--mode',choices=('A','B','C','D'),default='A')
    for command in (run,resume,fork,evaluation,native,migration,whole_migration):
        command.add_argument("--steps",type=int,default=64);command.add_argument("--out",required=True)
    rep=sub.add_parser("reproduce");rep.add_argument("--profile",choices=DESCRIPTIONS,required=True)
    rep.add_argument("--execute",action="store_true");rep.add_argument("--out")
    args=p.parse_args()
    try:
        if args.command=="list":emit([describe(x) for x in DESCRIPTIONS]);return
        if args.command=="data-status":
            from .audit import data_status
            emit(data_status());return
        if args.command=="verify-sources":
            from .audit import verify_sources
            result=verify_sources(args.full);emit(result)
            if not result["passed"]:raise SystemExit(1)
            return
        if args.command=="inspect":
            s=Session.load(args.checkpoint)
            emit(dict(profile=s.profile,event_index=s.event_index,semantics=describe(s.profile),lineage=s.lineage,
                      state_sha256=codec.digest(s.state(with_locks=False))));return
        if args.command=="reproduce":
            from .reproduce import execute,plan
            if args.execute and not args.out:p.error("--execute requires a new --out directory")
            emit(execute(args.profile,args.out) if args.execute else plan(args.profile));return
        if args.command=="run":
            s=Session.create(args.profile,args.seed,args.mode,args.case,args.horizon,args.initialization)
        elif args.command=="resume":s=Session.load(args.checkpoint)
        elif args.command=="fork":s=Session.load(args.checkpoint).fork(args.intervention,args.action)
        elif args.command=="evaluate-history":s=Session.load(args.checkpoint).evaluation(args.case,args.variant)
        elif args.command=="import-native":
            from .migration import import_native
            s=import_native(args.profile,args.checkpoint)
        elif args.command=='migrate-whole':
            from .whole_migration import migrate_whole
            s=migrate_whole(args.checkpoint,args.mode)
        else:
            from .migration import migrate08
            s=migrate08(args.checkpoint,args.individual,args.mode)
        emit(run_to_output(s,args.steps,args.out))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        p.exit(2,f"error: {exc}\n")

if __name__=="__main__":main()
