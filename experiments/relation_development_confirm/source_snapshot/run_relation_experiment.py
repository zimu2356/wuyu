"""Paired whole-agent relation experiments; protocol precedes every run."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified import codec
from wuyu_unified.runtime import Session,source_locks,sha
from wuyu_unified.relation_world import RelationWorld,CASES
from run_whole_experiment import DTYPE as OLD_DTYPE,row_of as old_row

ROOT=Path(__file__).resolve().parents[1]
MODES=('A','R00','R10','R01','R11','RT')
DTYPE=np.dtype(OLD_DTYPE.descr+[
 ('movement_success','f4',(6,2)),('prior_tau','f4'),('movement_brier','f8'),
 ('relation_count','i4',(6,2)),('structural_denied','?'),('movement_noise','?'),
 ('entropy','f8')])

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(codec.jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def write_trace(path,rows):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,trace=rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return sha(path)

def row_of(e,b):
    r=np.zeros((),DTYPE);old=old_row(e,b)
    for name in OLD_DTYPE.names:r[name]=old[name]
    m=e['movement'];o=e['native_payload']['outcome']
    r['movement_success']=m['success_before'];r['prior_tau']=m['prior_tau']
    r['movement_brier']=-1. if m['update'] is None else m['update']['brier']
    r['relation_count']=b.relations.count
    r['structural_denied']=o['structural_denied'];r['movement_noise']=o['movement_noise']
    probability=e['decision']['final_probability']
    r['entropy']=-float(probability@np.log(np.maximum(probability,1e-12)))
    return r

def summarize(t,world):
    live=t['before'][:,14:16].min(1)>.001
    moving=live&(t['action']<2)
    mask=t['tick']>=1024
    first=np.flatnonzero(live&t['structural_denied'])
    post=np.arange(len(t))>=(int(first[0])+1 if len(first) else len(t))
    recovery=t['tick']>=world.recovery_tick
    p=t['before'][:,:6].argmax(1)
    movement_error=t['movement_brier'][t['movement_brier']>=0]
    result=dict(total_burden=float(t['burden'].mean()),challenge_burden=float(t['burden'][mask].mean()),
        late_burden=float(t['burden'][t['tick']>=min(2176,len(t)-1)].mean()),
        final_alive=bool(t['alive'][-1]),alive_fraction=float(t['alive'].mean()),
        live_steps=int(live.sum()),movement_attempts=int(moving.sum()),
        live_route_denials=int((live&t['route_denied']).sum()),
        live_structural_denials=int((live&t['structural_denied']).sum()),
        live_noise_failures=int((live&t['movement_noise']).sum()),
        live_blocked_takes=int((live&t['blocked_take']).sum()),
        live_action_changes=int((live&(t['action']!=t['base_choice'])).sum()),
        contradiction_observed=bool(len(first)),first_denial=None if not len(first) else int(first[0]),
        mean_movement_brier=float(movement_error.mean()) if len(movement_error) else 0.,
        mean_tau=float(t['prior_tau'][live].mean()) if live.any() else 1.,
        mean_entropy=float(t['entropy'][live].mean()) if live.any() else 0.,
        actor_updates=int(t['actor_updates'][-1]),mean_kl=float(t['actor_kl'][live].mean()) if live.any() else 0.,
        post_contradiction_denials=int((post&live&t['structural_denied']).sum()),
        post_contradiction_live_steps=int((post&live).sum()),
        recovery_live_steps=int((recovery&live).sum()),
        recovery_denials=int((recovery&live&t['structural_denied']).sum()),
        death_tick=None if t['alive'].all() else int(np.flatnonzero(~t['alive'])[0]))
    result['reopened_crossings']=0
    result['first_reopened_crossing_latency']=None
    if world.case=='r_reopen':
        edge=set(world.schedule[0]['blocked_edge'])
        crosses=np.array([int(row['action'])<2 and {int(row['before'][:6].argmax()),int(row['after'][:6].argmax())}==edge for row in t])
        found=np.flatnonzero(recovery&live&crosses)
        result['reopened_crossings']=int(len(found))
        result['first_reopened_crossing_latency']=None if not len(found) else int(found[0]-world.recovery_tick)
    return result

def diagnose(parent,outdir,label,pair,steps=96):
    original=parent.state(False);digest=codec.digest(original)
    b=parent.backend;model=b.relations;pos,act=pair
    # Roll back only content: count, time, support and all other agent states stay.
    original_q=model.q[pos,act].copy();restored_q=model.prior[pos,act].copy()
    delta=restored_q-original_q
    amount=float(original_q[pos])
    candidates=[]
    for site in range(6):
        if site==pos:continue
        for action in range(2):
            nxt=(site+(-1 if action==0 else 1))%6
            if model.q[site,action,nxt]+1e-12>=amount:
                candidates.append((abs(int(model.count[site,action])-int(model.count[pos,act])),site,action))
    other=min(candidates)[1:] if candidates else None
    codec.save(outdir/(label+'_origin.wuyu'),parent.state())
    branches={};info={}
    for kind in ('intact','related_old','matched_other','restored'):
        child=Session.from_state(deepcopy(original),verify_sources=False)
        cb=child.backend;cb.actor.c['rate']=0.
        if kind=='related_old':cb.relations.q[pos,act]=restored_q
        if kind=='matched_other' and other is not None:
            site,action=other;nxt=(site+(-1 if action==0 else 1))%6
            cb.relations.q[site,action,nxt]-=amount;cb.relations.q[site,action,site]+=amount
        if kind=='restored':
            cb.relations.q[pos,act]=restored_q;cb.relations.q[pos,act]=original_q
        perturb=float(np.linalg.norm(cb.relations.q-model.q))
        obs=cb.native.world.observation()[0];cb.native.individual.ingest(obs[None])
        first=cb.decision(obs)[3]
        rows=np.array([row_of(child.step(),cb) for _ in range(min(steps,cb.native.world.total-cb.native.world.t))],dtype=DTYPE)
        write_trace(outdir/(label+'_'+kind+'.npz'),rows);branches[kind]=rows
        info[kind]=dict(mean_burden=float(rows['burden'].mean()),final_alive=bool(rows['alive'][-1]),
                        first_probability=first.tolist(),perturbation_norm=perturb)
    assert np.array_equal(branches['intact'],branches['restored'])
    assert digest==codec.digest(parent.state(False))
    for kind in info:
        common=(branches[kind]['before'][:,14:16].min(1)>.001)&(branches['intact']['before'][:,14:16].min(1)>.001)
        info[kind]['common_live_action_changes']=int((common&(branches[kind]['action']!=branches['intact']['action'])).sum())
        info[kind]['first_total_variation']=float(np.abs(np.asarray(info[kind]['first_probability'])-np.asarray(info['intact']['first_probability'])).sum()/2)
    result=dict(tick=b.native.world.t,pair=list(pair),matched_other_pair=other,alive=bool(b.native.world.alive[0]),
        parent_unchanged=True,exact_restore=True,branches=info,
        limitation='Other-edge intervention is norm matched when available, not guaranteed functionally irrelevant in a six-node ring. No identified goal-content claim.')
    write_json(outdir/(label+'_meta.json'),result)
    return result

def run_life(job):
    directory,seed,init,mode,case,p=job;directory=Path(directory)
    label=f'{seed}_i{init}_{case}_{mode}';start=time.perf_counter()
    session=Session.create('whole',seed,mode,case,p['horizon'],init,
        action_configuration=p['action_configuration'],relation_configuration=p['relation_configuration'],split=p['split'])
    b=session.backend;world=b.native.world
    session.lineage=dict(origin=p['name'],protocol_digest=p['protocol_digest'],seed=seed,initialization=init)
    rows=np.zeros(p['horizon'],DTYPE);ds=[];fail_counts=np.zeros((6,2),int)
    for tick in range(p['horizon']):
        e=session.step();rows[tick]=row_of(e,b)
        if rows[tick]['structural_denied'] and rows[tick]['before'][14:16].min()>.001:
            pos=int(rows[tick]['before'][:6].argmax());action=int(rows[tick]['action']);fail_counts[pos,action]+=1
            if (not ds and fail_counts[pos,action]>=3 and seed in p['diagnostic_seeds']
                and mode in ('R10','R11') and case in ('r_block','r_reopen','r_combined')
                and tick+97<p['horizon'] and world.alive[0]):
                ds.append(diagnose(session,directory/'diagnostics',label+f'_t{tick+1}',(pos,action)))
        if tick+1==1024 and seed==p['seeds'][0]:
            session.save(directory/'checkpoints'/(label+'_t1024.wuyu'))
    trace_path=directory/'traces'/(label+'.npz');trace_sha=write_trace(trace_path,rows)
    result=summarize(rows,world)
    result.update(seed=seed,initialization=init,mode=mode,case=case,
        elapsed_seconds=time.perf_counter()-start,trace_sha256=trace_sha,
        raw_array_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        final_state_sha256=codec.digest(session.state(False)),diagnostics=ds,
        actor_parameters=0 if b.actor is None else b.actor.count,
        relation_parameters=int(b.relations.q.size),world_schedule=world.schedule,
        change_tick=world.change_tick,recovery_tick=world.recovery_tick,feasibility=world.feasibility)
    if seed==p['seeds'][0]:session.save(directory/'checkpoints'/(label+'_final.wuyu'))
    write_json(directory/'summaries'/(label+'.json'),result)
    return dict(label=label,loss=result['challenge_burden'],alive=result['final_alive'],seconds=result['elapsed_seconds'])

def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for name in ('traces','summaries','checkpoints','diagnostics','source_snapshot'):(out/name).mkdir()
    seeds=list(range(args.seed_start,args.seed_start+args.n))
    config=dict(initial_tau=args.tau,fixed_tau=args.fixed_tau,rate=.001)
    p=dict(name=out.name,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        split=args.split,seeds=seeds,initializations=args.initializations,modes=args.modes,cases=args.cases,
        horizon=args.horizon,action_configuration=config,
        relation_configuration=dict(rate=args.relation_rate,recovery=args.recovery,floor=.04),
        diagnostic_seeds=seeds[:args.diagnostic_n],
        primary='r_combined mean absorbing bodily deficit on ticks 1024..horizon-1',
        primary_contrast=args.primary_contrast,minimum_useful_improvement=args.min_effect,
        guardrails='r_stable total burden upper paired CI <= .02 and survival difference lower CI >= -.02',
        secondary=['factorial effects and interaction','R11-RT','R11-A','mechanism and reopening diagnostics'],
        clustering='world seed; mean over two fixed inherited initializations before bootstrap',
        stop='all declared lives, no exclusions, no outcome-dependent extension',
        precision_note=args.precision_note,
        priors='six observable site identities, ring/action geometry, EWMA update, age recovery and geometric retry-cost encoding are designed; graph weights inherited',
        diagnostics='first third observed structural movement failure per life; at most one checkpoint; actor rate frozen; other learners continue in all branches',
        sources=source_locks('whole'))
    for name in ('tools/run_relation_experiment.py','tools/run_whole_experiment.py','tools/analyze_relations.py','tools/audit_relations.py'):
        p['sources'][name]=sha(ROOT/name)
    for seed in seeds:
        for case in args.cases:RelationWorld(seed,args.horizon,case,args.split).validate_schedule()
    for src in (ROOT/'wuyu_unified').glob('*.py'):shutil.copy2(src,out/'source_snapshot'/src.name)
    for name in ('run_relation_experiment.py','analyze_relations.py','audit_relations.py'):
        shutil.copy2(ROOT/'tools'/name,out/'source_snapshot'/name)
    p['protocol_digest']=codec.digest(p);write_json(out/'protocol.json',p)
    print(json.dumps(dict(prepared=str(out),lives=len(seeds)*len(args.initializations)*len(args.modes)*len(args.cases),digest=p['protocol_digest'])))

def run(args):
    out=Path(args.out);p=json.loads((out/'protocol.json').read_text());check=deepcopy(p);digest=check.pop('protocol_digest')
    if codec.digest(check)!=digest:raise ValueError('protocol changed')
    for name,expected in p['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('source changed: '+name)
    marker=out/'RUN_START.json'
    if marker.exists():raise FileExistsError('create-only execution')
    write_json(marker,dict(protocol_digest=digest,workers=args.workers,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    jobs=[(str(out),s,i,m,c,p) for s in p['seeds'] for i in p['initializations'] for c in p['cases'] for m in p['modes']]
    start=time.perf_counter();done=0
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(run_life,j) for j in jobs]):
            done+=1;print(json.dumps(dict(done=done,total=len(jobs),result=future.result())),flush=True)
    write_json(out/'RUN_COMPLETE.json',dict(lives=done,wall_seconds=time.perf_counter()-start,protocol_digest=digest))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'));parser.add_argument('--out',required=True)
    parser.add_argument('--split',choices=('development','holdout'),default='development')
    parser.add_argument('--seed-start',type=int,default=1500000);parser.add_argument('--n',type=int,default=4)
    parser.add_argument('--initializations',nargs='+',type=int,default=[0,1])
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES));parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--horizon',type=int,default=2560);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--tau',type=float,default=.9);parser.add_argument('--fixed-tau',type=float,default=.9)
    parser.add_argument('--relation-rate',type=float,default=.45);parser.add_argument('--recovery',type=float,default=256.)
    parser.add_argument('--diagnostic-n',type=int,default=0);parser.add_argument('--min-effect',type=float,default=.05)
    parser.add_argument('--primary-contrast',choices=('R11-R00','R10-R00'),default='R11-R00')
    parser.add_argument('--precision-note',default='Development characterization, not confirmatory evidence.')
    args=parser.parse_args();(prepare if args.command=='prepare' else run)(args)

if __name__=='__main__':main()

