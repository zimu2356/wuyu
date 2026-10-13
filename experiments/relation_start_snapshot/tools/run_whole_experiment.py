"""Frozen, paired whole-life runs. Run from the project root.

Development and holdout use disjoint world seeds. Metadata and ALL source
hashes are written before any life is run; --run requires a prepared protocol.
No world is removed based on an agent's outcome. NPZ files contain primitive
observations/actions, prediction/probability/state logs and physical debits.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified import codec
from wuyu_unified.runtime import Session,source_locks,sha
from wuyu_unified.whole import WholeAgent
from wuyu_unified.whole_world import WholeWorld,BOUNDARIES,HORIZON

DTYPE=np.dtype([
 ('tick','i4'),('action','i1'),('base_choice','i1'),('native_choice','i1'),('alive','?'),
 ('before','f8',(21,)),('after','f8',(21,)),('burden','f8'),
 ('prior','f4',(7,)),('base_probability','f4',(7,)),('probability','f4',(7,)),
 ('ridge_predictions','f4',(7,6)),('neural_predictions','f4',(7,6)),
 ('action_residual','f4',(7,)),('state','f4',(16,)),('write_gate','f4',(16,)),('erase_reset_gate','f4',(16,)),
 ('shock','f8',(2,)),('removed_yield','f8',(2,)),('target','f8',(6,)),
 ('blocked_take','?'),('route_denied','?'),('product','i1'),('revision','?'),
 ('prediction_mse','f8'),('actor_updates','i4'),('neural_updates','i4'),('memory_updates','i4'),
 ('actor_kl','f4'),('physical_cost','f8',(2,)),('evidence_before','f4',(7,3)),
 ('error_history_before','f4',(7,2)),('value_prediction','f4')])

def row_of(e,b):
    r=np.zeros((),DTYPE);d=e['decision'];o=e['native_payload']['outcome'];before=e['before']
    for name,key in [('tick','clock_before'),('action','action'),('alive','alive'),('before','before'),('after','after'),('burden','body_burden')]:r[name]=e[key]
    r['base_choice']=d['base_choice'];r['native_choice']=d['native_choice']
    r['prior']=d['prior_probability'];r['base_probability']=d['probability'];r['probability']=d['final_probability']
    r['ridge_predictions']=d['predictions'];r['neural_predictions']=d['neural_predictions'];r['action_residual']=d['residual']
    if d['work_state'] is not None:
        r['state']=d['work_state'];r['write_gate']=d['write_gate'];r['erase_reset_gate']=d['erase_or_reset_gate']
    r['shock']=o['shock'][0];r['removed_yield']=o.get('removed_yield',np.zeros((1,2)))[0]
    r['target']=e['learning']['target'];r['revision']=e['learning']['revision']
    req=int(before[9:12].argmax())-1
    r['blocked_take']=bool(e['action']==5 and before[6:9].argmax()==2 and req>=0 and before[12+req]<.5)
    r['route_denied']=o.get('route_denied',False);r['product']=int(o['product'][0])
    r['prediction_mse']=e['native_payload']['prediction_mse']
    r['actor_updates']=0 if b.actor is None else b.actor.updates
    r['neural_updates']=b.native.individual.predictor.updates;r['memory_updates']=b.native.individual.memory.updates
    pp=np.maximum(d['final_probability'],1e-12);bp=np.maximum(d['probability'],1e-12)
    r['actor_kl']=np.sum(pp*np.log(pp/bp))
    r['evidence_before']=np.array([[e['age'],e['n'],-1. if e['coverage'] is None else e['coverage']] for e in e['evidence_before']])
    r['error_history_before']=d['error_history_before'];r['value_prediction']=0. if d['value_prediction'] is None else d['value_prediction']
    alive_before=bool(o['was'][0]);phase=int(o['phase'][0]);costs=np.array([.003,.003,.004,.004,.004,.004,.0018])
    r['physical_cost']=[costs[e['action']]*(.55 if phase>=2 else 1.)*alive_before,(.0018 if phase>=2 else .0007)*alive_before]
    return r

def summarize(trace):
    rows=trace;challenge=rows['tick']>=1024
    product=rows['product'];alive=rows['alive']
    summary=dict(total_burden=float(rows['burden'].mean()),
        challenge_burden=float(rows['burden'][challenge].mean()) if challenge.any() else None,
        final_alive=bool(alive[-1]),alive_fraction=float(alive.mean()),
        death_tick=None if alive.all() else int(rows['tick'][np.flatnonzero(~alive)[0]]),
        products=int(np.sum(product<4)),blocked_takes=int(rows['blocked_take'].sum()),
        route_denials=int(rows['route_denied'].sum()),lost_yield_takes=int(np.sum(rows['removed_yield'].sum(1)>0)),
        changes_vs_base=int(np.sum(rows['action']!=rows['base_choice'])),
        changes_vs_native=int(np.sum(rows['action']!=rows['native_choice'])),
        actor_updates=int(rows['actor_updates'][-1]),neural_updates=int(rows['neural_updates'][-1]),
        memory_updates=int(rows['memory_updates'][-1]),revisions=int(rows['revision'].sum()),
        mean_kl=float(rows['actor_kl'].mean()),physical_cost=rows['physical_cost'].sum(0).tolist(),
        mean_prediction_mse=float(rows['prediction_mse'].mean()))
    role_counts=[]
    for token in (0,1):
        observed=(rows['tick']>=384)&(rows['tick']<896)&(rows['before'][:,9:12].argmax(1)==2-token)
        role_counts.append(int(np.sum(observed&(product==token+2))))
    summary['actual_dependent_token_acquisitions']=role_counts
    summary['both_dependent_token_roles_experienced']=bool(min(role_counts)>0)
    waste=np.zeros(len(rows),bool)
    for k in (0,1):
        gain=rows['after'][:,14+k]-rows['before'][:,14+k]+rows['physical_cost'][:,k]+rows['shock'][:,k]
        waste|=(product==k)&(gain<.02)&(rows['removed_yield'].sum(1)==0)
    summary['low_benefit_body_productions']=int(waste.sum())
    windows=[]
    for start in BOUNDARIES:
        mask=(rows['tick']>=start)&(rows['tick']<start+384);part=rows[mask]
        if not len(part):continue
        blocked=np.flatnonzero(part['blocked_take']);completion=False;latency=None
        if len(blocked):
            first=int(blocked[0]);site=int(part['before'][first,:6].argmax())
            matches=np.flatnonzero((part['tick']>part['tick'][first])&
                      (part['before'][:,:6].argmax(1)==site)&(part['product']<4))
            if len(matches):completion=True;latency=int(part['tick'][matches[0]]-part['tick'][first])
        windows.append(dict(start=start,body_burden=float(part['burden'].mean()),
             alive_at_start=bool(part['before'][0,14:16].min()>.001),
             blocked_opportunity_observed=bool(len(blocked)),returned_completion=completion,return_latency=latency,
             # Descriptive only: fixed times need not interrupt an active pursuit.
             partial_device_at_start=bool(part['before'][0,6:9].argmax()>0),
             blocked_takes=int(part['blocked_take'].sum())))
    summary['windows']=windows
    summary['completed_return_windows']=sum(x['returned_completion'] for x in windows)
    summary['observed_blocked_windows']=sum(x['blocked_opportunity_observed'] for x in windows)
    return summary

def diagnose(parent,history,outdir,label,steps=64):
    original=parent.state(False);parent_hash=codec.digest(original);b=parent.backend
    obs=b.native.world.observation()[0];tick=b.native.world.t
    candidates=[q for q in history if q['tick']<=tick-8 and q['obs'][:6].argmax()==obs[:6].argmax()
                and q['obs'][6:9].argmax()==obs[6:9].argmax()]
    donor=min(candidates,key=lambda q:np.linalg.norm(q['obs']-obs)) if candidates else None
    branches={};meta={}
    state_path=outdir/(label+'_origin.wuyu');codec.save(state_path,parent.state())
    for kind in ('intact','suppressed','matched_history','restored'):
        child=Session.from_state(deepcopy(original),verify_sources=False);a=child.backend.actor
        a.c['rate']=0. # Common diagnostic control: fixed action parameters.
        if kind=='suppressed':child.backend.suppress_until=tick+32
        if kind=='matched_history' and donor is not None:a.h=donor['state'].copy()
        if kind=='restored':
            saved=a.h.copy();a.h.fill(0);a.h=saved.copy()
        trajectory=[]
        for _ in range(min(steps,child.backend.native.world.total-tick)):
            event=child.step();trajectory.append(row_of(event,child.backend))
        rows=np.array(trajectory,dtype=DTYPE);branches[kind]=rows
        np.savez_compressed(outdir/(label+'_'+kind+'.npz'),trace=rows)
        meta[kind]=dict(mean_burden=float(rows['burden'].mean()),final_alive=bool(rows['alive'][-1]),
                        params_frozen=True)
    assert np.array_equal(branches['intact'],branches['restored'])
    for k in meta:meta[k]['action_differences_vs_intact']=int(np.sum(branches[k]['action']!=branches['intact']['action']))
    assert parent_hash==codec.digest(parent.state(False))
    result=dict(tick=tick,alive=bool(b.native.world.alive[0]),parent_state_sha256=parent_hash,
         exact_restore=True,parent_unchanged=True,branches=meta,
         donor_available=donor is not None,donor_tick=None if donor is None else donor['tick'],
         donor_observation_distance=None if donor is None else float(np.linalg.norm(donor['obs']-obs)),
         donor_state_distance=None if donor is None else float(np.linalg.norm(donor['state']-b.actor.h)),
         suppression_distance=float(np.linalg.norm(b.actor.h)),
         limitation='Historical donor is real and locally matched, but not proven irrelevant or equal-norm. This is a state-use probe, not selective goal-content evidence.')
    (outdir/(label+'_meta.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

def run_life(job):
    outpath,seed,initialization,mode,case,protocol=job;outpath=Path(outpath)
    start=time.perf_counter();label=f'{seed}_i{initialization}_{case}_{mode}'
    backend=WholeAgent.create(seed,mode,protocol['horizon'],initialization,case,protocol['split'],protocol['action_configuration'])
    session=Session('whole',backend,lineage=dict(origin=protocol['name'],seed=seed,initialization=initialization,
                                               protocol_sha256=protocol['protocol_digest']))
    feasibility=deepcopy(backend.native.world.feasibility)
    rows=np.zeros(protocol['horizon'],DTYPE);diagnostics=[];history=[]
    for tick in range(protocol['horizon']):
        if (case=='challenge' and mode in ('C','D') and seed in protocol['diagnostic_seeds']
                and tick in protocol['diagnostic_ticks']):
            diagnostics.append(diagnose(session,history,outpath/'diagnostics',label+f'_t{tick}'))
        if backend.actor is not None:
            history.append(dict(tick=tick,obs=backend.native.world.observation()[0].copy(),state=backend.actor.h.copy()))
            history=history[-128:]
        e=session.step();rows[tick]=row_of(e,backend)
        if tick+1==1024 and seed==protocol['seeds'][0]:session.save(outpath/'checkpoints'/(label+'_t1024.wuyu'))
    np.savez_compressed(outpath/'traces'/(label+'.npz'),trace=rows)
    summary=summarize(rows);summary.update(seed=seed,initialization=initialization,mode=mode,case=case,
        elapsed_seconds=time.perf_counter()-start,actor_parameters=0 if backend.actor is None else backend.actor.count,
        feasibility=feasibility,diagnostics=diagnostics,trace_sha256=sha(outpath/'traces'/(label+'.npz')),
        final_state_sha256=codec.digest(session.state(False)))
    if seed==protocol['seeds'][0]:session.save(outpath/'checkpoints'/(label+'_final.wuyu'))
    (outpath/'summaries'/(label+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return {k:summary[k] for k in ('seed','initialization','mode','case','final_alive','challenge_burden','elapsed_seconds')}

def prepare(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for name in ('traces','summaries','checkpoints','diagnostics'):(out/name).mkdir()
    seeds=list(range(args.seed_start,args.seed_start+args.n))
    protocol=dict(name=out.name,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        split=args.split,seeds=seeds,initializations=args.initializations,modes=list('ABCD'),cases=args.cases,
        horizon=HORIZON,action_configuration=dict(rate=args.rate),
        diagnostic_seeds=seeds[:args.diagnostic_n],diagnostic_ticks=[1408,1792],
        primary='mean absorbing body deficit in ticks 1024..2559 of challenge lives',
        primary_contrast='D minus A; negative is better',minimum_useful_improvement=.01,
        secondary_contrasts=['B-A','C-A','D-B','D-C'],
        guardrails='stable total deficit increase upper paired CI <= .02; survival difference lower CI >= -.02',
        clustering='paired world seed, average over crossed fixed graph initializations; bootstrap worlds, never steps or branches',
        stop='all predeclared seeds x initializations x modes x cases; no outcome-dependent extension or exclusions',
        reward='negative actual post-transition sum((1-body)^2); dead=2 with absorbing discounted terminal value',
        precision_note=args.precision_note,
        role_coverage='generator exposes both token dependent roles before challenge; actual experience measured, not assumed',
        diagnostic_limit='No semantically identified goal vector exists; donor control is not guaranteed equal norm or irrelevant.',
        trace_precision='physical observations/targets/debits float64; predictions/probabilities/recurrent probes float32',
        sources=source_locks('whole'))
    protocol['sources']['tools/run_whole_experiment.py']=sha(__file__)
    for name in ('tools/analyze_whole.py','tools/audit_whole_results.py'):
        protocol['sources'][name]=sha(Path(__file__).resolve().parents[1]/name)
    # Pre-check all generated worlds BEFORE outcomes; no outcome-based filtering.
    protocol['feasibility_checked_worlds']=len(seeds)*len(args.cases)
    for seed in seeds:
        for case in args.cases:WholeWorld(seed,HORIZON,case,args.split).validate_schedule()
    protocol['protocol_digest']=codec.digest(protocol)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(prepared=str(out),digest=protocol['protocol_digest'],lives=len(seeds)*len(args.initializations)*4*len(args.cases))))

def run(args):
    out=Path(args.out);p=json.loads((out/'protocol.json').read_text());check=deepcopy(p);digest=check.pop('protocol_digest')
    if codec.digest(check)!=digest:raise ValueError('protocol altered')
    for name,expected in p['sources'].items():
        if sha(Path(__file__).resolve().parents[1]/name)!=expected:raise ValueError('source altered since freeze: '+name)
    marker=out/'RUN_START.json'
    if marker.exists():raise FileExistsError('execution is create-only; prepare a new protocol to rerun')
    marker.write_text(json.dumps(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_digest=digest,workers=args.workers))+'\n')
    jobs=[(str(out),s,i,m,c,p) for s in p['seeds'] for i in p['initializations'] for c in p['cases'] for m in p['modes']]
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(args.workers) as ex:
        futures=[ex.submit(run_life,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result());print(json.dumps(dict(done=len(results),total=len(jobs),life=results[-1])),flush=True)
    (out/'RUN_COMPLETE.json').write_text(json.dumps(dict(lives=len(results),wall_seconds=time.perf_counter()-start,protocol_digest=digest),indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'));parser.add_argument('--out',required=True)
    parser.add_argument('--split',choices=('development','holdout'),default='development');parser.add_argument('--seed-start',type=int,default=1300000)
    parser.add_argument('--n',type=int,default=4);parser.add_argument('--initializations',type=int,nargs='+',default=[0,1]);parser.add_argument('--cases',nargs='+',choices=('challenge','stable'),default=['challenge','stable'])
    parser.add_argument('--rate',type=float,default=.001);parser.add_argument('--diagnostic-n',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--precision-note',default='Development only; estimates will determine a fixed feasible holdout size.');args=parser.parse_args()
    (prepare if args.command=='prepare' else run)(args)

if __name__=='__main__':main()
