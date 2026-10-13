"""Explicit Whole 1.1 A/C checkpoint migration into the current mainline."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu_unified import codec
from wuyu_unified.runtime import Session,sha
from wuyu_unified.paths import ROOT
from wuyu_unified.whole import WholeAgent,RELATION_MODES,ACTION_INPUTS
from wuyu_unified.action_learning import ActionLearner
from wuyu_unified.relations import MovementRelations

def migrate(path,mode='R00'):
    if mode not in RELATION_MODES:raise ValueError('requires explicit relation condition')
    s=codec.load(path)
    if s.get('profile')!='whole' or s['backend'].get('condition') not in ('A','C'):
        raise ValueError('supported source: exact Whole 1.1 A or C; other cell conversions are not silently assumed')
    if not s.get('source_locks'):raise ValueError('source provenance required')
    for rel,expected in s['source_locks'].items():
        file=ROOT/rel
        if not file.resolve().is_relative_to(ROOT.resolve()):raise ValueError('invalid source path')
        if rel.startswith('wuyu_unified/'):file=ROOT/'experiments/relation_start_snapshot'/rel
        if not file.is_file() or sha(file)!=expected:raise ValueError('unsupported source revision: '+rel)
    b=deepcopy(s['backend']);ind=b['native']['individual']
    old=b['actor']
    config=dict(old['c']) if old else {}
    config.update(allocate_gate=True,initial_tau=.9,fixed_tau=.9,
        prior_gate='learned' if mode in ('R01','R11') else 'fixed' if mode=='RT' else 'off')
    actor=ActionLearner(ACTION_INPUTS,ind['seed']*59+ind['initialization']*100003,'C',config)
    if old:
        for group in ('p','m','v'):
            target=getattr(actor,group)
            for key,value in old[group].items():
                if key=='O':target[key][:,:8]=value
                elif key=='bo':target[key][:8]=value
                else:target[key]=value.copy()
        for key in ('h','updates','transitions','last_update'):setattr(actor,key,deepcopy(old[key]))
        actor.buffer=deepcopy(old['buffer'])
        for q in actor.buffer:
            q['out']=np.concatenate([q['out'],[actor.p['bo'][8]]]);q['prior_tau']=1.
    relations=MovementRelations()
    replayed=0
    for item in ind['memory']['memory']:
        action=int(item['action'])
        if action not in (0,1):continue
        pos,npos,tick=int(item['pos']),int(item['npos']),int(item['t'])
        predicted=relations.distribution(tick)[pos,action]
        observed=np.eye(6)[npos];brier=float(np.sum((predicted-observed)**2))
        relations.q[pos,action]=(1-relations.rate)*predicted+relations.rate*observed
        relations.count[pos,action]+=1;relations.last[pos,action]=tick
        relations.brier_sum[pos,action]+=brier
        relations.events.append(dict(tick=tick,position=pos,action=action,next_position=npos,predicted=predicted.copy(),brier=brier))
        relations.events=relations.events[-128:];replayed+=1
    lineage=dict(origin='explicit Whole 1.1 migration',source_checkpoint_sha256=sha(path),
        procedure_sha256=sha(__file__),source_condition=b['condition'],condition=mode,
        body_inventory_skills_native_rng_predictors_preserved=True,
        old_actor_preserved=old is not None,new_gate_parameters=True,
        recent_real_movement_transitions_replayed=replayed,
        relation_history_truncated=True,stochastic_future_equivalence_promised=False,
        note='Only retained actual position transitions rebuild relation content; body labels are not fabricated. Source A initializes a new C actor. Source C retains its optimizer and unfinished buffer.')
    b.update(condition=mode,actor=actor.state(),relations=relations.state(),relation_read=mode in ('R10','R11','RT'),migration=lineage)
    result=Session('whole',WholeAgent.restore(b),s['event_index'],lineage)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--mode',choices=RELATION_MODES,default='R00')
    parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    session=migrate(args.checkpoint,args.mode);session.save(out)
    print(json.dumps(codec.jsonable(session.lineage),indent=2))

