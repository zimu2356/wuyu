"""Audited, explicit migration from the shipped v1 combined bridge.

Body, inventory, clock, graph policy/h, associative skills/history and action
RNG are kept. Ridge samples can be exactly relabeled. Neural target weights
cannot: reinitialize them, replay the real retained reservoir once, and rebuild
live h from the retained four-input prefix. This is a declared approximation.
"""
from copy import deepcopy
import importlib
import numpy as np
from . import codec
from .paths import ROOT,source_module
from .runtime import Session,sha
from .whole import WholeAgent,TARGET_SCHEMA,ACTION_FEATURE_SCHEMA,ACTION_INPUTS
from .action_learning import ActionLearner
from .evidence_model import WindowedEvidenceModel

def migrate_whole(path,mode='A'):
    s=codec.load(path)
    if s.get('profile')!='workshop_bridge' or s['backend'].get('mode')!='combined':
        raise ValueError('explicit migration currently accepts a complete combined workshop_bridge checkpoint')
    locks=s.get('source_locks')
    if not locks:raise ValueError('missing source provenance')
    for rel,expected in locks.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT.resolve()):raise ValueError('invalid source path')
        if rel.startswith('wuyu_unified/'):
            p=ROOT/'reference_runtime_v1'/rel.split('/',1)[1]
        if not p.is_file() or sha(p)!=expected:raise ValueError(f'unsupported source revision: {rel}')
    source_module('integrated');b=deepcopy(s['backend']);native=b['native'];ind=native['individual']
    models=[];relabeled=0
    for action,old in enumerate(b['models']):
        m=WindowedEvidenceModel.restore(old)
        if action<2:
            for sample in m.samples:sample['y'][4]=0.;relabeled+=1
        m.rebuild();m.coef=np.linalg.solve(m.gram+m.c['ridge']*np.eye(m.inputs),m.rhs)
        models.append(m.state())
    oldpred=ind['predictor'];Predictor=importlib.import_module('wuyu_core.predictor').Predictor
    pred=Predictor(ind['seed']*17+ind['initialization']*100003)
    memories=deepcopy(oldpred['memory'])
    for item in sorted(memories,key=lambda x:x['event']):
        if int(np.argmax(item['xs'][-1,-7:]))<2:item['y'][4]=0.
        pred.learn(item['xs'],item['y'])
    pred.memory=memories;pred.history=deepcopy(oldpred['history']);pred.seen=oldpred['seen']
    pred.rng.bit_generator.state=deepcopy(oldpred['rng'])
    if pred.history:pred.h=pred.sequence(pred.history)[1][-1].copy()
    ind['predictor']=pred.state()
    b.update(models=models,target_schema=TARGET_SCHEMA,condition=mode,action_feature_schema=ACTION_FEATURE_SCHEMA,
             actor=None if mode=='A' else ActionLearner(ACTION_INPUTS,ind['seed']*59+ind['initialization']*100003,mode).state(),
             errors=np.zeros((7,2)),error_counts=np.zeros(7,int),suppress_until=-1,last_action_learning=None,
             migration=dict(source_sha256=sha(path),exact_ridge_relabels=relabeled,
                            neural_reinitialized=True,real_reservoir_replayed=len(memories),
                            live_neural_state_rebuilt_from_prefix=len(pred.history),
                            old_neural_parameter_updates=oldpred['updates'],
                            action_adapter='zero initialization',stochastic_future_equivalent=False,
                            physical_and_old_skill_state_preserved=True))
    return Session('whole',WholeAgent.restore(b),s['event_index'],lineage=b['migration'])
