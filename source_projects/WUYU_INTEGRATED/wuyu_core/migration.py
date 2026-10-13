"""Explicit migration of an 08 individual; other worlds retain legacy profiles."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .legacy import module, ROOT
from .runtime import Session, Individual, Network
from .memory import ExperienceMemory
from .state import read_tree
from .world import World


def migrate_08(path,index=0,mode='prior'):
    path=Path(path);state=read_tree(path)
    if 'world' not in state or 'individual' not in state:raise ValueError('expected an original WUYU_08 checkpoint')
    w=World.from_state(state['world']);old=state['individual']
    if not 0<=index<w.n:raise IndexError('individual out of range')
    inits=json.loads((ROOT/'legacy/WUYU_08/CONFIG.json').read_text())['initializations']
    if old['initialization'] not in inits:raise ValueError('unrecognised source initialization')
    init=inits.index(old['initialization']);a=Individual(Network.from_state(old['network']),int(old['seeds'][index]),init,mode)
    a.h=old['h'][index:index+1].copy();a.prev=old['prev'][index:index+1].copy();a.tick=old['tick']
    a.memory=ExperienceMemory.from_state(old['memories'][index])
    # New predictor starts with no experiences; do not fabricate missing history.
    # The old batch-wide RNG cannot be split into equivalent independent streams.
    lineage=dict(source_version='WUYU_08',source_filename=path.name,
                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_individual=index,
                 source_event=int(w.t),preserved=['physical_world','graph_weights','recurrent_state','previous_feedback','experience_memory'],
                 changed=['private_policy_random_stream','new_untrained_consequence_predictor'],
                 stochastic_future_equivalent=False)
    return Session(w.one(index),a,lineage)
