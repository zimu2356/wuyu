"""Continuous relation tests on the original body and seven primitive actions.

The agent receives the ordinary observation only. Noise is indexed by world
seed and physical tick, independent of policy RNG consumption.
"""
from copy import deepcopy
import numpy as np
from .whole_world import WholeWorld, HORIZON

CASES = ('r_stable','r_block','r_noise','r_reopen','r_relocate','r_combined')
SCHEMA = 'whole-relation-world/1'


class RelationWorld(WholeWorld):
    def __init__(self, seed, total=HORIZON, case='r_combined', split='development'):
        if case not in CASES: raise ValueError(case)
        super().__init__(seed,total,'stable',split)
        self.relation_world_schema=SCHEMA; self.case=case
        self.regime_override=0 if case=='r_stable' else 2
        self.noise_start=1024; self.noise_probability=.12 if case=='r_noise' else 0.
        self.noise_draws=np.random.default_rng(int(seed)+890033).random(total)
        self.schedule=self.relation_schedule(seed,split)
        self.feasibility=self.validate_schedule()

    def relation_schedule(self, seed, split):
        r=np.random.default_rng(int(seed)+982741)
        edge=int(r.integers(6)); start=int(r.integers(1120,1377))
        later=int(r.integers(1920,2113))
        self.change_tick=start; self.recovery_tick=later
        if self.case in ('r_stable','r_noise'):return []
        if self.case=='r_combined':
            events=super().make_schedule(seed,split)
            main=[e for e in events if e['tick']>=1024]
            events=[e for e in events if e['tick']<1024]
            ticks=[1024,1408+int(r.integers(-96,97)),1792+int(r.integers(-96,97)),2176+int(r.integers(-96,97))]
            # Route and interruption order vary, with dependency structure kept feasible.
            if bool(r.integers(2)):main[1],main[2]=main[2],main[1]
            for ev,tick in zip(main,ticks):ev['tick']=tick
            events+=main
            self.change_tick=next(e['tick'] for e in main if 'blocked_edge' in e)
            return sorted(events,key=lambda e:e['tick'])
        events=[dict(tick=start,kind='route_closed',blocked_edge=[edge,(edge+1)%6])]
        if self.case=='r_reopen':events.append(dict(tick=later,kind='route_reopened',open_route=True))
        if self.case=='r_relocate':
            other=(edge+int(r.integers(1,6)))%6
            events.append(dict(tick=later,kind='route_relocated',blocked_edge=[other,(other+1)%6]))
        return events

    def step(self, action):
        tick=self.t; oldpos=int(self.pos[0]); was=bool(self.alive[0])
        out=super().step(action)
        noisy=bool(was and out['allowed'][0] and int(action[0])<2 and tick>=self.noise_start
                   and self.noise_draws[tick]<self.noise_probability)
        structural=bool(out['route_denied'] and was and out['allowed'][0])
        if noisy:self.pos[0]=oldpos
        for ev in self.schedule:
            if ev['tick']==self.t and ev.get('open_route'):self.blocked_edge=None
        out.update(position=self.pos.copy(),route_denied=bool(structural or noisy),
                   structural_denied=structural,movement_noise=noisy)
        return out

    @classmethod
    def from_state(cls, state):
        if state.get('relation_world_schema')!=SCHEMA:raise ValueError('incompatible relation world')
        obj=cls.__new__(cls);obj.__dict__.update(deepcopy(state));return obj
