"""Continuous physical challenges. Hidden schedules never enter agent features.

This extends the original world, rather than replacing its action/body/skill
semantics. A blocked ring edge has a five-edge alternative. One disabled
resource device always has another producer. Dependency graphs are acyclic.
"""
from copy import deepcopy
import importlib
import numpy as np
from .paths import source_module
source_module('integrated')
BaseWorld=importlib.import_module('wuyu_core.world').World

WORLD_SCHEMA='whole-workshop-challenges/1'
BOUNDARIES=(1024,1408,1792,2176)
HORIZON=2560

class WholeWorld(BaseWorld):
    def __init__(self,seed,total=HORIZON,case='challenge',split='development'):
        if case not in ('challenge','stable'): raise ValueError(case)
        if split not in ('development','holdout'): raise ValueError(split)
        super().__init__([seed],total,shocks=False)
        self.whole_schema=WORLD_SCHEMA;self.case=case;self.split=split
        self.regime_override=2 if case=='challenge' else 0
        self.req_override=np.full((1,6),-1,int)
        self.blocked_edge=None;self.inactive_site=None;self.event_log=[]
        self.schedule=self.make_schedule(seed,split) if case=='challenge' else []
        self.feasibility=self.validate_schedule()

    def make_schedule(self,seed,split):
        r=np.random.default_rng(int(seed)+884321)
        products=self.products[0];token=[int(np.flatnonzero(products==k+2)[0]) for k in range(2)]
        bodies=[np.flatnonzero(products==k).tolist() for k in range(2)]
        # Fixed development role coverage; holdout combines the same role types
        # with a different body-axis/token mapping. Roles, recipes, positions
        # and event truths are never provided as learning labels.
        order=int(r.integers(2));axis=int(r.integers(2));selected=int(r.choice(bodies[axis]))
        def dep(root,body_mapping):
            req=np.full(6,-1,int);req[token[root]]=1-root
            for k in range(2):req[bodies[k]]=body_mapping[k]
            return req.tolist()
        curriculum0=dep(0,[0,1]);curriculum1=dep(1,[0,1])
        mapping=[order,1-order] if split=='development' else [1-order,order]
        edge=int(r.integers(6));replacement=((self.codes[0,selected]+r.integers(1,3,2))%3).tolist()
        events=[dict(tick=192,kind='need_opportunity',shock=[axis,.22]),
                dict(tick=320,kind='need_opportunity',shock=[1-axis,.22]),
                dict(tick=384,kind='role_exposure_0',requirements=curriculum0),
                dict(tick=512,kind='need_opportunity',shock=[0,.24]),
                dict(tick=640,kind='role_exposure_1',requirements=curriculum1),
                dict(tick=768,kind='need_opportunity',shock=[1,.24]),
                dict(tick=896,kind='familiar_recovery',requirements=[-1]*6),
                dict(tick=1024,kind='prerequisite',requirements=dep(order,mapping),shock=[axis,.30]),
                dict(tick=1408,kind='alternative_route',blocked_edge=[edge,(edge+1)%6],
                     recipe_site=selected,recipe=replacement,shock=[axis,.24]),
                dict(tick=1792,kind='interruption',shock=[1-axis,.40]),
                dict(tick=2176,kind='benefit_removed',inactive_site=selected,shock=[axis,.22])]
        return events

    def validate_schedule(self):
        checks=[];products=self.products[0]
        for ev in self.schedule:
            if 'requirements' in ev:
                req=np.asarray(ev['requirements']);adj={i:[] for i in range(6)}
                for i,k in enumerate(req):
                    if k>=0:adj[i]=np.flatnonzero(products==k+2).tolist()
                def depth(i,path=()):
                    if i in path:raise ValueError('dependency cycle')
                    return 0 if not adj[i] else 1+max(depth(j,path+(i,)) for j in adj[i])
                depths=[depth(i) for i in range(6)]
                # Worst constructive acquisition: at most 3 devices, each
                # needs <=5 moves, 2 known buttons and 1 take, <=24 actions.
                checks.append(dict(tick=ev['tick'],max_depth=max(depths),
                                   known_recipe_action_bound=8*(max(depths)+1),
                                   fuel_cost_bound=.004*.55*8*(max(depths)+1),
                                   body1_drift_bound=.0018*8*(max(depths)+1)))
            if 'inactive_site' in ev:
                k=products[ev['inactive_site']]
                if k>=2 or np.sum(products==k)<2:raise ValueError('no alternate body producer')
        return dict(acyclic=True,alternate_ring_route=True,alternate_body_producer=True,
                    checks=checks,note='Constructive known-recipe feasibility; not a guarantee that the agent discovers it or survives earlier losses.')

    def step(self,action):
        oldpos=int(self.pos[0]);oldbody=self.body.copy();oldalive=self.alive.copy()
        intended=(oldpos+(-1 if int(action[0])==0 else 1))%6
        denied=bool(self.blocked_edge is not None and int(action[0])<2 and
                    {oldpos,intended}==set(self.blocked_edge))
        product_site=oldpos;out=super().step(action)
        if denied and out['allowed'][0]:self.pos[0]=oldpos
        removed=np.zeros_like(self.body)
        if self.inactive_site==product_site and self.last[0]<2:
            k=int(self.last[0]);phase=self.phase()
            cost=.004*(.55 if phase>=2 else 1.) if k==0 else (.0018 if phase>=2 else .0007)
            correct=max(0.,oldbody[0,k]-cost)
            removed[0,k]=self.body[0,k]-correct;self.body[0,k]=correct
        physical_events=[];external=np.zeros_like(self.body)
        for ev in self.schedule:
            if ev['tick']!=self.t:continue
            if 'requirements' in ev:self.req_override=np.array([ev['requirements']],int)
            if 'blocked_edge' in ev:self.blocked_edge=list(ev['blocked_edge'])
            if 'recipe' in ev:self.codes[0,ev['recipe_site']]=ev['recipe']
            if 'inactive_site' in ev:self.inactive_site=int(ev['inactive_site'])
            entry=deepcopy(ev)
            if 'shock' in ev:
                axis,amount=ev['shock'];debit=min(float(amount),float(self.body[0,axis])) if self.alive[0] else 0.
                self.body[0,axis]-=debit;external[0,axis]+=debit;entry['actual_debit']=debit
            physical_events.append(entry);self.event_log.append(entry)
        self.alive &= self.body.min(1)>.001
        out.update(alive=self.alive.copy(),body=self.body.copy(),position=self.pos.copy(),
                   delta=self.body-oldbody,shock=out['shock']+external,route_denied=denied,
                   removed_yield=removed,physical_events=physical_events)
        out['reward']=np.where(oldalive,(self.body-oldbody).sum(1)+.006*self.alive-1.*(oldalive&~self.alive),0.)
        return out

    def state(self):return deepcopy(vars(self))

    @classmethod
    def from_state(cls,state):
        if state.get('whole_schema')!=WORLD_SCHEMA:raise ValueError('incompatible continuous world')
        w=cls.__new__(cls);w.__dict__.update(deepcopy(state));return w
