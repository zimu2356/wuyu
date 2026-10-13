"""Lifecycle adapters; frozen learners and original task semantics stay intact."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib
import random
import numpy as np
from .paths import config, source_module

class LegacyBackend:
    def __init__(self, native): self.native = native

    @classmethod
    def create(cls, profile, seed, horizon=None, **_):
        source_module("integrated")
        legacy = importlib.import_module("wuyu_core.legacy")
        return cls(legacy.LegacySession.create(int(profile), seed, horizon or 2048))

    def step(self):
        n = self.native
        before = getattr(n.world, "t", n.cursor)
        record = n.step()
        return dict(clock_before=before, clock_after=getattr(n.world, "t", n.cursor),
                    action=record.get("action", record.get("answer")), payload=record,
                    accounting=dict(unit="legacy_native_event", primitive_actions=None,
                                    note="Original batch and macro/micro-action semantics retained"))

    def state(self): return self.native.state()

    @classmethod
    def restore(cls, state):
        source_module("integrated")
        legacy = importlib.import_module("wuyu_core.legacy")
        return cls(legacy.LegacySession.from_state(state))

class IntegratedBackend:
    def __init__(self, native): self.native = native

    @classmethod
    def create(cls, seed, mode="prior", horizon=None, initialization=0, **_):
        m = source_module("integrated")
        return cls(m.Session.create(seed, initialization, mode, horizon or 1536))

    def step(self):
        before = self.native.world.t
        record = self.native.step()
        return dict(clock_before=before, clock_after=self.native.world.t, action=record["action"],
                    before=record["obs"][0], after=self.native.world.observation()[0], payload=record,
                    accounting=dict(unit="primitive_action", primitive_actions=1, batch_size=1))

    def state(self): return self.native.state()

    @classmethod
    def restore(cls, state): return cls(source_module("integrated").Session.from_state(state))

class ContinuousBackend:
    def __init__(self, profile, world, agent, configuration, limit=None, event_namespace="main"):
        self.profile, self.world, self.agent = profile, world, agent
        self.configuration, self.limit = deepcopy(configuration), limit
        self.event_namespace = event_namespace

    @classmethod
    def create(cls, profile, seed, mode, case="stable", horizon=None, **_):
        c = config(profile); m = source_module(profile)
        if profile == "revision":
            w = m.World(seed, case, c); a = m.Agent(seed, c, mode, w.n_actions)
        else:
            w = m.World(seed, c); a = m.Agent(seed, c); a.learn = mode == "online"
        return cls(profile, w, a, c, c["life_ticks"] if horizon is None else horizon)

    def step(self):
        w, a = self.world, self.agent
        if self.limit is not None and w.tick >= self.limit: raise StopIteration("declared horizon reached")
        before = w.observation()
        action, decision = a.decide(before)
        if self.profile == "revision":
            event_id = f"{w.seed}:{w.case}:{a.method}:{w.tick}" if self.event_namespace=="main" else f"{self.event_namespace}:{w.tick}"
            prediction = decision["pred"][action].copy()
            evidence = a.models[action].evidence(before)
        else:
            event_id = f"{w.seed}:{self.event_namespace}:{w.tick}"
            prediction = decision["candidates"][action]["predicted_delta"]
            evidence = None
        after = w.step(action)
        if self.profile == "revision":
            residual, revised = a.observe(before, action, after, event_id, prediction)
        else:
            a.observe(before, action, after, event_id)
            actual = [after["body"][i]-before["body"][i] for i in range(2)]
            residual = float(np.linalg.norm(np.asarray(actual)-np.asarray(prediction)[:2]))
            revised = False
        return dict(clock_before=before["tick"], clock_after=after["tick"], event_id=event_id,
                    before=before, action=action, decision=decision, prediction=prediction,
                    evidence_before=evidence, after=after, learning=dict(residual=residual, revision=revised),
                    accounting=dict(unit="primitive_action", primitive_actions=1, active_actions=int(action != 0)),
                    metrics=dict(body_squared_loss=sum(x*x for x in after["body"]), fatigue=after["fatigue"]))

    def state(self):
        return dict(profile=self.profile, configuration=deepcopy(self.configuration), limit=self.limit,event_namespace=self.event_namespace,
                    world=self.world.snapshot(), agent=self.agent.snapshot())

    @classmethod
    def restore(cls, s):
        m = source_module(s["profile"]); c = s["configuration"]
        return cls(s["profile"], m.World.restore(s["world"], c), m.Agent.restore(s["agent"], c), c, s["limit"],s.get("event_namespace","main"))

def world_state(world):
    s = {k: deepcopy(v) for k,v in vars(world).items() if k != "effect_rng"}
    s["effect_rng_state"] = world.effect_rng.getstate()
    return s

def restore_world(module, state):
    world = module.World.__new__(module.World)
    for k,v in state.items():
        if k != "effect_rng_state": setattr(world,k,deepcopy(v))
    world.effect_rng = random.Random(); world.effect_rng.setstate(state["effect_rng_state"])
    return world

def learner_state(agent):
    s = {k: deepcopy(v) for k,v in vars(agent).items() if k != "rng"}
    s["rng_state"] = agent.rng.getstate()
    return s

def restore_learner(module, state):
    agent = module.Learner.__new__(module.Learner)
    for k,v in state.items():
        if k != "rng_state": setattr(agent,k,deepcopy(v))
    agent.rng = random.Random(); agent.rng.setstate(state["rng_state"])
    return agent

class GoalBackend:
    """Original episodic study, including actual post-outcome stop decision."""
    @classmethod
    def create(cls, seed, mode="learned", case="quiet", **_):
        m = source_module("goals"); c = config("goals")
        events = m.training_experiences(seed, c["training_repeats_per_state_action"],
                                       c["effect_noise_sd"], tuple(c["train_body"]))
        return cls.from_experiences(seed, mode, case, events, c, "balanced_teacher")

    @classmethod
    def from_experiences(cls, seed, mode, case, events, c, acquisition):
        obj = cls(); obj.seed=seed; obj.configuration=deepcopy(c)
        obj.case=case; obj.mode=mode; obj.acquisition=acquisition
        m=source_module("goals"); obj.world=m.World(seed,c["effect_noise_sd"])
        if acquisition != "balanced_teacher":
            obj.world.effect_rng=random.Random(seed+c["test_effect_seed_offset"]+c["scenarios"].index(case))
        obj.agent=m.Learner(mode,seed,c["action_cost"],obj.world.outcomes[0] if mode=="fixed_target" else None)
        for e in events: obj.agent.observe(e)
        if case=="blocked": obj.world.block_short_a()
        key="shift_body" if case=="need_shift" else "satiated_body" if case=="satiated" else "train_body"
        obj.body=tuple(c[key]); obj.position=obj.world.start
        obj.cursor=0; obj.actions=0; obj.done=False
        obj.max_actions=c.get("max_test_actions",c.get("max_steps",12))
        obj.training_count=len(events)
        obj.training_events=[asdict(e) for e in events]
        return obj

    def step(self):
        if self.done or self.actions>=self.max_actions: raise StopIteration("original episode finished")
        before=dict(state=self.position,body=self.body)
        action,goal,decision=self.agent.decide(self.position,self.body)
        record=dict(clock_before=self.cursor,before=before,action=action,goal=goal,decision=decision,
                    accounting=dict(unit="primitive_action",primitive_actions=int(action is not None)))
        if action is None:
            self.done=True; record["after"]=deepcopy(before)
        else:
            e=self.world.step(self.position,action,self.body)
            self.agent.observe(e); self.actions+=1
            self.position,self.body=e.next_state,e.after
            record["experience"]=asdict(e); record["after"]=dict(state=self.position,body=self.body)
            if e.outcome is not None:
                a,g,d=self.agent.decide(self.position,self.body)
                record["post_completion_decision"]=dict(action=a,goal=g,decision=d)
                self.done=True
        self.cursor+=1; record["clock_after"]=self.cursor
        record["episode_finished"]=self.done or self.actions>=self.max_actions
        return record

    def state(self):
        s={k:deepcopy(v) for k,v in vars(self).items() if k not in ("world","agent")}
        s.update(world=world_state(self.world),agent=learner_state(self.agent))
        return s

    @classmethod
    def restore(cls,s):
        obj=cls(); m=source_module("goals")
        for k,v in s.items():
            if k not in ("world","agent"):setattr(obj,k,deepcopy(v))
        obj.world=restore_world(m,s["world"]); obj.agent=restore_learner(m,s["agent"])
        return obj

class ExplorationBackend:
    """Body reset belongs to this acquisition protocol and is explicitly logged."""
    @classmethod
    def create(cls,seed,mode="least_visited",horizon=None,**_):
        obj=cls(); obj.seed=seed; obj.configuration=config("exploration")
        m=source_module("exploration")
        obj.world=m.World(seed,obj.configuration["effect_noise_sd"])
        obj.collector=m.Collector(mode,seed,obj.configuration)
        obj.position=obj.world.start; obj.cursor=0; obj.events=[]
        obj.limit=max(obj.configuration["budgets"]) if horizon is None else horizon
        return obj

    def step(self):
        if self.cursor>=self.limit:raise StopIteration("acquisition budget reached")
        action=self.collector.choose(self.position)
        e=self.world.step(self.position,action,tuple(self.configuration["train_body"]))
        self.collector.observe(e.state,e.action,e.next_state)
        self.events.append(e); self.position=e.next_state
        before=self.cursor;self.cursor+=1
        return dict(clock_before=before,clock_after=self.cursor,action=action,
                    before=dict(state=e.state,body=e.before),after=dict(state=e.next_state,body=e.after),
                    experience=asdict(e),body_reset_before_action=True,
                    accounting=dict(unit="primitive_action",primitive_actions=1))

    def evaluation(self,case="quiet",variant="learned"):
        return GoalBackend.from_experiences(self.seed,variant,case,self.events,self.configuration,self.collector.kind)

    def state(self):
        return dict(seed=self.seed,configuration=deepcopy(self.configuration),position=self.position,
                    cursor=self.cursor,limit=self.limit,world=world_state(self.world),
                    collector=dict(kind=self.collector.kind,rng_state=self.collector.rng.getstate(),
                                   transitions=deepcopy(self.collector.transitions),counts=dict(self.collector.counts)),
                    events=[asdict(e) for e in self.events])

    @classmethod
    def restore(cls,s):
        obj=cls();m=source_module("exploration")
        for k in ("seed","configuration","position","cursor","limit"):setattr(obj,k,deepcopy(s[k]))
        obj.world=restore_world(m,s["world"])
        c=s["collector"];obj.collector=m.Collector(c["kind"],s["seed"],s["configuration"])
        obj.collector.rng.setstate(c["rng_state"])
        obj.collector.transitions=deepcopy(c["transitions"]);obj.collector.counts=Counter(c["counts"])
        obj.events=[m.Experience(**deepcopy(e)) for e in s["events"]]
        return obj

BACKENDS={**{f"{i:02}":LegacyBackend for i in range(1,9)},"integrated":IntegratedBackend,
          "goals":GoalBackend,"exploration":ExplorationBackend,
          "history":ContinuousBackend,"revision":ContinuousBackend}
