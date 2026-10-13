"""A single versioned event/checkpoint/fork contract over all research profiles."""
from copy import deepcopy
import hashlib
from pathlib import Path
from . import codec
from .adapters import BACKENDS,ExplorationBackend
from .paths import ROOT,SOURCES,project,source_module
from .registry import DESCRIPTIONS,MODES,DEFAULT_MODES,CASES,describe
from .workshop_bridge import WorkshopBridge
from .whole import WholeAgent

SCHEMA="wuyu-unified-session/1"
BACKENDS={**BACKENDS,"workshop_bridge":WorkshopBridge,"whole":WholeAgent}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def source_locks(profile):
    files=set((ROOT/"wuyu_unified").glob("*.py"))
    names=["integrated"] if profile.isdigit() or profile in ("integrated","workshop_bridge","whole") else [profile]
    if profile in ("workshop_bridge","whole"):names.append("revision")
    if profile=="exploration":names.append("goals")
    for name in names:
        directory=project(name)
        files.update(directory.rglob("*.py"))
        for f in directory.glob("*protocol.json"):files.add(f)
        if name=="integrated":files.add(directory/"legacy/WUYU_08/CONFIG.json")
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}

class Session:
    def __init__(self,profile,backend,event_index=0,lineage=None):
        self.profile=profile;self.backend=backend;self.event_index=event_index
        self.lineage=deepcopy(lineage)

    @classmethod
    def create(cls,profile="whole",seed=990001,mode=None,case=None,horizon=None,initialization=0):
        if profile not in BACKENDS:raise ValueError(f"unknown profile: {profile}")
        if seed<0:raise ValueError("seed must be nonnegative")
        if horizon is not None and horizon<1:raise ValueError("horizon must be positive")
        mode=mode or DEFAULT_MODES.get(profile,"native_checkpoint")
        if profile in MODES and mode not in MODES[profile]:raise ValueError(f"invalid mode for {profile}: {mode}")
        if profile.isdigit() and mode!="native_checkpoint":raise ValueError("legacy mode is defined by its native checkpoint")
        if profile.isdigit() and int(profile)>=4 and horizon is not None:
            raise ValueError("04–08 retain their checkpoint's original world horizon")
        if profile=="goals" and horizon is not None:
            raise ValueError("goals retains its original episodic action budget; use --steps to stop early")
        if profile in CASES:
            case=case or CASES[profile][0]
            if case not in CASES[profile]:raise ValueError(f"invalid case for {profile}: {case}")
        elif case is not None:raise ValueError(f"case is not applicable to {profile}")
        backend=BACKENDS[profile].create(profile=profile,seed=seed,mode=mode,case=case,
                                         horizon=horizon,initialization=initialization)
        return cls(profile,backend,lineage=dict(origin="new adapter session",scope="engineering run; original scientific results remain separate"))

    def step(self):
        payload=self.backend.step()
        event=dict(schema="wuyu-event/1",profile=self.profile,event_index=self.event_index,
                   semantics=describe(self.profile),**payload)
        self.event_index+=1
        return event

    def run(self,steps):
        if steps<0:raise ValueError("steps must be nonnegative")
        result=[]
        for _ in range(steps):
            try:result.append(self.step())
            except StopIteration:break
        return result

    def state(self,with_locks=True):
        result=dict(schema=SCHEMA,profile=self.profile,event_index=self.event_index,
                    lineage=deepcopy(self.lineage),backend=self.backend.state())
        if with_locks:result["source_locks"]=source_locks(self.profile)
        return result

    def save(self,path):codec.save(path,self.state())

    @classmethod
    def from_state(cls,state,verify_sources=True):
        if state.get("schema")!=SCHEMA:raise ValueError("unsupported unified schema")
        profile=state["profile"]
        if profile not in BACKENDS:raise ValueError("unsupported profile")
        if verify_sources:
            expected=state.get("source_locks")
            if expected is None or source_locks(profile)!=expected:
                raise ValueError("source/configuration changed since this checkpoint; refuse silent migration")
        return cls(profile,BACKENDS[profile].restore(state["backend"]),state["event_index"],state.get("lineage"))

    @classmethod
    def load(cls,path):return cls.from_state(codec.load(path))

    def fork(self,intervention=None,action=None):
        original=self.state(with_locks=False)
        child=self.from_state(deepcopy(original),verify_sources=False)
        if intervention=="erase_model":
            if self.profile not in ("history","revision","workshop_bridge"):
                raise ValueError("erase_model is defined only for explicit per-action evidence models")
            models=child.backend.models if self.profile=="workshop_bridge" else child.backend.agent.models
            if action is None or action<0 or action>=len(models):raise ValueError("invalid model action")
            if self.profile=="workshop_bridge":
                from .evidence_model import WindowedEvidenceModel
                models[action]=WindowedEvidenceModel(22,6,child.backend.configuration)
            else:
                models[action]=source_module(self.profile).LocalModel(child.backend.configuration)
        elif intervention=="freeze_learning":
            if self.profile not in ("history","revision"):
                raise ValueError("freeze_learning here refers only to the continuous study's model updates")
            child.backend.agent.learn=False
        elif intervention=="suppress_work_state":
            if self.profile!='whole' or child.backend.actor is None:raise ValueError('requires whole B/C/D')
            child.backend.suppress_until=child.backend.native.world.t+32
        elif intervention is not None:raise ValueError("unsupported intervention")
        child.lineage=dict(origin="diagnostic fork",parent_profile=self.profile,parent_event=self.event_index,
                           parent_state_sha256=codec.digest(original),intervention=intervention,action=action,
                           note="Researcher intervention; not autonomous self-change")
        if codec.digest(self.state(with_locks=False))!=codec.digest(original):
            raise AssertionError("fork mutated its parent")
        return child

    def evaluation(self,case="quiet",variant="learned"):
        if not isinstance(self.backend,ExplorationBackend):raise ValueError("evaluation requires exploration history")
        if case not in CASES["goals"] or variant not in MODES["goals"]:raise ValueError("invalid evaluation condition")
        return Session("goals",self.backend.evaluation(case,variant),lineage=dict(
            origin="isolated evaluation from actual acquisition history",parent_profile=self.profile,
            parent_event=self.event_index,body_reset=True,position_reset=True,
            acquisition_events=len(self.backend.events),note="Not a continuation of continuous bodily life"))

