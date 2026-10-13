"""Engineering bridge: old bodily policy/memory + new evidence-update component.

The original graph-policy weights remain fixed. Only observable transitions train
the new head. Shadow preserves the old prior's complete physical trajectory.
No teacher, hidden recipe, world phase, or counterfactual label enters this head.
"""
from copy import deepcopy
import importlib
import numpy as np
from .paths import config,source_module
from .evidence_model import WindowedEvidenceModel,MODEL_KEYS

FEATURE_SCHEMA="workshop-visible-observation21-plus-intercept/1"
TARGET_SCHEMA="body-delta2-inventory-delta2-local-stage-change-product-present/1"

def features(observation):
    observation=np.asarray(observation,dtype=float)
    if observation.shape!=(21,):raise ValueError("expected 21 visible channels")
    return np.concatenate([[1.],observation])

def target(before,after):
    return np.concatenate([after[14:16]-before[14:16],after[12:14]-before[12:14],
                           [(after[6:9].argmax()-before[6:9].argmax())/2.,float(after[16:20].sum()>.5)]])

class WorkshopBridge:
    @classmethod
    def create(cls,seed,mode="shadow",horizon=None,initialization=0,**_):
        obj=cls();m=source_module("integrated")
        obj.native=m.Session.create(seed,initialization,"shadow" if mode=="combined" else "prior",horizon or 1536)
        obj.mode=mode; raw=config("revision")
        obj.configuration={k:raw[k] for k in MODEL_KEYS}
        obj.configuration.update(recency_cap=raw["recency_cap"],recency_horizon=raw["recency_horizon"],
                                 warmup=32,bias_strength=1.,body_scale=.28,neural_blend=.5)
        obj.models=[WindowedEvidenceModel(22,6,obj.configuration) for _ in range(7)]
        obj.visits=[0]*7;obj.last_use=[-1]*7
        obj.feature_schema=FEATURE_SCHEMA;obj.target_schema=TARGET_SCHEMA
        return obj

    def candidate_decision(self,obs):
        """Pure prediction and scoring. It never sees a World object."""
        ind=self.native.individual
        encode=importlib.import_module("wuyu_core.features").encode
        softmax=importlib.import_module("wuyu_core.memory").softmax
        encoded=encode(obs[None],ind.prev,[ind.memory])
        prior=ind.net.forward(encoded,ind.h)["ap"][0]
        native_probability=softmax(np.log(np.maximum(prior,1e-12)))
        x=features(obs);pred=np.array([m.predict(x) for m in self.models])
        need=(1-obs[14:16])**2+.02
        utility=(pred[:,:2]/self.configuration["body_scale"])@need+.05*pred[:,2:4].sum(1)+.05*pred[:,4]
        neural_predictions=None
        if self.mode=="combined":
            context=np.concatenate([obs,ind.prev[0],encoded["node"][0,obs[:6].argmax()]])
            neural_predictions,_=ind.predictor.predict(context)
            neural_utility=neural_predictions[:,:2]@need+.05*neural_predictions[:,2:4].sum(1)+.05*neural_predictions[:,4]
            blend=self.configuration["neural_blend"]
            utility=(1-blend)*utility+blend*neural_utility
        bias=np.clip(utility*self.configuration["bias_strength"],-1.,1.)
        age=ind.tick-np.asarray(self.last_use)
        age_bonus=self.configuration["recency_cap"]*np.minimum(age/self.configuration["recency_horizon"],1.)
        use=self.mode!="shadow" and sum(self.visits)>=self.configuration["warmup"]
        applied=bias+(age_bonus if self.mode in ("evidence","revision","combined") else 0.) if use else np.zeros(7)
        probability=softmax(np.log(np.maximum(prior,1e-12))+applied)
        # Preview exactly the private draw which native.step will consume once.
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(ind.rng.bit_generator.state)
        draw=rng.random()
        chosen=int(min(np.searchsorted(probability.cumsum(),draw),6))
        native_chosen=int(min(np.searchsorted(native_probability.cumsum(),draw),6))
        return dict(features=x,predictions=pred,neural_predictions=neural_predictions,probability=probability,prior_probability=native_probability,
                    bias=bias,age_bonus=age_bonus,use=use,chosen=chosen,native_choice=native_chosen)

    def step(self):
        if self.native.world.t>=self.native.world.total:raise StopIteration("declared horizon reached")
        before=self.native.world.observation()[0].copy();tick=self.native.world.t
        # Native ingest is idempotent for this event. Feature encoding stays read-only.
        self.native.individual.ingest(before[None])
        d=self.candidate_decision(before)
        evidence=[m.evidence(d["features"],tick) for m in self.models]
        record=self.native.step(forced=d["chosen"] if d["use"] else None)
        action=record["action"];after=self.native.world.observation()[0].copy()
        y=target(before,after);event_id=f"bridge:{self.native.individual.seed}:{tick}"
        updated=bool(before[14:16].min()>.001)
        residual=None;revision=False
        if updated:
            residual,revision=self.models[action].update(d["features"],y,tick,event_id,
                                                        d["predictions"][action],self.mode in ("update","revision","combined"))
            self.visits[action]+=1;self.last_use[action]=tick
        return dict(clock_before=tick,clock_after=self.native.world.t,event_id=event_id,
                    before=before,action=action,after=after,prediction=d["predictions"][action],
                    decision={k:v for k,v in d.items() if k!="features"},evidence_before=evidence,
                    learning=dict(updated=updated,residual=residual,revision=revision,target=y),
                    native_payload=record,feature_schema=self.feature_schema,target_schema=self.target_schema,
                    accounting=dict(unit="primitive_action",primitive_actions=1,
                                    model_updates=int(updated),extra_model_fits=int(updated),
                                    graph_forward_passes=2,neural_updates=record["alive"]*int(self.mode=="combined"),
                                    replay_updates=int(record["alive"] and self.mode=="combined" and sum(self.visits)>1),
                                    comparison_note="extra computation is reported, not budget-matched"),
                    evidence_role="engineering verification only; no performance superiority claim")

    def state(self):
        return dict(native=self.native.state(),mode=self.mode,configuration=deepcopy(self.configuration),
                    models=[m.state() for m in self.models],visits=list(self.visits),last_use=list(self.last_use),
                    feature_schema=self.feature_schema,target_schema=self.target_schema)

    @classmethod
    def restore(cls,s):
        if s["feature_schema"]!=FEATURE_SCHEMA or s["target_schema"]!=TARGET_SCHEMA:
            raise ValueError("incompatible world features/targets; weights cannot be merged")
        obj=cls();obj.native=source_module("integrated").Session.from_state(s["native"])
        for k in ("mode","configuration","visits","last_use","feature_schema","target_schema"):
            setattr(obj,k,deepcopy(s[k]))
        obj.models=[WindowedEvidenceModel.restore(m) for m in s["models"]]
        return obj
