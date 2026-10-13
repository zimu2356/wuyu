"""Dimension-general extraction of the frozen evidence study's local update.

With six inputs and three outputs this is checked against the original code.
Other feature/target definitions are new engineering mappings, not validated
scientific transfer claims. Source: WUYU_Evidence_Revision_Study/core.py.
"""
from collections import deque
from copy import deepcopy
import numpy as np

MODEL_KEYS=("model_window","ridge","residual_threshold","residual_cosine",
            "detector_min_samples","residual_streak","revision_old_weight","revision_cooldown")

class WindowedEvidenceModel:
    def __init__(self, inputs, outputs, configuration, residual_dimensions=2):
        self.inputs=int(inputs);self.outputs=int(outputs);self.residual_dimensions=int(residual_dimensions)
        self.c={k:configuration[k] for k in MODEL_KEYS}
        if self.inputs<1 or self.outputs<1 or not 1<=self.residual_dimensions<=self.outputs:
            raise ValueError("invalid model dimensions")
        if self.c["model_window"]<1 or self.c["ridge"]<=0 or self.c["residual_streak"]<2:
            raise ValueError("invalid model configuration")
        self.samples=deque();self.gram=np.zeros((inputs,inputs));self.rhs=np.zeros((inputs,outputs))
        self.coef=np.zeros((inputs,outputs));self.total_y=np.zeros(outputs)
        self.mass=0.;self.mass2=0.;self.version=0;self.revisions=0;self.streak=0;self.cooldown=0
        self.previous_residual=np.zeros(residual_dimensions)

    def stats(self,s,sign):
        x=np.asarray(s["x"]);y=np.asarray(s["y"]);w=sign*s["w"]
        self.gram+=w*np.outer(x,x);self.rhs+=w*np.outer(x,y);self.total_y+=w*y
        self.mass+=w;self.mass2+=sign*s["w"]**2

    def rebuild(self):
        self.gram.fill(0);self.rhs.fill(0);self.total_y.fill(0);self.mass=0.;self.mass2=0.
        for s in self.samples:self.stats(s,1)

    def predict(self,x,kind="conditional"):
        x=np.asarray(x,dtype=float)
        if x.shape!=(self.inputs,):raise ValueError("feature shape mismatch")
        if not self.samples:return np.zeros(self.outputs)
        if kind=="mean":return self.total_y/self.mass
        if kind!="conditional":raise ValueError(kind)
        return x@self.coef

    def update(self,x,y,tick,event_id,prediction,revise=False):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);prediction=np.asarray(prediction,dtype=float)
        if x.shape!=(self.inputs,) or y.shape!=(self.outputs,) or prediction.shape!=(self.outputs,):
            raise ValueError("feature/target/prediction shape mismatch")
        c=self.c;n=self.residual_dimensions
        residual=y[:n]-prediction[:n];norm=float(np.linalg.norm(residual));prev=float(np.linalg.norm(self.previous_residual))
        aligned=prev>0 and float(residual@self.previous_residual)>=c["residual_cosine"]*norm*prev
        mature=len(self.samples)>=c["detector_min_samples"];large=norm>c["residual_threshold"]
        if self.cooldown:self.cooldown-=1;self.streak=0
        elif mature and large:self.streak=self.streak+1 if aligned and self.streak else 1
        else:self.streak=0
        self.previous_residual=residual.copy()
        trigger=bool(revise and self.streak>=c["residual_streak"])
        if trigger:
            retain=c["residual_streak"]-1
            for s in list(self.samples)[:-retain]:s["w"]*=c["revision_old_weight"]
            self.rebuild();self.revisions+=1;self.streak=0;self.cooldown=c["revision_cooldown"]
        if len(self.samples)==c["model_window"]:self.stats(self.samples.popleft(),-1)
        s=dict(event_id=event_id,tick=int(tick),x=x.tolist(),y=y.tolist(),w=1.)
        self.samples.append(s);self.stats(s,1)
        self.coef=np.linalg.solve(self.gram+c["ridge"]*np.eye(self.inputs),self.rhs)
        self.version+=1
        return norm,trigger

    def evidence(self,x,tick):
        if not self.samples:return dict(age=tick+1,n=0,mass=0.,ess=0.,coverage=None)
        distance=float(np.linalg.norm(np.asarray([s["x"] for s in self.samples])[:,1:]-np.asarray(x)[None,1:],axis=1).min())
        return dict(age=tick-self.samples[-1]["tick"],n=len(self.samples),mass=self.mass,
                    ess=self.mass**2/max(self.mass2,1e-30),coverage=distance,
                    note="raw feature distance; not a calibrated probability")

    def state(self):return deepcopy(vars(self))

    @classmethod
    def restore(cls,s):
        obj=cls.__new__(cls);obj.__dict__.update(deepcopy(s));return obj

