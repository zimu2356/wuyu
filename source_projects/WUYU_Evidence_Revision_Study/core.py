"""Transparent continuous learning. World truth is never passed to Agent."""
from collections import deque
from copy import deepcopy
import hashlib, json, math, random
import numpy as np

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()
def tuples(x):
    return tuple(tuples(v) for v in x) if isinstance(x,list) else x
def clip(x,lo,hi): return max(lo,min(hi,x))
def features(o):
    b=o['body'];f=o['fatigue'];m=o['material']
    return np.array([1.,b[0],b[1],f,m,f*m])
def physical_loss(o): return sum(x*x for x in o['body'])
def prediction_after(o,d):
    return dict(tick=o['tick']+1,body=[clip(o['body'][i]+float(d[i]),-1,1) for i in range(2)],
                fatigue=clip(o['fatigue']+float(d[2]),0,1),material=o['material'])

class World:
    def __init__(self,seed,case,c):
        self.seed=seed;self.case=case;self.config=c;self.tick=0
        rng=random.Random(seed);roles=[(0,-1),(0,1),(1,-1),(1,1)];rng.shuffle(roles)
        self.tools=[None]+[dict(axis=a,sign=s,base=rng.uniform(.14,.20),slope=rng.uniform(.06,.10)) for a,s in roles]
        boundary=random.Random(seed^0x7831AD)
        self.tau=boundary.randint(c['change_low'],c['change_high']);self.changed_axis=boundary.randrange(2)
        self.n_actions=6 if case=='opportunity' else 5
        self.body=[.55,.45];self.fatigue=.2;self.material_override=None
    def observation(self):
        c=self.config;m=self.material_override
        if m is None: m=c['training_materials'][(self.tick//c['material_period']+self.seed%2)%2]
        return dict(tick=self.tick,body=list(self.body),fatigue=self.fatigue,material=m)
    def swapped(self,t):
        return self.case in ('local_swap','recurrence') and t>=self.tau and (self.case!='recurrence' or t<self.tau+self.config['recurrence_gap'])
    def noise_scale(self,t):
        c=self.config
        return c['noisy_sd'] if self.case=='noise_pulse' and self.tau<=t<self.tau+c['noise_duration'] else c['noise_sd']
    def noise(self,t):
        raw=hashlib.blake2b(f'{self.seed}:{t}:body'.encode(),digest_size=16).digest()
        u=(int.from_bytes(raw[:8],'big')+.5)/2**64;v=(int.from_bytes(raw[8:],'big')+.5)/2**64
        radius=math.sqrt(-2*math.log(max(u,1e-18)))
        return [radius*math.cos(2*math.pi*v),radius*math.sin(2*math.pi*v)]
    def mean_delta(self,o,a):
        c=self.config;d=[c['body_drift'][i]-c['body_relaxation']*o['body'][i] for i in range(2)]
        if a==0: df=-min(c['wait_recovery'],o['fatigue'])
        else:
            df=min(c['work_fatigue'],1-o['fatigue'])
            if a==5:
                strength=.40 if o['tick']>=self.tau else .005
                for i in range(2): d[i]-=strength*o['body'][i]
            else:
                tool=self.tools[a];pol=-1 if self.swapped(o['tick']) and tool['axis']==self.changed_axis else 1
                strength=(tool['base']+tool['slope']*o['material'])*(1-.7*o['fatigue'])
                d[tool['axis']]+=pol*tool['sign']*strength
        return np.array([*d,df])
    def expected_after(self,o,a):
        # Noiseless reference, not exact E[clipped noisy outcome] near boundaries.
        result=prediction_after(o,self.mean_delta(o,a))
        if self.case=='visible_shock' and o['tick']==self.tau:
            result['body']=[clip(result['body'][0]+.45,-1,1),clip(result['body'][1]-.35,-1,1)]
            result['fatigue']=.95
        return result
    def step(self,a):
        if not 0<=a<self.n_actions: raise ValueError(a)
        o=self.observation();d=self.mean_delta(o,a);noise=self.noise(self.tick);sd=self.noise_scale(self.tick)
        self.body=[clip(self.body[i]+float(d[i])+sd*noise[i],-1,1) for i in range(2)]
        self.fatigue=clip(self.fatigue+float(d[2]),0,1);self.tick+=1
        if self.case=='visible_shock' and self.tick-1==self.tau:
            self.body=[clip(self.body[0]+.45,-1,1),clip(self.body[1]-.35,-1,1)];self.fatigue=.95
        return self.observation()
    def snapshot(self):
        return deepcopy({k:getattr(self,k) for k in ('seed','case','tick','tools','tau','changed_axis','n_actions','body','fatigue','material_override')})
    @classmethod
    def restore(cls,state,c):
        w=cls(state['seed'],state['case'],c)
        for k,v in deepcopy(state).items(): setattr(w,k,v)
        return w

class LocalModel:
    def __init__(self,c):
        self.c=c;self.samples=deque();self.gram=np.zeros((6,6));self.rhs=np.zeros((6,3))
        self.coef=np.zeros((6,3));self.mass=0.;self.mass2=0.;self.total_y=np.zeros(3)
        self.version=0;self.revisions=0;self.streak=0;self.previous_residual=np.zeros(2);self.cooldown=0
    def stats(self,s,sign):
        x=np.asarray(s['x']);y=np.asarray(s['y']);w=sign*s['w']
        self.gram+=w*np.outer(x,x);self.rhs+=w*np.outer(x,y);self.total_y+=w*y
        self.mass+=w;self.mass2+=sign*s['w']**2
    def rebuild(self):
        self.gram.fill(0);self.rhs.fill(0);self.total_y.fill(0);self.mass=0.;self.mass2=0.
        for s in self.samples: self.stats(s,1)
    def fit(self): self.coef=np.linalg.solve(self.gram+self.c['ridge']*np.eye(6),self.rhs)
    def predict_x(self,x,kind='conditional'):
        if not self.samples: return np.zeros(3)
        return self.total_y/self.mass if kind=='mean' else x@self.coef
    def evidence(self,o):
        if not self.samples: return dict(age=o['tick']+1,mass=0.,ess=0.,coverage=-1.)
        # Distance heuristic in raw (body0,body1,fatigue,material) coordinates.
        x=features(o);d=min(sum((x[j]-s['x'][j])**2 for j in (1,2,3,4)) for s in self.samples)
        return dict(age=o['tick']-self.samples[-1]['tick'],mass=self.mass,
                    ess=self.mass**2/max(self.mass2,1e-30),coverage=math.sqrt(d))
    def observe(self,o,after,event_id,pred,enabled):
        c=self.c;x=features(o);y=np.array([after['body'][i]-o['body'][i] for i in range(2)]+[after['fatigue']-o['fatigue']])
        residual=y[:2]-pred[:2];norm=float(np.linalg.norm(residual));prev=float(np.linalg.norm(self.previous_residual))
        aligned=prev>0 and float(residual@self.previous_residual)>=c['residual_cosine']*norm*prev
        mature=len(self.samples)>=c['detector_min_samples'];large=norm>c['residual_threshold']
        if self.cooldown: self.cooldown-=1;self.streak=0
        elif mature and large: self.streak=self.streak+1 if aligned and self.streak else 1
        else: self.streak=0
        self.previous_residual=residual.copy()
        trigger=bool(enabled and self.streak>=c['residual_streak'])
        if trigger:
            # Preserve the immediately preceding contradictory observations; discount older ones.
            retain=c['residual_streak']-1
            for s in list(self.samples)[:-retain]: s['w']*=c['revision_old_weight']
            self.rebuild();self.revisions+=1;self.streak=0;self.cooldown=c['revision_cooldown']
        if len(self.samples)==c['model_window']: self.stats(self.samples.popleft(),-1)
        s=dict(event_id=event_id,tick=o['tick'],x=x.tolist(),y=y.tolist(),w=1.)
        self.samples.append(s);self.stats(s,1);self.fit();self.version+=1
        return norm,trigger
    def snapshot(self):
        return dict(samples=deepcopy(list(self.samples)),gram=self.gram.tolist(),rhs=self.rhs.tolist(),coef=self.coef.tolist(),
                    total_y=self.total_y.tolist(),mass=self.mass,mass2=self.mass2,version=self.version,revisions=self.revisions,
                    streak=self.streak,previous_residual=self.previous_residual.tolist(),cooldown=self.cooldown)
    @classmethod
    def restore(cls,s,c):
        m=cls(c)
        for k,v in deepcopy(s).items():
            setattr(m,k,deque(v) if k=='samples' else np.array(v) if k in ('gram','rhs','coef','total_y','previous_residual') else v)
        return m

AGENT_KEYS=('fatigue_penalty','action_cost','exploration_weight','model_window','ridge','recency_cap','recency_horizon',
            'residual_threshold','residual_streak','residual_cosine','detector_min_samples','revision_old_weight',
            'revision_cooldown','random_probe_probability')
class Agent:
    def __init__(self,seed,c,method,n_actions):
        self.c={k:c[k] for k in AGENT_KEYS};self.method=method;self.n_actions=n_actions
        self.models=[LocalModel(self.c) for _ in range(n_actions)];self.visits=[0]*n_actions;self.last_use=[-1]*n_actions
        self.rng=random.Random(seed+710003);self.probe_rng=random.Random(seed+0x91ACB);self.learn=True
    def candidates(self,o,override=None,kind='conditional'):
        c=self.c;x=features(o);pred=np.array([m.predict_x(x,kind) for m in self.models]) if override is None else np.asarray(override)
        before=physical_loss(o)+c['fatigue_penalty']*o['fatigue']**2
        b=np.clip(np.asarray(o['body'])[None,:]+pred[:,:2],-1,1);f=np.clip(o['fatigue']+pred[:,2],0,1)
        benefit=before-np.sum(b*b,axis=1)-c['fatigue_penalty']*f*f
        scores=benefit+c['exploration_weight']/np.sqrt(1+np.asarray(self.visits))-np.array([0]+[c['action_cost']]*(self.n_actions-1))
        ages=o['tick']-np.asarray(self.last_use);bonus=c['recency_cap']*np.minimum(ages/c['recency_horizon'],1.)
        total=scores+bonus if self.method in ('BE','BEU') else scores
        return pred,scores,total,bonus
    def decide(self,o,override=None):
        pred,base,total,bonus=self.candidates(o,override)
        tied=np.flatnonzero(np.abs(total-total.max())<=1e-12).tolist();a=self.rng.choice(tied)
        base_choice=int(np.argmax(base));purpose=1 if self.method in ('BE','BEU') and base[a]<base.max()-1e-12 else 0
        if self.method=='BR' and self.probe_rng.random()<self.c['random_probe_probability']:
            a=self.probe_rng.randrange(self.n_actions);purpose=2
        return a,dict(pred=pred,score=total,base_choice=base_choice,purpose=purpose,bonus=bonus)
    def observe(self,o,a,after,event_id,pred=None,block=False):
        self.visits[a]+=1;self.last_use[a]=o['tick']
        if pred is None: pred=self.models[a].predict_x(features(o))
        if not self.learn or block: return float(np.linalg.norm(np.asarray(after['body'])-o['body']-pred[:2])),False
        return self.models[a].observe(o,after,event_id,pred,self.method in ('BU','BEU'))
    def snapshot(self):
        return dict(method=self.method,n_actions=self.n_actions,models=[m.snapshot() for m in self.models],visits=list(self.visits),
                    last_use=list(self.last_use),rng_state=self.rng.getstate(),probe_rng_state=self.probe_rng.getstate(),learn=self.learn)
    @classmethod
    def restore(cls,s,c):
        a=cls(0,c,s['method'],s['n_actions']);a.models=[LocalModel.restore(m,a.c) for m in s['models']]
        a.visits=list(s['visits']);a.last_use=list(s['last_use']);a.rng.setstate(tuples(s['rng_state']))
        a.probe_rng.setstate(tuples(s['probe_rng_state']));a.learn=s['learn'];return a
