"""A continuing low-dimensional world and an agent that sees observations only."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import random


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def clip(value,low,high):
    return max(low,min(high,value))


def features(obs):
    b0,b1=obs['body'];f=obs['fatigue'];m=obs['material']
    return [1.0,b0,b1,f,m,f*m]


def physical_loss(obs):
    return sum(b*b for b in obs['body'])


def evaluation_loss(obs,config):
    return physical_loss(obs)+config['fatigue_penalty']*obs['fatigue']**2


def predicted_observation(obs,delta):
    return dict(tick=obs['tick']+1,body=[clip(obs['body'][i]+delta[i],-1,1) for i in range(2)],
                fatigue=clip(obs['fatigue']+delta[2],0,1),material=obs['material'])


class World:
    def __init__(self,seed,config):
        self.seed=seed;self.config=config;self.tick=0
        rng=random.Random(seed)
        roles=[(0,-1),(0,1),(1,-1),(1,1)];rng.shuffle(roles)
        self.tools=[None]+[dict(axis=axis,sign=sign,base=rng.uniform(.14,.20),slope=rng.uniform(.06,.10)) for axis,sign in roles]
        self.body=[.55,.45];self.fatigue=.2;self.material_override=None

    def material(self):
        if self.material_override is not None: return self.material_override
        index=(self.tick//self.config['material_period']+self.seed%2)%2
        return self.config['training_materials'][index]

    def observation(self):
        return dict(tick=self.tick,body=list(self.body),fatigue=self.fatigue,material=self.material())

    def mean_delta(self,obs,action):
        """Evaluator/world only; never supplied to Agent."""
        c=self.config
        delta=[c['body_drift'][i]-c['body_relaxation']*obs['body'][i] for i in range(2)]
        if action==0:
            df=-min(c['wait_recovery'],obs['fatigue'])
        else:
            tool=self.tools[action]
            polarity=-1 if obs['tick']>=c['change_tick'] else 1
            strength=(tool['base']+tool['slope']*obs['material'])*(1-.7*obs['fatigue'])
            delta[tool['axis']]+=polarity*tool['sign']*strength
            df=min(c['work_fatigue'],1-obs['fatigue'])
        return [*delta,df]

    def noise(self,tick):
        raw=hashlib.blake2b(f'{self.seed}:{tick}:body'.encode(),digest_size=16).digest()
        u=(int.from_bytes(raw[:8],'big')+.5)/2**64
        v=(int.from_bytes(raw[8:],'big')+.5)/2**64
        radius=math.sqrt(-2*math.log(max(u,1e-18)))
        return [radius*math.cos(2*math.pi*v),radius*math.sin(2*math.pi*v)]

    def step(self,action):
        if action not in range(5): raise ValueError(action)
        before=self.observation();delta=self.mean_delta(before,action);noise=self.noise(self.tick)
        self.body=[clip(self.body[i]+delta[i]+self.config['noise_sd']*noise[i],-1,1) for i in range(2)]
        self.fatigue=clip(self.fatigue+delta[2],0,1);self.tick+=1
        return self.observation()

    def snapshot(self):
        return deepcopy(dict(seed=self.seed,tick=self.tick,tools=self.tools,body=self.body,
                             fatigue=self.fatigue,material_override=self.material_override))

    @classmethod
    def restore(cls,state,config):
        obj=cls(state['seed'],config)
        for key,value in deepcopy(state).items(): setattr(obj,key,value)
        return obj


class LocalModel:
    """Sliding-window sufficient statistics and state-conditioned ridge regression."""
    def __init__(self,config):
        self.window=config['model_window'];self.ridge=config['ridge'];self.samples=deque()
        self.gram=[[0.0]*6 for _ in range(6)]
        self.rhs=[[0.0]*3 for _ in range(6)]
        self.coefficients=[[0.0]*3 for _ in range(6)]
        self.total_y=[0.0]*3

    def add_statistics(self,x,y,sign):
        for i in range(6):
            for j in range(6): self.gram[i][j]+=sign*x[i]*x[j]
            for k in range(3): self.rhs[i][k]+=sign*x[i]*y[k]
        for k in range(3): self.total_y[k]+=sign*y[k]

    def fit(self):
        matrix=[[self.gram[i][j]+(self.ridge if i==j else 0) for j in range(6)]+list(self.rhs[i]) for i in range(6)]
        for col in range(6):
            pivot=max(range(col,6),key=lambda i:abs(matrix[i][col]))
            matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
            div=matrix[col][col]
            if abs(div)<1e-14: raise ArithmeticError('Singular regularized model')
            matrix[col]=[v/div for v in matrix[col]]
            for row in range(6):
                if row==col: continue
                factor=matrix[row][col]
                matrix[row]=[a-factor*b for a,b in zip(matrix[row],matrix[col])]
        self.coefficients=[row[6:] for row in matrix]

    def observe(self,obs,after,event_id):
        x=features(obs)
        y=[after['body'][i]-obs['body'][i] for i in range(2)]+[after['fatigue']-obs['fatigue']]
        if len(self.samples)==self.window:
            old=self.samples.popleft();self.add_statistics(old['x'],old['y'],-1)
        self.samples.append(dict(event_id=event_id,x=x,y=y));self.add_statistics(x,y,1);self.fit()

    def predict(self,obs,kind='conditional'):
        if not self.samples: return [0.0]*3
        if kind=='action_mean': return [v/len(self.samples) for v in self.total_y]
        x=features(obs)
        return [sum(x[i]*self.coefficients[i][k] for i in range(6)) for k in range(3)]

    def snapshot(self):
        return deepcopy(dict(window=self.window,ridge=self.ridge,samples=list(self.samples),gram=self.gram,
                             rhs=self.rhs,coefficients=self.coefficients,total_y=self.total_y))

    @classmethod
    def restore(cls,state,config):
        obj=cls(config)
        for key,value in deepcopy(state).items(): setattr(obj,key,deque(value) if key=='samples' else value)
        return obj


class Agent:
    def __init__(self,seed,config):
        self.config={k:config[k] for k in ('fatigue_penalty','action_cost','exploration_weight','model_window','ridge')}
        self.visits=[0]*5;self.models=[LocalModel(config) for _ in range(5)]
        self.rng=random.Random(seed+710003);self.learn=True

    def decide(self,obs):
        c=self.config;prior=evaluation_loss(obs,c);candidates=[]
        for action,model in enumerate(self.models):
            delta=model.predict(obs);after=predicted_observation(obs,delta)
            benefit=prior-evaluation_loss(after,c)
            bonus=c['exploration_weight']/math.sqrt(1+self.visits[action])
            cost=c['action_cost'] if action else 0.0
            candidates.append(dict(action=action,benefit=benefit,exploration=bonus,cost=cost,
                                   score=benefit+bonus-cost,predicted_delta=delta,n=len(model.samples)))
        best=max(x['score'] for x in candidates)
        tied=[x for x in candidates if abs(x['score']-best)<=1e-12]
        chosen=self.rng.choice(tied)
        return chosen['action'],dict(candidates=candidates,chosen=chosen['action'],
                                    knowledge_samples=[len(m.samples) for m in self.models],visits=list(self.visits))

    def observe(self,obs,action,after,event_id):
        self.visits[action]+=1
        if self.learn: self.models[action].observe(obs,after,event_id)

    def snapshot(self):
        return deepcopy(dict(models=[m.snapshot() for m in self.models],visits=self.visits,
                             rng_state=self.rng.getstate(),learn=self.learn))

    @classmethod
    def restore(cls,state,config):
        obj=cls(0,config);obj.models=[LocalModel.restore(m,config) for m in state['models']]
        obj.visits=list(state['visits']);obj.rng.setstate(tuples(state['rng_state']));obj.learn=state['learn']
        return obj


def intervene(snapshot,name,tools,config):
    """Evaluator creates a fork; agent is never told which task or tool role was erased."""
    state=deepcopy(snapshot);removed=[]
    axis=0 if name in ('erase_axis0','restore_axis0') else 1 if name=='erase_axis1' else None
    if name not in config['interventions']: raise ValueError(name)
    if axis is not None:
        for action in range(1,5):
            if tools[action]['axis']==axis:
                removed.append(dict(action=action,n_samples=len(state['models'][action]['samples']),
                                    event_ids=[x['event_id'] for x in state['models'][action]['samples']],
                                    before_sha256=digest(state['models'][action])))
                state['models'][action]=LocalModel(config).snapshot()
    erased_digest=digest(state)
    if name=='restore_axis0':
        for row in removed: state['models'][row['action']]=deepcopy(snapshot['models'][row['action']])
        if digest(state)!=digest(snapshot): raise AssertionError('Restoration failed')
    return state,dict(intervention=name,removed=removed,after_erase_sha256=erased_digest,
                      final_sha256=digest(state),original_sha256=digest(snapshot))
