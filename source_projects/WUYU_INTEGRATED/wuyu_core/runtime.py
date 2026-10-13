"""Explicit observe -> decide -> feedback lifecycle and versioned individual state."""
import copy
from pathlib import Path
import json
import numpy as np
from .features import encode
from .memory import ExperienceMemory, softmax
from .network import Network
from .predictor import Predictor
from .world import World
from .state import read_tree, write_tree

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'wuyu.integrated/1'
MODES = ('prior', 'online', 'replay', 'shadow')


def model_path(initialization=0):
    step = {0: 1536, 1: 2048, 2: 2048, 3: 1536, 4: 2048}[initialization]
    return ROOT / f'legacy/WUYU_08/results/model_graph_varied_{initialization}_{step}.npz'


def load_network(initialization=0):
    s = read_tree(model_path(initialization))
    return Network.from_state(s['network'] if 'network' in s else s)


class Individual:
    """One organism owns every mutable state and random stream.

    No clock runs outside calls to step. A cohort is a list of independent
    instances, making extraction/reordering semantics explicit.
    """
    def __init__(self, net, seed, initialization=0, mode='prior'):
        if mode not in MODES: raise ValueError(mode)
        self.net = Network.from_state(net.state())
        self.seed, self.initialization, self.mode = int(seed), initialization, mode
        initializations = json.loads((ROOT/'legacy/WUYU_08/CONFIG.json').read_text())['initializations']
        self.memory = ExperienceMemory(seed, initializations[initialization])
        self.h, self.prev = net.zero(1), np.zeros((1, 9))
        self.predictor = Predictor(seed * 17 + initialization * 100003)
        self.rng = np.random.default_rng(seed * 41 + initialization)
        self.tick = 0; self.observed = -1; self.pending = None
        self.bias_strength = 1.; self.warmup = 32
        self.last_obs = None

    def ingest(self, obs):
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (1, 21): raise ValueError('expected one 21-channel observation')
        if self.observed == self.tick:
            if not np.array_equal(obs, self.last_obs):
                raise ValueError('observation changed within the same event')
            return
        pos, _, need, _, body = self.memory.decode(obs[0])
        if body.min() > .001:
            self.memory.seen[pos] += 1; self.memory.requirement[pos] = need
        self.last_obs = obs.copy(); self.observed = self.tick

    def decide(self, obs, greedy=False, forced=None):
        if self.pending is not None: raise RuntimeError('feedback required before next decision')
        self.ingest(obs)
        x = encode(obs, self.prev, [self.memory])
        r = self.net.forward(x, self.h)
        context = np.concatenate([obs[0], self.prev[0], x['node'][0, obs[0,:6].argmax()]])
        predictions, candidates = self.predictor.predict(context)
        # Declared external value prior. It evaluates short-term learned outcomes;
        # it cannot search a dependency chain or invent a target label.
        need = (1 - obs[0,14:16]) ** 2 + .02
        utility = predictions[:,:2] @ need + .05 * predictions[:,2:4].sum(1) + .05 * predictions[:,4]
        bias = np.clip(utility * self.bias_strength, -1., 1.)
        use = self.mode in ('online', 'replay') and self.predictor.seen >= self.warmup
        probability = softmax(np.log(np.maximum(r['ap'][0], 1e-12)) + (bias if use else 0.))
        # Consume one draw in every mode, including forced histories, for coupling.
        draw = self.rng.random()
        action = int(probability.argmax() if greedy else min(np.searchsorted(probability.cumsum(), draw), 6))
        if forced is not None:
            if int(forced) not in range(7): raise ValueError('invalid action')
            action = int(forced)
        self.h = r['h'].copy()
        self.pending = dict(action=action, obs=obs.copy(), x=candidates[action].copy(),
                            prediction=predictions[action].copy(), probability=probability.copy(),
                            bias=bias.copy(), bias_applied=bool(use))
        return action

    def feedback(self, next_obs):
        if self.pending is None: raise RuntimeError('decision required before feedback')
        q = self.pending; before = q['obs'][0]; after = np.asarray(next_obs)[0]
        if after.shape != (21,): raise ValueError('invalid feedback')
        alive = before[14:16].min() > .001
        stage_delta = (after[6:9].argmax() - before[6:9].argmax()) / 2.
        y = np.concatenate([(after[14:16]-before[14:16])/.28,
                            after[12:14]-before[12:14], [stage_delta, float(after[16:20].sum() > .5)]])
        if alive and self.mode != 'prior':
            self.predictor.feedback(q['x'], y, replay=self.mode in ('replay','shadow'))
        self.memory.feedback(dict(action=q['action'], obs=before), after)
        self.prev = np.concatenate([np.eye(7)[[q['action']]], ((after[14:16]-before[14:16])/.28)[None]], axis=1)
        q.update(target=y, prediction_mse=float(np.mean((q['prediction']-y)**2)), alive=bool(alive))
        self.pending = None; self.tick += 1
        return q

    def state(self):
        return dict(seed=self.seed, initialization=self.initialization, mode=self.mode,
                    network=self.net.state(), memory=self.memory.state(), predictor=self.predictor.state(),
                    rng=self.rng.bit_generator.state, h=self.h.copy(), prev=self.prev.copy(),
                    tick=self.tick, observed=self.observed, last_obs=copy.deepcopy(self.last_obs),
                    pending=copy.deepcopy(self.pending), bias_strength=self.bias_strength, warmup=self.warmup)

    @classmethod
    def from_state(cls, s):
        a = cls.__new__(cls)
        for k in ('seed','initialization','mode','h','prev','tick','observed','last_obs','pending','bias_strength','warmup'):
            setattr(a,k,copy.deepcopy(s[k]))
        a.net = Network.from_state(s['network']); a.memory = ExperienceMemory.from_state(s['memory'])
        a.predictor = Predictor.from_state(s['predictor']); a.rng = np.random.default_rng()
        a.rng.bit_generator.state = s['rng']
        return a


class Session:
    def __init__(self, world, individual, lineage=None):
        if world.n != 1: raise ValueError('each session owns exactly one world')
        self.world, self.individual = world, individual
        self.lineage = copy.deepcopy(lineage)

    @classmethod
    def create(cls, seed=990001, initialization=0, mode='prior', total=1536, shocks=True):
        return cls(World([seed], total, shocks=shocks), Individual(load_network(initialization), seed, initialization, mode))

    def step(self, greedy=False, forced=None):
        if self.world.t >= self.world.total: raise StopIteration('declared world horizon reached')
        action = self.individual.decide(self.world.observation(), greedy, forced)
        outcome = self.world.step(np.array([action]))
        record = self.individual.feedback(self.world.observation())
        record['outcome'] = outcome; record['event'] = self.world.t
        return record

    def state(self):
        return dict(schema=SCHEMA, profile='integrated', world=self.world.state(), individual=self.individual.state(), lineage=copy.deepcopy(self.lineage))

    def save(self, path): write_tree(path, self.state())

    @classmethod
    def from_state(cls, s):
        if s.get('schema') != SCHEMA or s.get('profile') != 'integrated':
            raise ValueError('unsupported checkpoint schema/profile')
        return cls(World.from_state(copy.deepcopy(s['world'])), Individual.from_state(s['individual']),s.get('lineage'))

    @classmethod
    def load(cls, path): return cls.from_state(read_tree(path))

    def fork(self): return self.from_state(self.state())
