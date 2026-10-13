"""Private recurrent consequence learning; observable targets only.

The replay stores raw past inputs, not stale hidden states. The four-step prefix
is recomputed with current weights; truncated gradients flow through that prefix.
This is a new generalisation of the WUYU_03--06 mechanisms, not weight merging.
"""
import copy
import numpy as np


class Predictor:
    def __init__(self, seed, input_size=68, hidden=16, outputs=6, capacity=64, burn=4):
        init = np.random.default_rng(seed)
        self.p = dict(W=init.normal(0, .07, (input_size, hidden)),
                      R=init.normal(0, .08, (hidden, hidden)), b=np.zeros(hidden),
                      V=init.normal(0, .02, (hidden, outputs)), c=np.zeros(outputs))
        self.h = np.zeros(hidden)
        self.rng = np.random.default_rng(seed + 1000003)
        self.capacity, self.burn = capacity, burn
        self.history, self.memory = [], []
        self.seen = self.updates = self.replay_updates = 0
        self.rate = .02

    def sequence(self, xs, h0=None):
        p = self.p
        h = np.zeros_like(self.h) if h0 is None else h0.copy()
        states, previous = [], []
        for x in xs:
            previous.append(h)
            h = np.tanh(x @ p['W'] + h @ p['R'] + p['b'])
            states.append(h)
        y = np.tanh(h @ p['V'] + p['c'])
        return y, states, previous

    def predict(self, context):
        xs = np.concatenate([np.repeat(context[None], 7, axis=0), np.eye(7)], axis=1)
        h = np.tanh(xs @ self.p['W'] + self.h @ self.p['R'] + self.p['b'])
        return np.tanh(h @ self.p['V'] + self.p['c']), xs

    def gradients(self, xs, y):
        pred, states, previous = self.sequence(xs)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        dy = (pred - y) / len(y) * (1 - pred ** 2)
        g['V'] = np.outer(states[-1], dy); g['c'] = dy
        dh = dy @ self.p['V'].T
        for x, h, old in zip(xs[::-1], states[::-1], previous[::-1]):
            dz = dh * (1 - h ** 2)
            g['W'] += np.outer(x, dz); g['R'] += np.outer(old, dz); g['b'] += dz
            dh = dz @ self.p['R'].T
        return .5 * np.mean((pred - y) ** 2), g

    def learn(self, xs, y):
        _, grad = self.gradients(xs, y)
        norm = np.sqrt(sum(np.sum(g*g) for g in grad.values()))
        for k, g in grad.items():
            self.p[k] -= self.rate * g * min(1., 5 / max(norm, 1e-12))
        self.updates += 1

    def feedback(self, x, y, replay=False):
        # Advance the live recurrent state before updating parameters. This state
        # records the actual action, never any counterfactual candidate action.
        _, states, _ = self.sequence([x], self.h)
        self.h = states[-1].copy()
        xs = np.array(self.history + [x])
        # Sample only earlier completed experience; new event is inserted below.
        old = self.memory[int(self.rng.integers(len(self.memory)))] if replay and self.memory else None
        self.learn(xs, y)
        if old is not None:
            self.learn(old['xs'], old['y']); self.replay_updates += 1
        item = dict(xs=xs.copy(), y=y.copy(), event=self.seen)
        self.seen += 1
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            slot = int(self.rng.integers(self.seen))
            if slot < self.capacity: self.memory[slot] = item
        self.history = (self.history + [x.copy()])[-self.burn:]

    def state(self):
        s = copy.deepcopy(self.__dict__); s['rng'] = self.rng.bit_generator.state
        return s

    @classmethod
    def from_state(cls, s):
        x = cls.__new__(cls); x.__dict__.update(copy.deepcopy(s))
        x.rng = np.random.default_rng(); x.rng.bit_generator.state = s['rng']
        return x
