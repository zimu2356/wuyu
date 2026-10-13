"""Q/replay/Dyna controls. Models learn only supplied observed transitions."""
import numpy as np


class Control:
    def __init__(self, n, states=25, mode='q', extra=8, seed=1, capacity=2048, epsilon=.08):
        self.n, self.states, self.mode, self.extra = n, states, mode, extra
        self.alpha, self.gamma, self.epsilon = .15, .95, epsilon
        self.q = np.zeros((n, states, 5))
        self.mr = np.zeros((n, states*5))
        self.mn = np.zeros((n, states*5), dtype=np.int16)
        self.seen = np.zeros((n, states*5), bool)
        self.keys = np.zeros((n, states*5), dtype=np.int16)
        self.key_count = np.zeros(n, dtype=np.int16)
        self.capacity = capacity
        self.bs = np.zeros((n, capacity), dtype=np.int16)
        self.ba = np.zeros((n, capacity), dtype=np.int8)
        self.br = np.zeros((n, capacity))
        self.bn = np.zeros((n, capacity), dtype=np.int16)
        self.count = 0
        self.real_updates = 0
        self.extra_updates = 0
        self.rng = np.random.default_rng(seed)

    def choose(self, state, explore, random_action, tie, greedy=False):
        values = self.q[np.arange(self.n), state]
        candidates = np.where(values == values.max(axis=1, keepdims=True), tie, -1.)
        action = candidates.argmax(axis=1)
        return action if greedy else np.where(explore < self.epsilon, random_action, action)

    def backup(self, state, action, reward, after):
        rows = np.arange(self.n)
        old = self.q[rows, state, action]
        target = reward+self.gamma*self.q[rows, after].max(axis=1)
        self.q[rows, state, action] += self.alpha*(target-old)

    def remember(self, state, action, reward, after):
        rows = np.arange(self.n)
        key = state*5+action
        fresh = ~self.seen[rows, key]
        rr = rows[fresh]
        self.keys[rr, self.key_count[rr]] = key[fresh]
        self.key_count += fresh.astype(np.int16)
        self.mr[rows, key] = np.where(fresh, reward,
                                     self.mr[rows, key]+self.alpha*(reward-self.mr[rows, key]))
        self.mn[rows, key] = after
        self.seen[rows, key] = True
        pos = self.count % self.capacity
        self.bs[:, pos], self.ba[:, pos] = state, action
        self.br[:, pos], self.bn[:, pos] = reward, after
        self.count += 1

    def update(self, state, action, reward, after):
        self.remember(state, action, reward, after)
        self.backup(state, action, reward, after)
        self.real_updates += 1
        rows = np.arange(self.n)
        for _ in range(self.extra if self.mode != 'q' else 0):
            if self.mode == 'replay':
                ix = (self.rng.random(self.n)*min(self.count, self.capacity)).astype(int)
                self.backup(self.bs[rows, ix], self.ba[rows, ix], self.br[rows, ix], self.bn[rows, ix])
            elif self.mode == 'dyna':
                ix = (self.rng.random(self.n)*self.key_count).astype(int)
                key = self.keys[rows, ix]
                self.backup(key//5, key % 5, self.mr[rows, key], self.mn[rows, key])
            else:
                raise ValueError(self.mode)
            self.extra_updates += 1


def encode(charge, context=None):
    state = charge[:, 0]*5+charge[:, 1]
    return state if context is None else 25*np.asarray(context)+state
