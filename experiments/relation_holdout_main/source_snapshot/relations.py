"""Observable movement beliefs, with explicit ring/action priors.

No World, schedule, obstacle flag, goal, or hidden route is accepted. The
distance conversion is a designed input transformation, not learned planning.
"""
from copy import deepcopy
import numpy as np

SCHEMA = 'observed-movement-relations/1'


class MovementRelations:
    def __init__(self, rate=.45, recovery=256., floor=.04):
        self.schema = SCHEMA
        self.rate, self.recovery, self.floor = float(rate), float(recovery), float(floor)
        self.prior = np.zeros((6, 2, 6))
        for p in range(6):
            for a, direction in enumerate((-1, 1)):
                self.prior[p, a, (p + direction) % 6] = 1.
        self.q = self.prior.copy()
        self.count = np.zeros((6, 2), int)
        self.last = np.full((6, 2), -1, int)
        self.brier_sum = np.zeros((6, 2))
        self.events = []

    def distribution(self, tick):
        age = np.maximum(0, int(tick) - self.last)
        weight = np.exp(-age / self.recovery)
        return self.prior + weight[:, :, None] * (self.q - self.prior)

    def success(self, tick):
        q = self.distribution(tick)
        return np.array([[q[p, a, (p + (-1 if a == 0 else 1)) % 6]
                          for a in range(2)] for p in range(6)])

    def distances(self, pos, tick):
        # Expected repeated-attempt cost along each declared ring direction.
        # Unknown edges retain the declared original prior. This does not read
        # a shortest route or an obstacle map from the evaluation world.
        costs = 1. / np.maximum(self.success(tick), self.floor)
        result = np.zeros((6, 2))
        for channel, action in enumerate((1, 0)):  # original clockwise, counterclockwise
            current, distance = int(pos), 0.
            direction = 1 if action == 1 else -1
            for _ in range(5):
                distance += costs[current, action]
                current = (current + direction) % 6
                result[current, channel] = min(distance / 5., 2.)
        return result

    def update(self, before, action, after, tick):
        # Terminal transitions are allowed; after death no new evidence exists.
        if int(action) not in (0, 1) or np.min(before[14:16]) <= .001:
            return None
        p, n = int(np.argmax(before[:6])), int(np.argmax(after[:6]))
        a = int(action)
        predicted = self.distribution(tick)[p, a].copy()
        observed = np.eye(6)[n]
        loss = float(np.sum((predicted - observed) ** 2))
        self.q[p, a] = (1. - self.rate) * predicted + self.rate * observed
        self.count[p, a] += 1
        self.last[p, a] = int(tick)
        self.brier_sum[p, a] += loss
        event = dict(tick=int(tick), position=p, action=a, next_position=n,
                     predicted=predicted, brier=loss)
        self.events.append(deepcopy(event))
        self.events = self.events[-128:]
        return event

    def state(self):
        return deepcopy(vars(self))

    @classmethod
    def restore(cls, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('incompatible movement relation state')
        obj = cls.__new__(cls)
        obj.__dict__.update(deepcopy(state))
        return obj
