"""Tabular Q learning in a world with functional, consumable sensing tools.

Tool kinds, rewards, decoding, maintenance and capacities are designer supplied.
The learner chooses macro-actions, not sensor encodings or its own reward.
"""
from dataclasses import dataclass
import copy
import numpy as np

ACTIONS = ("basic", "use_A", "use_B", "maintain_A", "maintain_B")
CAPACITY = 4


@dataclass
class Events:
    target: np.ndarray
    sensory_uniform: np.ndarray
    explore: np.ndarray
    random_action: np.ndarray
    tie: np.ndarray
    seeds: np.ndarray

    @property
    def horizon(self):
        return self.target.shape[0]

    @property
    def count(self):
        return self.target.shape[1]

    def columns(self, indices):
        return Events(self.target[:, indices], self.sensory_uniform[:, indices],
                      self.explore[:, indices], self.random_action[:, indices],
                      self.tie[:, indices], self.seeds[indices])


def events(count, horizon, seed_base):
    items = []
    for j in range(count):
        rng = np.random.default_rng(seed_base+j)
        items.append((rng.integers(0, 2, horizon), rng.random((horizon, 3)),
                      rng.random(horizon), rng.integers(0, 5, horizon),
                      rng.random((horizon, 5))))
    return Events(*(np.stack([a[k] for a in items], axis=1) for k in range(5)),
                  np.arange(seed_base, seed_base+count))


class World:
    def __init__(self, count, quality=(.9, .9), support=None):
        self.charge = np.zeros((count, 2), dtype=np.int8)
        self.quality = np.broadcast_to(np.asarray(quality), (count, 2)).copy()
        self.support = np.zeros((count, 2), bool) if support is None else np.array(support, bool)

    def observe(self):
        return self.charge.copy()

    def step(self, action, target, uniforms):
        n = len(action)
        rows = np.arange(n)
        reward = np.full(n, -.2)
        cue = np.full(n, -1, dtype=np.int8)
        channel = np.clip(action, 0, 2)
        available = (action == 0)
        for tool in range(2):
            use = action == tool+1
            available |= use & ((self.charge[:, tool] > 0) | self.support[:, tool])
        reliability = np.concatenate([np.full((n, 1), .60), self.quality], axis=1)
        correct = uniforms[rows, channel] < reliability[rows, channel]
        cue[available] = np.where(correct, target, 1-target)[available]
        # The macro-action decoder answers using the revealed cue.
        reward[available] = np.where(cue == target, 1., -1.)[available]
        for tool in range(2):
            use = (action == tool+1) & (self.charge[:, tool] > 0)
            self.charge[use, tool] -= 1
            self.charge[action == tool+3, tool] = CAPACITY
        return reward, cue


class Learner:
    def __init__(self, count, alpha=.08, gamma=.95, mode="q", epsilon=.08):
        self.q = np.zeros((count, 5, 5, 5))
        self.alpha, self.gamma, self.mode, self.epsilon = float(alpha), float(gamma), mode, epsilon

    def choose(self, observation, explore, random_action, tie):
        rows = np.arange(len(observation))
        if self.mode == "random":
            return random_action.copy()
        if self.mode in ("script_A", "script_B"):
            k = 0 if self.mode == "script_A" else 1
            return np.where(observation[:, k] > 0, k+1, k+3)
        values = self.q[rows, observation[:, 0], observation[:, 1]]
        # Random numbers are used only to break exact ties, not to perturb Q values.
        candidates = np.where(values == values.max(axis=1, keepdims=True), tie, -1.)
        greedy = candidates.argmax(axis=1)
        return np.where(explore < self.epsilon, random_action, greedy)

    def update(self, observation, action, reward, after):
        if self.mode != "q" or self.alpha == 0.:
            return
        rows = np.arange(len(action))
        old = self.q[rows, observation[:, 0], observation[:, 1], action]
        future = self.q[rows, after[:, 0], after[:, 1]].max(axis=1)
        self.q[rows, observation[:, 0], observation[:, 1], action] += self.alpha*(reward+self.gamma*future-old)

    def copy(self):
        return copy.deepcopy(self)


def run(ev, world, learner, start=0, stop=None, forced=None):
    stop = ev.horizon if stop is None else stop
    rewards = np.zeros((stop-start, ev.count))
    actions = np.zeros_like(rewards, dtype=np.int8)
    cues = np.zeros_like(actions)
    for t in range(start, stop):
        obs = world.observe()
        action = (forced[t-start] if forced is not None else
                  learner.choose(obs, ev.explore[t], ev.random_action[t], ev.tie[t]))
        reward, cue = world.step(action, ev.target[t], ev.sensory_uniform[t])
        learner.update(obs, action, reward, world.observe())
        rewards[t-start], actions[t-start], cues[t-start] = reward, action, cue
    return {"reward": rewards, "action": actions, "cue": cues}


def checkpoint(path, world, learner):
    np.savez_compressed(path, charge=world.charge, quality=world.quality, support=world.support,
                        q=learner.q, alpha=learner.alpha, gamma=learner.gamma,
                        epsilon=learner.epsilon, mode=np.array(learner.mode))


def restore(path):
    with np.load(path, allow_pickle=False) as data:
        n = len(data["charge"])
        world = World(n, data["quality"], data["support"])
        world.charge = data["charge"].copy()
        learner = Learner(n, data["alpha"], data["gamma"], str(data["mode"]), data["epsilon"])
        learner.q = data["q"].copy()
    return world, learner


def branch(world, learner, kind, preferred):
    w, a = copy.deepcopy(world), learner.copy()
    rows = np.arange(len(w.charge))
    if kind == "erase_values":
        a.q.fill(0)
    if kind in ("frozen", "break_frozen", "replay_break"):
        a.alpha = 0.
    if kind in ("break_preferred", "break_frozen", "replay_break"):
        w.quality[rows, preferred] = .50
    if kind == "external_support":
        w.support[rows, preferred] = True
    return w, a


def metrics(output, preferred, tail=512):
    act, reward = output["action"][-tail:], output["reward"][-tail:]
    uses = (act == 1).sum(axis=0)+(act == 2).sum(axis=0)
    fraction = np.divide((act == 1).sum(axis=0), uses,
                         out=np.full(len(uses), .5), where=uses > 0)
    return {"reward": reward.mean(axis=0), "a_share": fraction,
            "preferred_use": (act == preferred[None, :]+1).mean(axis=0),
            "preferred_maintenance": (act == preferred[None, :]+3).mean(axis=0),
            "any_maintenance": (act >= 3).mean(axis=0),
            "invalid_use": ((act >= 1)&(act <= 2)&(output["cue"][-tail:] == -1)).mean(axis=0)}
