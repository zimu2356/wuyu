"""NumPy-only closed-loop foraging and history-conditioned plasticity.

These are deliberately small controllers with an externally specified objective.
No variable in this file is a measure of consciousness, desire, or personhood.
"""
from dataclasses import dataclass
import copy
import json
import numpy as np

K = 6
FEATURES = 9
TRAIN_KINDS = ("slow", "fast", "noisy")
TEST_KINDS = ("in_distribution", "rapid", "high_noise", "drift", "stationary", "mixed")
FEATURE_NAMES = ["bias", "absolute_error", "fast_error_magnitude", "slow_error_magnitude",
                 "fast_slow_disagreement", "feedback_change", "visit_age",
                 "error_times_slow_error", "experience_count"]


@dataclass
class Scenario:
    quality: np.ndarray
    noise: np.ndarray
    explore: np.ndarray
    random_action: np.ndarray
    initial_position: np.ndarray
    kinds: tuple
    seeds: tuple

    @property
    def horizon(self):
        return self.quality.shape[0]

    @property
    def size(self):
        return self.quality.shape[1]


def make_scenario(kinds, per_kind, horizon, seed_base):
    qualities, noises, explores, actions, positions, labels, seeds = [], [], [], [], [], [], []
    for ki, kind in enumerate(kinds):
        for j in range(per_kind):
            seed = seed_base + ki * 1000 + j
            rng = np.random.default_rng(seed)
            actual = TRAIN_KINDS[j % 3] if kind == "in_distribution" else kind
            q = rng.permutation(np.linspace(.15, .85, K))
            out = np.empty((horizon, K))
            sigma = np.full(horizon, .08)
            next_change = {"slow": 128, "fast": 48, "noisy": 128,
                           "rapid": 20, "high_noise": 160,
                           "stationary": horizon + 1}.get(actual, 64)
            if actual in ("noisy", "high_noise"):
                sigma[:] = .25 if actual == "noisy" else .40
            interval = next_change
            for t in range(horizon):
                if actual == "drift":
                    q = np.clip(q + rng.normal(0, .025, K), .1, .9)
                elif actual == "mixed":
                    block = (t // 192) % 3
                    sigma[t] = (.08, .35, .12)[block]
                    if t == next_change:
                        q = rng.permutation(np.linspace(.15, .85, K))
                        next_change += int(rng.integers(*((80, 161), (120, 201), (16, 41))[block]))
                elif t == next_change:
                    q = rng.permutation(np.linspace(.15, .85, K))
                    next_change += max(8, int(interval * rng.uniform(.75, 1.25)))
                out[t] = q
            qualities.append(out)
            noises.append(rng.normal(size=(horizon, K)) * sigma[:, None])
            explores.append(rng.random(horizon))
            actions.append(rng.integers(0, K, horizon))
            positions.append(rng.integers(0, K))
            labels.append(kind)
            seeds.append(seed)
    return Scenario(np.stack(qualities, axis=1), np.stack(noises, axis=1),
                    np.stack(explores, axis=1), np.stack(actions, axis=1),
                    np.array(positions), tuple(labels), tuple(seeds))


class World:
    def __init__(self, scenario, candidates=1):
        self.scenario = scenario
        self.index = np.tile(np.arange(scenario.size), candidates)
        self.stock = np.ones((len(self.index), K))
        self.position = scenario.initial_position[self.index].copy()
        self.t = 0

    def observe(self):
        # No hidden quality, scenario name, change point, or seed is exposed.
        return {"stock": self.stock.copy(), "position": self.position.copy()}

    def costs(self):
        d = np.abs(np.arange(K)[None, :] - self.position[:, None])
        return .035 * np.minimum(d, K - d)

    def step(self, action):
        rows = np.arange(len(action))
        q = self.scenario.quality[self.t, self.index]
        noisy = q[rows, action] + self.scenario.noise[self.t, self.index, action]
        values = q * self.stock - self.costs()
        expected = values[rows, action]
        reward = noisy * self.stock[rows, action] - self.costs()[rows, action]
        selected_quality = q[rows, action]
        oracle_gap = np.max(values, axis=1) - expected
        self.stock[rows, action] *= .30
        self.stock += .08 * (1. - self.stock)
        self.position = action.copy()
        self.t += 1
        return noisy, reward, selected_quality, oracle_gap


class Agent:
    def __init__(self, n, mode, parameter):
        self.mode = mode
        self.parameter = np.asarray(parameter, dtype=float)
        self.q = np.full((n, K), .5)
        self.count = np.zeros((n, K), dtype=np.int64)
        self.age = np.zeros((n, K), dtype=np.int64)
        self.last = np.full((n, K), .5)
        self.fast = np.zeros((n, K))
        self.slow = np.zeros((n, K))
        self.change = np.zeros((n, K))
        self.memory = np.zeros((n, K, int(parameter))) if mode == "memory" else None

    def choose(self, observation, explore, random_action, t):
        stock, pos = observation["stock"], observation["position"]
        d = np.abs(np.arange(K)[None, :] - pos[:, None])
        cost = .035 * np.minimum(d, K - d)
        bonus = .05 * np.sqrt(np.log(t + 2) / (self.count + 1))
        score = self.q * stock - cost + bonus
        action = np.argmax(score, axis=1)
        return np.where(explore < .08, random_action, action)

    def features(self, action, feedback):
        r = np.arange(len(action))
        err = np.clip(feedback - self.q[r, action], -2, 2)
        f, s, c = self.fast[r, action], self.slow[r, action], self.change[r, action]
        return np.stack([np.ones_like(err), np.abs(err), np.abs(f), np.abs(s),
                         np.abs(f-s), c, np.minimum(self.age[r, action]/64, 1),
                         err*s, np.minimum(self.count[r, action]/16, 1)], axis=1)

    def learn(self, action, feedback):
        r = np.arange(len(action))
        error = feedback - self.q[r, action]
        if self.mode in ("adaptive", "state_equivalent"):
            x = self.features(action, feedback)
            z = np.sum(x * self.parameter, axis=1)
            alpha = .01 + .94 / (1 + np.exp(-np.clip(z, -12, 12)))
        elif self.mode in ("fixed", "constant_gate"):
            alpha = np.broadcast_to(self.parameter, (len(action),)).copy()
        else:
            alpha = 1 / np.minimum(self.count[r, action] + 1, self.memory.shape[2])

        if self.mode == "memory":
            width = self.memory.shape[2]
            self.memory[r, action, self.count[r, action] % width] = feedback
            qnew = self.memory[r, action].sum(axis=1) / np.minimum(self.count[r, action]+1, width)
        elif self.mode == "state_equivalent":
            # Explicit state-transition interpretation; same mathematical operation.
            previous_state = self.q[r, action]
            qnew = previous_state + alpha * (feedback - previous_state)
        else:
            qnew = self.q[r, action] + alpha * error
        self.q[r, action] = np.clip(qnew, 0, 1)
        clipped_error = np.clip(error, -2, 2)
        self.fast[r, action] = .65*self.fast[r, action] + .35*clipped_error
        self.slow[r, action] = .95*self.slow[r, action] + .05*clipped_error
        delta = np.where(self.count[r, action] > 0, np.abs(feedback-self.last[r, action]), 0)
        self.change[r, action] = .85*self.change[r, action] + .15*np.minimum(delta, 2)
        self.last[r, action] = feedback
        self.count[r, action] += 1
        self.age += 1
        self.age[r, action] = 0
        return alpha

    def erase_history_statistics(self):
        self.fast.fill(0)
        self.slow.fill(0)
        self.change.fill(0)

    def copy(self):
        return copy.deepcopy(self)


def rollout(scenario, mode, parameter, capture=False, initial=None, start=0, stop=None):
    """Candidates are batched; returns metrics per candidate and independent life."""
    parameter = np.asarray(parameter)
    candidates = parameter.shape[0] if mode in ("adaptive", "state_equivalent") and parameter.ndim == 2 else 1
    n = candidates * scenario.size
    if mode in ("adaptive", "state_equivalent"):
        theta = np.repeat(parameter.reshape(candidates, FEATURES), scenario.size, axis=0)
    else:
        theta = parameter
    if initial is None:
        world, agent = World(scenario, candidates), Agent(n, mode, theta)
    else:
        world, agent = initial
        if world.t != start:
            raise ValueError("checkpoint time must equal start")
    stop = scenario.horizon if stop is None else stop
    metrics = {k: np.zeros(n) for k in ("reward", "prediction_mse", "oracle_gap", "alpha")}
    trajectory = []
    action_counts = np.zeros((n, K), dtype=np.int64)
    segments = []
    rows = np.arange(n)
    for t in range(start, stop):
        obs = world.observe()
        action = agent.choose(obs, scenario.explore[t, world.index], scenario.random_action[t, world.index], t)
        prediction = agent.q[rows, action].copy()
        feedback, reward, quality, gap = world.step(action)
        alpha = agent.learn(action, feedback)
        values = (reward, (prediction-quality)**2, gap, alpha)
        for key, value in zip(metrics, values):
            metrics[key] += value
        action_counts[rows, action] += 1
        if capture:
            trajectory.append([float(reward[0]), float(prediction[0]), float(quality[0]), float(alpha[0]), int(action[0])])
        if (t-start+1) % 256 == 0:
            segments.append(metrics["reward"].copy())
    duration = stop - start
    for key in metrics:
        metrics[key] = (metrics[key]/duration).reshape(candidates, scenario.size)
    p = action_counts/duration
    metrics["visit_entropy"] = (-np.sum(p*np.log(np.maximum(p, 1e-15)), axis=1)).reshape(candidates, scenario.size)
    if segments:
        segments = np.diff(np.stack([np.zeros(n)] + segments), axis=0)/256
    return {"metrics": metrics, "world": world, "agent": agent,
            "trajectory": np.asarray(trajectory), "segments": np.asarray(segments)}


def save_checkpoint(path, world, agent):
    payload = {"world_stock": world.stock, "world_position": world.position,
               "world_index": world.index, "parameter": agent.parameter}
    for key in ("q", "count", "age", "last", "fast", "slow", "change"):
        payload["agent_"+key] = getattr(agent, key)
    if agent.memory is not None:
        payload["agent_memory"] = agent.memory
    for key in ("quality", "noise", "explore", "random_action", "initial_position"):
        payload["scenario_"+key] = getattr(world.scenario, key)
    payload["metadata"] = np.array(json.dumps({"mode": agent.mode, "t": world.t,
        "kinds": world.scenario.kinds, "seeds": world.scenario.seeds}))
    np.savez_compressed(path, **payload)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        scenario = Scenario(*(data["scenario_"+key].copy() for key in
            ("quality", "noise", "explore", "random_action", "initial_position")),
            tuple(meta["kinds"]), tuple(meta["seeds"]))
        n = len(data["world_index"])
        world = World(scenario, n//scenario.size)
        world.stock = data["world_stock"].copy()
        world.position = data["world_position"].copy()
        world.index = data["world_index"].copy()
        world.t = meta["t"]
        agent = Agent(n, meta["mode"], data["parameter"].copy())
        for key in ("q", "count", "age", "last", "fast", "slow", "change"):
            setattr(agent, key, data["agent_"+key].copy())
        if "agent_memory" in data:
            agent.memory = data["agent_memory"].copy()
    return world, agent
