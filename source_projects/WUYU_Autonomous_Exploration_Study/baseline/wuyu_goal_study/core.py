from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Optional

Body = tuple[float, float]


def drive(body: Body) -> float:
    return sum(x * x for x in body)


@dataclass(frozen=True)
class Experience:
    state: str
    action: int
    next_state: str
    before: Body
    after: Body
    outcome: Optional[str]


class World:
    """Experiment-owned world. Never passed into a Learner."""

    def __init__(self, seed: int, noise: float = 0.025, swapped: bool = False):
        rng = random.Random(seed)
        self.states = [f"s_{x:06x}" for x in rng.sample(range(2**24), 9)]
        self.outcomes = [f"o_{x:06x}" for x in rng.sample(range(2**24), 2)]
        self.action_maps = [rng.sample(range(4), 4) for _ in self.states]
        self.edges: dict[tuple[str, int], str] = {}
        self.emits: dict[tuple[str, int], str] = {}
        self.effects = dict(zip(self.outcomes, [(0.0, -0.6), (-0.6, 0.0)]
                                if swapped else [(-0.6, 0.0), (0.0, -0.6)]))
        self.noise = noise
        self.effect_rng = random.Random(seed + 100_003)
        self.start = self.states[0]
        for i, s in enumerate(self.states):
            for a in range(4):
                self.edges[s, a] = s
            self.edges[s, self.action_maps[i][3]] = self.start
        # At hub all four primitive actions are routes; at other nodes slot 3 returns.
        for slot, node in enumerate([1, 2, 3, 4]):
            self.edges[self.start, self.action_maps[0][slot]] = self.states[node]
        for node, target in [(1, 5), (2, 6), (6, 5), (3, 7), (4, 8), (8, 7)]:
            self.edges[self.states[node], self.action_maps[node][0]] = self.states[target]
        for node, outcome in [(5, self.outcomes[0]), (7, self.outcomes[1])]:
            key = self.states[node], self.action_maps[node][0]
            self.edges[key] = self.start
            self.emits[key] = outcome

    def block_short_a(self) -> None:
        self.edges[self.start, self.action_maps[0][0]] = self.start

    def step(self, state: str, action: int, body: Body) -> Experience:
        key = state, action
        outcome = self.emits.get(key)
        after = body
        if outcome is not None:
            effect = self.effects[outcome]
            after = tuple(body[i] + effect[i] + self.effect_rng.gauss(0, self.noise)
                          for i in range(2))
        return Experience(state, action, self.edges[key], body, after, outcome)


def training_experiences(seed: int, repeats: int, noise: float,
                         body: Body, swapped: bool = False) -> list[Experience]:
    world = World(seed, noise, swapped)
    pairs = [(s, a) for s in world.states for a in range(4)] * repeats
    random.Random(seed + 200_003).shuffle(pairs)
    # Controlled state resets are an explicit experimental scaffold.
    return [world.step(s, a, body) for s, a in pairs]


class Learner:
    def __init__(self, variant: str, seed: int, cost: float,
                 fixed_target: Optional[str] = None):
        if variant not in {"learned", "no_goal_buffer", "cached_value",
                           "erased_effects", "fixed_target"}:
            raise ValueError(variant)
        self.variant = variant
        self.cost = cost
        self.rng = random.Random(seed + 300_007)
        self.transitions: dict[tuple[str, int], str] = {}
        self.emits: dict[tuple[str, int], str] = {}
        self.effects: dict[str, Body] = {}
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.catalog: set[str] = set()
        self.goal: Optional[str] = None
        self.fixed_target = fixed_target

    def observe(self, e: Experience) -> None:
        self.transitions[e.state, e.action] = e.next_state
        if e.outcome is None:
            self.emits.pop((e.state, e.action), None)
            return
        g = e.outcome
        self.emits[e.state, e.action] = g
        self.catalog.add(g)
        n = self.counts.get(g, 0) + 1
        old = self.effects.get(g, (0.0, 0.0))
        effect = tuple(e.after[i] - e.before[i] for i in range(2))
        if self.variant != "erased_effects":
            self.effects[g] = tuple(old[i] + (effect[i] - old[i]) / n for i in range(2))
            old_value = self.values.get(g, 0.0)
            self.values[g] = old_value + (drive(e.before) - drive(e.after) - old_value) / n
        self.counts[g] = n
        if self.goal == g:
            self.goal = None

    def path(self, state: str, goal: str) -> Optional[list[int]]:
        """Shortest observed route, including the outcome-producing action."""
        queue = deque([(state, [])])
        visited = {state}
        while queue:
            s, actions = queue.popleft()
            for a in range(4):
                key = s, a
                if key not in self.transitions:
                    continue
                route = actions + [a]
                if self.emits.get(key) == goal:
                    return route
                # Do not plan through another consuming outcome.
                if key in self.emits:
                    continue
                nxt = self.transitions[key]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, route))
        return None

    def utility(self, goal: str, body: Body) -> float:
        if self.variant == "cached_value":
            return self.values[goal]
        effect = self.effects[goal]
        return drive(body) - drive(tuple(body[i] + effect[i] for i in range(2)))

    def decide(self, state: str, body: Body) -> tuple[Optional[int], Optional[str], dict]:
        paths = {g: self.path(state, g) for g in sorted(self.catalog)}
        paths = {g: p for g, p in paths.items() if p}
        if not paths:
            return None, None, {"reason": "no_observed_reachable_outcome"}
        if self.variant == "fixed_target":
            chosen = self.fixed_target if self.fixed_target in paths else None
            scores = {}
        elif self.variant == "erased_effects":
            chosen = self.goal if self.goal in paths else self.rng.choice(sorted(paths))
            scores = {}
        else:
            scores = {g: self.utility(g, body) - self.cost * len(paths[g]) for g in paths}
            if self.variant != "no_goal_buffer" and self.goal in scores and scores[self.goal] > 0:
                chosen = self.goal
            else:
                chosen = max(scores, key=scores.get)
                if scores[chosen] <= 0:
                    chosen = None
        self.goal = None if self.variant == "no_goal_buffer" else chosen
        info = {"scores": scores, "path_lengths": {g: len(p) for g, p in paths.items()},
                "chosen": chosen, "goal_buffer": self.goal}
        if chosen is None:
            info["reason"] = "no_positive_goal" if scores else "unreachable_fixed_goal"
            return None, None, info
        return paths[chosen][0], chosen, info

    def snapshot(self) -> dict:
        return {"variant": self.variant, "catalog": sorted(self.catalog),
                "effects": self.effects, "cached_values": self.values,
                "outcome_samples": self.counts,
                "transitions": [[s, a, n] for (s, a), n in sorted(self.transitions.items())],
                "emits": [[s, a, g] for (s, a), g in sorted(self.emits.items())]}
