"""Finite experience acquisition. Collectors never receive a World or evaluator."""
from collections import Counter, deque
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import math
import random
import time

from baseline.wuyu_goal_study.core import Experience, Learner, World, training_experiences


def history_hash(events):
    h = hashlib.sha256()
    for e in events:
        h.update((json.dumps(asdict(e), sort_keys=True, separators=(',', ':')) + '\n').encode())
    return h.hexdigest()


class Collector:
    def __init__(self, kind, seed, config):
        if kind not in ('random', 'least_visited'):
            raise ValueError(kind)
        self.kind = kind
        offset = config['random_collector_seed_offset'] if kind == 'random' else config['directed_collector_seed_offset']
        self.rng = random.Random(seed + offset)
        self.transitions = {}
        self.counts = Counter()

    def observe(self, state, action, next_state):
        self.transitions[state, action] = next_state
        self.counts[state, action] += 1

    def choose(self, state):
        if self.kind == 'random':
            return self.rng.randrange(4)
        paths = {state: []}
        queue = deque([state])
        while queue:
            s = queue.popleft()
            for a in range(4):
                nxt = self.transitions.get((s, a))
                if nxt is not None and nxt not in paths:
                    paths[nxt] = paths[s] + [a]
                    queue.append(nxt)
        candidates = [(self.counts[s, a], len(path), s, a, path)
                      for s, path in paths.items() for a in range(4)]
        best = min((x[0], x[1]) for x in candidates)
        tied = sorted((x for x in candidates if x[:2] == best), key=lambda x: (x[2], x[3]))
        _, _, _, action, path = self.rng.choice(tied)
        return path[0] if path else action


def collect(seed, kind, budget, config):
    started = time.process_time()
    if kind == 'balanced_teacher':
        if budget % 36:
            raise ValueError('teacher budget must be a multiple of 36')
        events = training_experiences(seed, budget // 36, config['effect_noise_sd'], tuple(config['train_body']))
        return events, time.process_time() - started
    world = World(seed, config['effect_noise_sd'])
    collector = Collector(kind, seed, config)
    state = world.start
    events = []
    for _ in range(budget):
        action = collector.choose(state)
        event = world.step(state, action, tuple(config['train_body']))
        collector.observe(event.state, event.action, event.next_state)
        events.append(event)
        state = event.next_state
    return events, time.process_time() - started


def fit(events, variant, seed, config, fixed_target=None):
    agent = Learner(variant, seed, config['action_cost'], fixed_target if variant == 'fixed_target' else None)
    for e in events:
        agent.observe(e)
    return agent


def knowledge_metrics(events, seed, config):
    # Evaluator only: World/effect truth never reaches a collector.
    truth = World(seed, config['effect_noise_sd'])
    agent = fit(events, 'learned', seed, config)
    counts = Counter(e.outcome for e in events if e.outcome is not None)
    discovered = len(counts)
    rmse = None
    if discovered == 2:
        rmse = math.sqrt(sum((agent.effects[g][i] - truth.effects[g][i]) ** 2
                             for g in truth.outcomes for i in range(2)) / 4)
    row = dict(interactions=len(events), state_action_coverage=len(agent.transitions)/36,
               visited_states=len({truth.start} | {e.state for e in events} | {e.next_state for e in events}),
               discovered_outcomes=discovered, both_discovered=int(discovered == 2), effect_rmse=rmse,
               self_loop_fraction=sum(e.state == e.next_state for e in events)/len(events),
               repeat_pair_fraction=1-len(agent.transitions)/len(events),
               history_sha256=history_hash(events))
    for label, g in zip(('a', 'b'), truth.outcomes):
        row['samples_'+label] = counts[g]
        row['first_discovery_'+label] = next((i+1 for i, e in enumerate(events) if e.outcome == g), None)
        path = agent.path(truth.start, g)
        row['known_route_'+label] = int(path is not None)
        row['known_route_length_'+label] = len(path) if path else None
    return row


def evaluate(events, seed, variant, scenario, config):
    world = World(seed, config['effect_noise_sd'])
    world.effect_rng = random.Random(seed + config['test_effect_seed_offset'] + config['scenarios'].index(scenario))
    agent = fit(events, variant, seed, config, world.outcomes[0])
    initial = deepcopy(agent.snapshot())
    target = world.outcomes[1] if scenario == 'need_shift' else world.outcomes[0]
    if scenario == 'satiated':
        target = None
    known_target = target in agent.catalog if target else None
    known_route = bool(agent.path(world.start, target)) if target else None
    if scenario == 'blocked':
        world.block_short_a()
    body = tuple(config['shift_body'] if scenario == 'need_shift' else
                 config['satiated_body'] if scenario == 'satiated' else config['train_body'])
    state = world.start
    trace = []
    steps = 0
    outcome = None
    post_stop = None
    for tick in range(config['max_test_actions']):
        action, goal, info = agent.decide(state, body)
        record = dict(tick=tick, state=state, body=body, action=action, goal=goal, decision=info)
        trace.append(record)
        if action is None:
            break
        e = world.step(state, action, body)
        record['experience'] = asdict(e)
        agent.observe(e)
        steps += 1
        state, body = e.next_state, e.after
        if e.outcome is not None:
            outcome = e.outcome
            a2, g2, i2 = agent.decide(state, body)
            post_stop = int(a2 is None)
            record['post_completion_decision'] = dict(action=a2, goal=g2, decision=i2)
            break
    first = trace[0]
    success = int(first['action'] is None) if target is None else int(outcome == target)
    first_correct = int(first['action'] is None) if target is None else int(first['goal'] == target)
    if success:
        reason = 'success'
    elif target is None:
        reason = 'acted_when_satiated'
    elif not known_target:
        reason = 'target_not_discovered'
    elif not known_route:
        reason = 'no_initial_observed_route'
    elif first['goal'] != target:
        reason = 'wrong_first_goal_or_stop'
    elif trace[-1]['action'] is None:
        reason = 'stopped_after_feedback'
    elif outcome is not None:
        reason = 'wrong_outcome'
    else:
        reason = 'budget_exhausted'
    row = dict(success=success, first_choice_correct=first_correct, steps=steps,
               first_goal=first['goal'], first_action=first['action'], outcome=outcome,
               expected_outcome=target, initial_target_discovered=known_target,
               initial_known_route=known_route, post_completion_stop=post_stop,
               failure_reason=reason)
    return row, trace, initial
