#!/usr/bin/env python3
"""Run the complete, pre-specified toy study; Python standard library only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from wuyu_goal_study.core import Learner, World, training_experiences

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(values):
    n = len(values)
    p = sum(values) / n
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def paired_interval(a, b, repeats, rng_seed):
    diffs = [x-y for x, y in zip(a, b)]
    rng = random.Random(rng_seed)
    n = len(diffs)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return {"mean_difference": statistics.mean(diffs),
            "ci95": [samples[int(.025*(repeats-1))], samples[int(.975*(repeats-1))]],
            "paired_seeds": n}


def fit(variant, seed, cost, events, fixed_target):
    agent = Learner(variant, seed, cost, fixed_target if variant == "fixed_target" else None)
    for event in events:
        agent.observe(event)
    return agent


def episode(agent, world, body, max_steps):
    state = world.start
    trace = []
    outcome = None
    post_rest = None
    goals = []
    moves = 0
    for tick in range(max_steps):
        action, goal, info = agent.decide(state, body)
        entry = {"tick": tick, "state": state, "body": body,
                 "action": action, "goal": goal, "decision": info}
        trace.append(entry)
        if action is None:
            break
        goals.append(goal)
        event = world.step(state, action, body)
        entry["experience"] = asdict(event)
        agent.observe(event)
        state, body = event.next_state, event.after
        moves += 1
        if event.outcome is not None:
            outcome = event.outcome
            # This is an actual extra decision, not the runner's episode stop.
            next_action, next_goal, next_info = agent.decide(state, body)
            post_rest = next_action is None
            entry["post_completion_decision"] = {"action": next_action, "goal": next_goal,
                                                  "decision": next_info}
            break
    return {"outcome": outcome, "steps": moves, "first_goal": goals[0] if goals else None,
            "consistent_goal": bool(goals) and len(set(goals)) == 1,
            "post_completion_rest": post_rest, "trace": trace}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    source_files = [args.protocol, ROOT / "SPECIFICATION.md", ROOT / "run_study.py",
                    ROOT / "wuyu_goal_study/core.py"]
    source_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
                     for p in source_files}
    # Written before generation of any experimental outcome.
    (out / "run_manifest.json").write_text(json.dumps({
        "started_utc": started, "python": sys.version, "protocol": protocol,
        "source_sha256_before_run": source_hashes,
        "status": "running", "external_registration": False,
    }, indent=2, ensure_ascii=False))

    rows, histories, snapshots = [], [], []
    costs = [protocol["action_cost"]] + protocol["sensitivity_costs"]
    seeds = range(protocol["seed_start"], protocol["seed_start"]+protocol["n_seeds"])
    with gzip.open(out / "training.jsonl.gz", "wt", encoding="utf-8") as trainfile, \
            gzip.open(out / "traces.jsonl.gz", "wt", encoding="utf-8") as tracefile:
        for seed in seeds:
            world = World(seed, protocol["effect_noise_sd"])
            data = {}
            for swapped in [False, True]:
                events = training_experiences(seed, protocol["training_repeats_per_state_action"],
                                               protocol["effect_noise_sd"], tuple(protocol["train_body"]), swapped)
                data[swapped] = events
                for e in events:
                    trainfile.write(json.dumps({"seed": seed, "swapped_history": swapped,
                                                "experience": asdict(e)}, sort_keys=True)+"\n")
            for cost in costs:
                for variant in protocol["variants"]:
                    for scenario in protocol["scenarios"]:
                        agent = fit(variant, seed, cost, data[False], world.outcomes[0])
                        test_world = World(seed, protocol["effect_noise_sd"])
                        body = tuple(protocol["train_body"])
                        target = world.outcomes[0]
                        if scenario == "blocked":
                            test_world.block_short_a()
                        elif scenario == "need_shift":
                            body = tuple(protocol["shift_body"])
                            target = world.outcomes[1]
                        elif scenario == "satiated":
                            body = tuple(protocol["satiated_body"])
                            target = None
                        result = episode(agent, test_world, body, protocol["max_steps"])
                        success = (result["steps"] == 0) if target is None else (result["outcome"] == target)
                        key = {"seed": seed, "cost": cost, "variant": variant, "scenario": scenario}
                        rows.append({**key, "success": int(success), "steps": result["steps"],
                                     "first_goal": result["first_goal"], "outcome": result["outcome"],
                                     "expected_outcome": target,
                                     "consistent_goal": int(result["consistent_goal"]),
                                     "post_completion_rest": result["post_completion_rest"]})
                        tracefile.write(json.dumps({**key, **result}, sort_keys=True)+"\n")
                    if cost == protocol["action_cost"]:
                        # First choice only, with EXACTLY the same current observation.
                        agents = [fit(variant, seed, cost, data[s], world.outcomes[0]) for s in [False, True]]
                        selected = [a.decide(world.start, tuple(protocol["train_body"]))[1] for a in agents]
                        histories.append({"seed": seed, "variant": variant,
                                          "normal_goal": selected[0], "swapped_goal": selected[1],
                                          "changed": int(selected[0] != selected[1]),
                                          "both_follow_history": int(selected == world.outcomes)})
                        snapshots.append({"seed": seed, "normal_history": agents[0].snapshot(),
                                          "swapped_history": agents[1].snapshot()})

    for filename, entries in [("episodes.csv", rows), ("history_pairs.csv", histories)]:
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    with gzip.open(out / "learned_states.jsonl.gz", "wt", encoding="utf-8") as f:
        for snapshot in snapshots:
            f.write(json.dumps(snapshot, sort_keys=True)+"\n")

    summary = {"episode_count": len(rows), "history_pair_count": len(histories),
               "groups": [], "contrasts": [], "history": []}
    for cost in costs:
        for scenario in protocol["scenarios"]:
            vectors = {}
            for variant in protocol["variants"]:
                group = [r for r in rows if r["cost"] == cost and r["scenario"] == scenario and r["variant"] == variant]
                vals = [r["success"] for r in group]
                vectors[variant] = vals
                post = [int(r["post_completion_rest"]) for r in group if r["post_completion_rest"] is not None]
                summary["groups"].append({"cost": cost, "scenario": scenario, "variant": variant,
                    "n": len(vals), "success_count": sum(vals), "success_rate": statistics.mean(vals),
                    "wilson_ci95": wilson(vals), "mean_steps": statistics.mean(r["steps"] for r in group),
                    "consistent_goal_rate": statistics.mean(r["consistent_goal"] for r in group),
                    "post_completion_rest_n": len(post),
                    "post_completion_rest_rate": statistics.mean(post) if post else None})
            if cost == protocol["action_cost"]:
                for variant in ["no_goal_buffer", "cached_value", "erased_effects", "fixed_target"]:
                    summary["contrasts"].append({"scenario": scenario, "comparison": "learned - " + variant,
                        **paired_interval(vectors["learned"], vectors[variant], protocol["bootstrap_resamples"], protocol["bootstrap_seed"])})
    for variant in protocol["variants"]:
        group = [r for r in histories if r["variant"] == variant]
        summary["history"].append({"variant": variant, "n": len(group),
                                   "changed_rate": statistics.mean(r["changed"] for r in group),
                                   "both_follow_history_rate": statistics.mean(r["both_follow_history"] for r in group)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    changed = [str(p) for p in source_files if digest(p) != source_hashes[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]]
    if changed:
        raise RuntimeError("Source changed during run: " + str(changed))
    manifest = json.loads((out / "run_manifest.json").read_text())
    manifest.update({"status": "completed", "ended_utc": datetime.now(timezone.utc).isoformat(),
                     "episodes": len(rows), "history_pairs": len(histories),
                     "output_sha256": {p.name: digest(p) for p in sorted(out.iterdir())
                                       if p.is_file() and p.name != "run_manifest.json"}})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    for g in summary["groups"]:
        if g["cost"] == protocol["action_cost"]:
            print(f"{g['scenario']:12} {g['variant']:15} {g['success_count']:3}/{g['n']} mean_steps={g['mean_steps']:.2f}")
    print(json.dumps({"episodes": len(rows), "history_pairs": len(histories)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
