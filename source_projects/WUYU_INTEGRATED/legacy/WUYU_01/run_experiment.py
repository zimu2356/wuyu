"""Run the fixed protocol; writes raw data before any narrative report."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from wuyu.core import (FEATURES, FEATURE_NAMES, TRAIN_KINDS, TEST_KINDS,
                       make_scenario, rollout, save_checkpoint)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
CONFIG = {"train_per_kind": 8, "train_horizon": 384,
          "validation_per_kind": 8, "validation_horizon": 512,
          "test_per_kind": 128, "test_horizon": 1024,
          "search_seeds": [11, 23, 47], "generations": 60, "population": 48,
          "memory_grid": [1, 2, 4, 8, 16, 32],
          "fixed_grid": [.02, .05, .1, .2, .35, .5, .7, .9],
          "bootstrap_replicates": 2000, "bootstrap_seed": 9191}


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def constant_theta(alpha):
    result = np.zeros(FEATURES)
    p = (alpha-.01)/.94
    result[0] = np.log(p/(1-p))
    return result


def train(train_scenario, validation):
    baselines = {}
    for mode, grid in (("memory", CONFIG["memory_grid"]), ("fixed", CONFIG["fixed_grid"])):
        rows = []
        for parameter in grid:
            tr = rollout(train_scenario, mode, parameter)["metrics"]["reward"].mean()
            va = rollout(validation, mode, parameter)["metrics"]["reward"].mean()
            rows.append({"parameter": parameter, "train": float(tr), "validation": float(va)})
        baselines[mode] = {"grid": rows, "selected": max(rows, key=lambda r:r["validation"])}
        print(mode, baselines[mode]["selected"], flush=True)
    dump("baseline_selection.json", baselines)
    runs = []
    history = []
    constants = np.stack([constant_theta(a) for a in CONFIG["fixed_grid"]])
    initial = constant_theta(baselines["fixed"]["selected"]["parameter"])
    for seed in CONFIG["search_seeds"]:
        rng = np.random.default_rng(seed)
        mean, std = initial.copy(), np.full(FEATURES, 1.8)
        best, best_score = initial.copy(), -np.inf
        for generation in range(CONFIG["generations"]):
            population = rng.normal(mean, std, (CONFIG["population"], FEATURES))
            population = np.clip(population, -12, 12)
            population[0], population[1] = mean, best
            population[2:2+len(constants)] = constants
            scores = rollout(train_scenario, "adaptive", population)["metrics"]["reward"].mean(axis=1)
            order = np.argsort(scores)
            if scores[order[-1]] > best_score:
                best_score = float(scores[order[-1]])
                best = population[order[-1]].copy()
            elites = population[order[-10:]]
            mean = .35*mean + .65*elites.mean(axis=0)
            std = np.maximum(.18, .35*std + .65*elites.std(axis=0))
            history.append({"seed": seed, "generation": generation,
                            "best_train_reward": best_score, "population_mean": float(scores.mean())})
            if (generation+1) % 10 == 0:
                print(f"search {seed} generation {generation+1}/60 train {best_score:.6f}", flush=True)
        val = float(rollout(validation, "adaptive", best)["metrics"]["reward"].mean())
        runs.append({"seed": seed, "theta": best.tolist(), "train": best_score, "validation": val})
        dump("training_runs.json", runs)
        dump("training_history.json", history)
        print("finished search", seed, "validation", val, flush=True)
    selected = max(runs, key=lambda r:r["validation"])
    theta = np.array(selected["theta"])
    mean_gate = float(rollout(train_scenario, "adaptive", theta)["metrics"]["alpha"].mean())
    model = {"baselines": baselines, "runs": runs, "selected": selected,
             "mean_training_gate": mean_gate, "feature_names": FEATURE_NAMES,
             "config": CONFIG}
    dump("models.json", model)
    return model


def boot_interval(differences, groups, repeats=2000, seed=9191):
    # Equal weighting of predefined environments; bootstrap independent lives within each.
    rng = np.random.default_rng(seed)
    group_arrays = [differences[np.array(groups) == group] for group in sorted(set(groups))]
    samples = np.zeros(repeats)
    for arr in group_arrays:
        indices = rng.integers(0, len(arr), (repeats, len(arr)))
        samples += arr[indices].mean(axis=1)/len(group_arrays)
    return {"mean": float(np.mean([x.mean() for x in group_arrays])),
            "ci95": np.quantile(samples, [.025, .975]).tolist()}


def evaluate(model):
    test = make_scenario(TEST_KINDS, CONFIG["test_per_kind"], CONFIG["test_horizon"], 100000)
    theta = np.array(model["selected"]["theta"])
    methods = [("memory", model["baselines"]["memory"]["selected"]["parameter"]),
               ("fixed", model["baselines"]["fixed"]["selected"]["parameter"]),
               ("adaptive", theta), ("constant_gate", model["mean_training_gate"])]
    results = {}
    allrows = []
    for mode, parameter in methods:
        result = rollout(test, mode, parameter, capture=True)
        results[mode] = {k:v[0] for k,v in result["metrics"].items()}
        np.save(OUT/f"trace_{mode}.npy", result["trajectory"])
        for i, (kind, seed) in enumerate(zip(test.kinds, test.seeds)):
            allrows.append({"method": mode, "environment": kind, "seed": seed,
                            **{key: float(value[i]) for key,value in results[mode].items()}})
        print("test", mode, float(results[mode]["reward"].mean()), flush=True)
    with (OUT/"test_lifetimes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(allrows[0]))
        writer.writeheader(); writer.writerows(allrows)
    summaries = {}
    for group in ("overall",) + TEST_KINDS:
        mask = np.ones(test.size, dtype=bool) if group == "overall" else np.array(test.kinds) == group
        groups = np.array(test.kinds)[mask]
        summaries[group] = {"means": {mode:{key: float(value[mask].mean()) for key,value in metric.items()}
                                      for mode,metric in results.items()}}
        for baseline in ("fixed", "memory", "constant_gate"):
            delta = results["adaptive"]["reward"][mask]-results[baseline]["reward"][mask]
            summaries[group]["adaptive_minus_"+baseline] = boot_interval(delta, groups)
    seed_results = []
    for run in model["runs"]:
        r = rollout(test, "adaptive", np.array(run["theta"]))
        seed_results.append({"training_seed": run["seed"],
                             "test_reward": float(r["metrics"]["reward"].mean()),
                             "by_environment": {kind: float(r["metrics"]["reward"][0, np.array(test.kinds)==kind].mean())
                                                for kind in TEST_KINDS}})
    summaries["training_seed_sensitivity"] = seed_results
    dump("evaluation.json", summaries)
    return summaries


def interventions(model):
    theta = np.array(model["selected"]["theta"])
    scenario = make_scenario(("fast", "noisy"), 64, 768, 200000)
    prefix = rollout(scenario, "adaptive", theta, stop=384)
    save_checkpoint(OUT/"checkpoint_before_intervention.npz", prefix["world"], prefix["agent"])
    intact, erased = prefix["agent"].copy(), prefix["agent"].copy()
    erased.erase_history_statistics()
    qdiff, adiff = [], []
    rows = np.arange(scenario.size)
    # Common forced observations isolate updater differences from changed actions.
    for t in range(384, 448):
        action = np.full(scenario.size, (t-384)%6, dtype=int)
        feedback = scenario.quality[t, rows, action] + scenario.noise[t, rows, action]
        a = intact.learn(action, feedback)
        b = erased.learn(action, feedback)
        adiff.append(np.abs(a-b))
        qdiff.append(np.mean(np.abs(intact.q-erased.q), axis=1))
    first_branch = (copy.deepcopy(prefix["world"]), prefix["agent"].copy())
    second_branch = (copy.deepcopy(prefix["world"]), prefix["agent"].copy())
    second_branch[1].erase_history_statistics()
    a = rollout(scenario, "adaptive", theta, initial=first_branch, start=384)
    b = rollout(scenario, "adaptive", theta, initial=second_branch, start=384)
    eq_scenario = make_scenario(("mixed",), 16, 1024, 400000)
    aa = rollout(eq_scenario, "adaptive", theta, capture=True)
    bb = rollout(eq_scenario, "state_equivalent", theta, capture=True)
    result = {"forced_tail_mean_abs_alpha_difference": float(np.mean(adiff)),
              "forced_tail_mean_abs_prediction_difference": float(np.mean(qdiff)),
              "intact_minus_erased_free_tail_reward": boot_interval(
                  a["metrics"]["reward"][0]-b["metrics"]["reward"][0], scenario.kinds),
              "state_equivalent_max_trace_difference": float(np.max(np.abs(aa["trajectory"]-bb["trajectory"]))),
              "state_equivalent_max_final_state_difference": float(np.max(np.abs(aa["agent"].q-bb["agent"].q)))}
    np.savez_compressed(OUT/"intervention_raw.npz", alpha_difference=np.array(adiff),
                        prediction_difference=np.array(qdiff),
                        intact_reward=a["metrics"]["reward"][0], erased_reward=b["metrics"]["reward"][0])
    dump("interventions.json", result)
    print("interventions", json.dumps(result), flush=True)
    return result


def long_run(model):
    scenario = make_scenario(TEST_KINDS, 32, 4096, 300000)
    output = {}
    for mode, parameter in (("adaptive", np.array(model["selected"]["theta"])),
                             ("fixed", model["baselines"]["fixed"]["selected"]["parameter"])):
        r = rollout(scenario, mode, parameter)
        output[mode] = {"mean_reward": float(r["metrics"]["reward"].mean()),
                        "finite": bool(np.isfinite(r["agent"].q).all()),
                        "min_prediction": float(r["agent"].q.min()),
                        "max_prediction": float(r["agent"].q.max()),
                        "segment_rewards": r["segments"].mean(axis=1).tolist(),
                        "by_environment": {kind: float(r["metrics"]["reward"][0, np.array(scenario.kinds)==kind].mean())
                                           for kind in TEST_KINDS}}
    dump("long_run.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-models", action="store_true", help="Skip outer training; use saved model parameters.")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    started = time.time()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "numpy": np.__version__, "config": CONFIG,
                "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT/"PROTOCOL.md", ROOT/"run_experiment.py", ROOT/"wuyu/core.py"]}}
    dump("run_manifest.json", manifest)
    if args.reuse_models:
        model = json.loads((OUT/"models.json").read_text())
    else:
        tr = make_scenario(TRAIN_KINDS, 8, 384, 1000)
        va = make_scenario(TRAIN_KINDS, 8, 512, 10000)
        model = train(tr, va)
    evaluate(model)
    interventions(model)
    long_run(model)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = time.time()-started
    dump("run_manifest.json", manifest)
    print("COMPLETE seconds", manifest["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
