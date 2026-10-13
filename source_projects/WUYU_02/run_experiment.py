"""Run the fixed WUYU_02 protocol. Existing evidence is never overwritten."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
from datetime import datetime, timezone
import numpy as np
from wuyu02 import calibration as cal
from wuyu02 import workshop as ws

ROOT = Path(__file__).resolve().parent
CONFIG = {"calibration_horizon": 512, "calibration_train": 24, "calibration_validation": 24,
          "calibration_test": 256, "calibration_burn": 64, "bootstrap": 2000,
          "development_horizon": 2048, "common_horizon": 1024, "branch_horizon": 2048,
          "development_train_pairs": 24, "development_validation_pairs": 24,
          "development_test_pairs": 128, "alpha_grid": [.03, .08, .15, .30],
          "gamma": .95, "epsilon": .08}


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write_json(path, x):
    path.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=serial)+"\n")


def interval(values, seed=9202):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(CONFIG["bootstrap"], len(values)))
    ci = np.quantile(values[index].mean(axis=1), [.025, .975])
    return {"mean": values.mean(), "ci95": ci, "independent_units": len(values)}


def hashes():
    paths = [ROOT/"PROTOCOL.md", ROOT/"run_experiment.py", ROOT/"requirements.txt"]
    paths += sorted((ROOT/"wuyu02").glob("*.py"))
    paths += sorted((ROOT/"tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def calibration(out):
    selected, trials = {}, {}
    for ki, kind in enumerate(cal.KINDS):
        train = cal.generate(kind, 24, 512, 10000+ki*1000)
        val = cal.generate(kind, 24, 512, 20000+ki*1000)
        trials[kind] = []
        for h in cal.HAZARDS:
            trial = {"hazard": h}
            for name, seq in (("train", train), ("validation", val)):
                pred = cal.fixed_filter(seq.cue, seq.reliability, h)
                trial[name] = cal.score(pred, seq.truth)["log_loss"].mean()
            trials[kind].append(trial)
        selected[kind] = min(trials[kind], key=lambda x:x["validation"])["hazard"]
    write_json(out/"calibration_selection.json", {"selected_hazards": selected, "trials": trials})
    report, raw, curves = {}, {}, {}
    maximum = {}
    for ki, kind in enumerate(cal.KINDS):
        seq = cal.generate(kind, 256, 512, 100000+ki*1000)
        pred = cal.predictions(seq, selected[kind])
        report[kind] = {"selected_hazard": selected[kind], "means": {}}
        for name, p in pred.items():
            scores = cal.score(p, seq.truth)
            report[kind]["means"][name] = {k:v.mean() for k,v in scores.items() if k != "loss_curve"}
            raw[kind+"__"+name] = scores["log_loss"]
            curves[kind+"__"+name] = scores["loss_curve"]
        ref = raw[kind+"__bayes_reference"]
        report[kind]["loss_improvement"] = {
            name:interval(raw[kind+"__"+name]-ref) for name in pred if name != "bayes_reference"}
        maximum[kind] = {"reference_minus_current": np.max(np.abs(pred["bayes_reference"]-pred["current_only"])),
                         "reference_minus_half": np.max(np.abs(pred["bayes_reference"]-.5))}
    entropy = -.7*np.log(.7)-.3*np.log(.3)
    gates = {
        "A1_iid": bool(maximum["iid"]["reference_minus_current"] < 1e-12 and
                       abs(report["iid"]["means"]["bayes_reference"]["log_loss"]-entropy) < .02),
        "A2_history": bool(report["stable"]["loss_improvement"]["current_only"]["ci95"][0] > .10),
        "A3_adaptation": bool(report["switching"]["loss_improvement"]["frozen"]["ci95"][0] > .03 and
                              report["switching"]["loss_improvement"]["cumulative"]["ci95"][0] > .05),
        "A4_no_signal": bool(maximum["no_signal"]["reference_minus_half"] < 1e-12 and
                             abs(report["no_signal"]["means"]["bayes_reference"]["log_loss"]-np.log(2)) < 1e-12)
    }
    result = {"gates": gates, "all_passed": all(gates.values()), "worlds": report,
              "analytic_checks": maximum, "iid_theoretical_loss": entropy,
              "test_trajectories": 4*256, "methods_per_trajectory": 5}
    np.savez_compressed(out/"calibration_raw.npz", **raw)
    np.savez_compressed(out/"calibration_curves.npz", **curves)
    write_json(out/"calibration.json", result)
    print("A gates:", gates, flush=True)
    return result


def paired_events(n, horizon, seed):
    return ws.events(n, horizon, seed).columns(np.tile(np.arange(n), 2))


def history_world(n):
    return ws.World(2*n, np.repeat([[.95, .55], [.55, .95]], n, axis=0))


def train_selection(out):
    trials = []
    for alpha in CONFIG["alpha_grid"]:
        trial = {"alpha": alpha}
        for name, seed in (("train", 200000), ("validation", 220000)):
            ev = paired_events(24, 2048, seed)
            world, learner = history_world(24), ws.Learner(48, alpha)
            res = ws.run(ev, world, learner)
            trial[name] = res["reward"][-512:].mean()
            trial[name+"_by_history"] = res["reward"][-512:].mean(axis=0).reshape(2, 24).mean(axis=1)
        trials.append(trial)
    selected = max(trials, key=lambda x:x["validation"])["alpha"]
    write_json(out/"development_selection.json", {"selected_alpha": selected, "trials": trials})
    return selected


def development(out):
    alpha = train_selection(out)
    n = CONFIG["development_test_pairs"]
    preferred = np.repeat([0, 1], n)
    ev = paired_events(n, 2048, 300000)
    raw, summary, development_outputs = {}, {}, {}
    developed_world = developed_learner = None
    for name in ("q", "myopic", "random", "script_A", "script_B"):
        world = history_world(n)
        a = ws.Learner(2*n, alpha, gamma=0. if name == "myopic" else .95,
                       mode="q" if name == "myopic" else name)
        result = ws.run(ev, world, a)
        development_outputs[name] = result
        m = ws.metrics(result, preferred)
        summary[name] = {k:v.reshape(2,n).mean(axis=1) for k,v in m.items()}
        for key, value in result.items():
            raw["development__"+name+"__"+key] = value
        if name == "q":
            developed_world, developed_learner = world, a
    ws.checkpoint(out/"checkpoint_developed.npz", developed_world, developed_learner)
    common_world = ws.World(2*n)
    matched_agent = developed_learner.copy()
    ws.checkpoint(out/"checkpoint_matched.npz", common_world, matched_agent)
    common_ev = paired_events(n, 1024, 400000)
    common = ws.run(common_ev, copy.deepcopy(common_world), matched_agent.copy())
    natural_world = copy.deepcopy(common_world)
    natural_world.charge = developed_world.charge.copy()
    natural = ws.run(common_ev, natural_world, matched_agent.copy())
    for name, result in (("common",common),("natural_common",natural)):
        for k,v in result.items():
            raw[name+"__"+k] = v
    common_metrics = ws.metrics(common, preferred, tail=256)
    natural_metrics = ws.metrics(natural, preferred, tail=256)
    branch_ev = paired_events(n, 2048, 500000)
    branch_outputs, branch_metrics = {}, {}
    for name in ("intact", "frozen", "erase_values", "break_preferred", "break_frozen", "external_support", "replay_break"):
        world, a = ws.branch(common_world, matched_agent, name, preferred)
        forced = branch_outputs["intact"]["action"] if name == "replay_break" else None
        result = ws.run(branch_ev, world, a, forced=forced)
        branch_outputs[name] = result
        branch_metrics[name] = ws.metrics(result, preferred)
        for k,v in result.items():
            raw["branch__"+name+"__"+k] = v
        ws.checkpoint(out/("checkpoint_final_"+name+".npz"), world, a)
    delta = lambda x: interval(x.reshape(2,n).mean(axis=0), seed=9302)
    q_mean = ws.metrics(development_outputs["q"], preferred)["reward"].reshape(2,n)
    random_mean = ws.metrics(development_outputs["random"], preferred)["reward"].reshape(2,n)
    b1 = [interval(q_mean[k]-random_mean[k], 9302) for k in range(2)]
    share = common_metrics["a_share"].reshape(2,n)
    b2 = interval(share[0]-share[1], 9302)
    erased_first1024 = {k:v[:1024] for k,v in branch_outputs["erase_values"].items()}
    erased = ws.metrics(erased_first1024, preferred, tail=256)["a_share"].reshape(2,n)
    erased_diff = interval(erased[0]-erased[1], 9302)
    b3 = delta(branch_metrics["break_preferred"]["reward"]-branch_metrics["break_frozen"]["reward"])
    b4 = delta(branch_metrics["intact"]["preferred_maintenance"]-branch_metrics["external_support"]["preferred_maintenance"])
    gates = {"B1_learned_control": all(x["ci95"][0] > .1 for x in b1),
             "B2_history_mediation": bool(b2["ci95"][0] > .1 and abs(erased_diff["mean"]) < .1 and
                                          erased_diff["ci95"][0] <= 0 <= erased_diff["ci95"][1]),
             "B3_revaluation": bool(b3["ci95"][0] > .05),
             "B4_dependency_release": bool(b4["ci95"][0] > .03)}
    ns = natural_metrics["a_share"].reshape(2,n)
    result = {"selected_alpha": alpha, "independent_pairs": n, "development_means_by_history": summary,
              "gates": gates, "all_passed": all(gates.values()),
              "comparisons": {"development_q_minus_random_by_history": b1,
                              "history_share_difference": b2, "history_difference_after_value_erasure": erased_diff,
                              "natural_history_share_difference": interval(ns[0]-ns[1], 9302),
                              "broken_online_minus_frozen": b3, "maintenance_reduction_after_support": b4,
                              "broken_online_minus_replay": delta(branch_metrics["break_preferred"]["reward"]-branch_metrics["replay_break"]["reward"]),
                              "intact_online_minus_frozen": delta(branch_metrics["intact"]["reward"]-branch_metrics["frozen"]["reward"])},
              "branch_means": {name:{k:v.mean() for k,v in m.items()} for name,m in branch_metrics.items()},
              "branch_by_history": {name:{k:v.reshape(2,n).mean(axis=1) for k,v in m.items()} for name,m in branch_metrics.items()},
              "common_means_by_history": {k:v.reshape(2,n).mean(axis=1) for k,v in common_metrics.items()},
              "script_expected_development_return": {"favored": (4*.9-.2)/5, "unfavored": (4*.1-.2)/5},
              "note": "All B phenomena are generated by ordinary tabular Q learning with external rewards; not evidence of consciousness or novel goal generation."}
    np.savez_compressed(out/"development_raw.npz", **raw)
    write_json(out/"development.json", result)
    print("B gates:", gates, flush=True)
    print("B history, revaluation, release:", b2["mean"], b3["mean"], b4["mean"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/"run_manifest.json").exists():
        raise SystemExit("Refusing to overwrite evidence; choose a new --output directory.")
    begin = time.perf_counter()
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
                "python": platform.python_version(), "numpy": np.__version__, "source_hashes": hashes()}
    write_json(out/"run_manifest.json", manifest)
    a = calibration(out)
    b = development(out) if a["all_passed"] else None
    manifest.update({"completed_at": datetime.now(timezone.utc).isoformat(),
                     "elapsed_seconds": time.perf_counter()-begin,
                     "hashes_unchanged": manifest["source_hashes"] == hashes(),
                     "calibration_passed": a["all_passed"], "development_ran": b is not None})
    write_json(out/"run_manifest.json", manifest)
    print("Completed in", round(manifest["elapsed_seconds"], 2), "seconds.", flush=True)


if __name__ == "__main__":
    main()
