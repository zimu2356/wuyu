"""Post-main diagnostic, explicitly separate from confirmatory outcomes."""
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from wuyu.core import TEST_KINDS, Agent, World, make_scenario, rollout
from run_experiment import OUT, ROOT, boot_interval, dump


def main():
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "status": "post-main diagnostic; new independent seeds; no retraining",
                "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT/"SUPPLEMENT_01.md", ROOT/"supplement.py")}}
    dump("supplement_manifest.json", manifest)
    model = json.loads((OUT/"models.json").read_text())
    scenario = make_scenario(TEST_KINDS, 64, 1024, 500000)
    results = {}
    for label, mode, parameter in (("adaptive", "adaptive", np.array(model["selected"]["theta"])),
                                  ("constant_gate", "constant_gate", model["mean_training_gate"]),
                                  ("fixed_002", "fixed", .02), ("fixed_001", "fixed", .01),
                                  ("frozen", "fixed", 0.)):
        results[label] = rollout(scenario, mode, parameter)["metrics"]["reward"][0]
    w = World(scenario)
    total = np.zeros(scenario.size)
    for t in range(scenario.horizon):
        # A diagnostic one-step greedy strategy, NOT a long-run optimal bound.
        action = np.argmax(scenario.quality[t]*w.stock-w.costs(), axis=1)
        _, reward, _, _ = w.step(action)
        total += reward
    results["greedy_quality"] = total/scenario.horizon
    output = {}
    for kind in ("overall",)+TEST_KINDS:
        mask = np.ones(scenario.size, dtype=bool) if kind == "overall" else np.array(scenario.kinds)==kind
        output[kind] = {"means": {k:float(v[mask].mean()) for k,v in results.items()}}
        for other in ("fixed_001", "constant_gate", "frozen", "greedy_quality"):
            output[kind]["adaptive_minus_"+other] = boot_interval(
                results["adaptive"][mask]-results[other][mask], np.array(scenario.kinds)[mask])
    np.savez_compressed(OUT/"supplement_raw.npz", **results, kinds=np.array(scenario.kinds), seeds=scenario.seeds)
    dump("supplement.json", output)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump("supplement_manifest.json", manifest)
    print(json.dumps(output["overall"], indent=2))


if __name__ == "__main__":
    main()
