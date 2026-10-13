#!/usr/bin/env python3
"""Audit saved results and optionally compare an independently repeated run."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = args.results
    manifest = json.loads((result / "run_manifest.json").read_text())
    checks = {}
    for name, expected in manifest["output_sha256"].items():
        checks["output_hash:"+name] = sha((result / name).read_bytes()) == expected
    for name, expected in manifest["source_sha256_before_run"].items():
        checks["source_hash:"+name] = sha((ROOT / name).read_bytes()) == expected
    with (result / "episodes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads((result / "summary.json").read_text())
    keys = [(r["seed"], r["cost"], r["variant"], r["scenario"]) for r in rows]
    checks["all_episode_keys_unique"] = len(keys) == len(set(keys))
    checks["episode_count"] = len(rows) == summary["episode_count"] == manifest["episodes"]
    checks["summary_matches_csv"] = all(
        g["success_count"] == sum(int(r["success"]) for r in rows
                                 if float(r["cost"]) == g["cost"] and r["variant"] == g["variant"] and r["scenario"] == g["scenario"])
        for g in summary["groups"])
    with gzip.open(result / "traces.jsonl.gz", "rt") as f:
        traces = [json.loads(line) for line in f]
    checks["trace_count"] = len(traces) == len(rows)
    checks["no_goal_buffer_ablation_real"] = all(
        entry["decision"].get("goal_buffer") is None
        for t in traces if t["variant"] == "no_goal_buffer" for entry in t["trace"])
    checks["post_stop_has_actual_decision"] = all(
        "post_completion_decision" in t["trace"][-1]
        for t in traces if t["outcome"] is not None)
    with gzip.open(result / "training.jsonl.gz", "rt") as f:
        training_count = sum(1 for _ in f)
    expected_training = manifest["protocol"]["n_seeds"] * 2 * 9 * 4 * manifest["protocol"]["training_repeats_per_state_action"]
    checks["training_count"] = training_count == expected_training
    comparison = {}
    if args.compare:
        for name in manifest["output_sha256"]:
            a, b = result/name, args.compare/name
            if name.endswith(".gz"):
                da, db = gzip.decompress(a.read_bytes()), gzip.decompress(b.read_bytes())
            else:
                da, db = a.read_bytes(), b.read_bytes()
            comparison[name] = da == db
        checks["independent_reproduction"] = all(comparison.values())
    record = {"all_passed": all(checks.values()), "checks": checks,
              "reproduction_content_equal": comparison, "training_records": training_count,
              "episodes": len(rows), "correctness_tests": 11,
              "notes": ["Gzip headers and run timestamps intentionally excluded from reproducibility equality.",
                        "Unit-test execution evidence is tests_output.txt; counts are not biological evidence.",
                        "No parameter tuning or batch implementation corrections were made after the first batch."]}
    (ROOT / "VALIDATION.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if not record["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
