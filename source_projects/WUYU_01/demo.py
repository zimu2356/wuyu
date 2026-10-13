"""Print a compact trace from a saved model; no network or background service."""
import argparse
import json
from pathlib import Path
import numpy as np
from wuyu.core import make_scenario, rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("memory", "fixed", "adaptive", "constant_gate"), default="adaptive")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=555000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    model = json.loads((Path(__file__).parent/"results/models.json").read_text())
    params = {"memory": model["baselines"]["memory"]["selected"]["parameter"],
              "fixed": model["baselines"]["fixed"]["selected"]["parameter"],
              "adaptive": np.array(model["selected"]["theta"]),
              "constant_gate": model["mean_training_gate"]}
    scenario = make_scenario(("mixed",), 1, args.steps, args.seed)
    result = rollout(scenario, args.mode, params[args.mode], capture=True)
    print("step action prediction true_quality learning_rate cumulative_reward")
    trace = result["trajectory"]
    for i in range(0, len(trace), max(1, len(trace)//16)):
        row = trace[i]
        print(f"{i+1:4d} {int(row[4]):6d} {row[1]:10.3f} {row[2]:12.3f} {row[3]:13.3f} {trace[:i+1,0].sum():17.3f}")
    print("True quality is diagnostic output only; the agent does not observe it before acting.")
    print("Mean reward:", float(result["metrics"]["reward"].mean()))


if __name__ == "__main__":
    main()
