"""Inspect one developed individual without changing saved evidence."""
import argparse
from pathlib import Path
import numpy as np
from wuyu02 import workshop as w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", choices=["A","B"], default="A")
    p.add_argument("--branch", choices=["intact","frozen","erase_values","break_preferred","break_frozen","external_support"], default="intact")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=900000)
    args = p.parse_args()
    if not 1 <= args.steps <= 100000:
        p.error("--steps must be between 1 and 100000")
    root=Path(__file__).resolve().parent
    worlds,learners=w.restore(root/"results/checkpoint_matched.npz")
    idx=0 if args.history=="A" else len(worlds.charge)//2
    world=w.World(1, worlds.quality[[idx]], worlds.support[[idx]])
    world.charge=worlds.charge[[idx]].copy()
    learner=w.Learner(1,learners.alpha,learners.gamma,epsilon=learners.epsilon)
    learner.q=learners.q[[idx]].copy()
    preferred=np.array([0 if args.history=="A" else 1])
    world,learner=w.branch(world,learner,args.branch,preferred)
    output=w.run(w.events(1,args.steps,args.seed),world,learner)
    print("Ordinary tabular Q learner; this demonstration is not consciousness evidence.")
    print("History:",args.history,"branch:",args.branch,"seed:",args.seed)
    print("Average reward:",round(float(output["reward"].mean()),6))
    print("Action counts:",{name:int((output["action"]==i).sum()) for i,name in enumerate(w.ACTIONS)})
    print("Final tool charges:",world.charge[0].tolist())


if __name__=="__main__":
    main()
