"""Post-main exact MDP analysis. See SUPPLEMENT_01.md before interpreting."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from wuyu02.workshop import World
from run_experiment import interval, write_json

ROOT = Path(__file__).resolve().parent
STATES = [(a,b) for a in range(5) for b in range(5)]


def model(quality, support):
    reward = np.empty((25,5))
    nxt = np.empty((25,5), int)
    for i, charge in enumerate(STATES):
        for action in range(5):
            q = list(charge)
            r = -.2
            if action == 0:
                r = .2
            elif action in (1,2):
                k = action-1
                if q[k] or support[k]:
                    r = 2*quality[k]-1
                q[k] = max(0,q[k]-1)
            else:
                q[action-3] = 4
            reward[i,action], nxt[i,action] = r, 5*q[0]+q[1]
    return reward, nxt


def verify_model(quality, support, rewards, nxt):
    # Each action has a two-outcome observation; exact expectation, not sampling.
    largest = 0.
    for i, charge in enumerate(STATES):
        for action in range(5):
            r = []
            for u in (0., .999999):
                world = World(1, quality, [support])
                world.charge[0] = charge
                reward, _ = world.step(np.array([action]), np.array([1]), np.full((1,3),u))
                r.append(reward[0])
                assert 5*world.charge[0,0]+world.charge[0,1] == nxt[i,action]
            p = .6 if action == 0 else quality[action-1] if action in (1,2) else .5
            largest = max(largest, abs(p*r[0]+(1-p)*r[1]-rewards[i,action]))
    assert largest < 1e-12
    return largest


def solve(rewards, nxt, gamma=.95):
    value = np.zeros(25)
    for iteration in range(2000):
        q = rewards+gamma*value[nxt]
        v = q.max(axis=1)
        if np.max(np.abs(v-value)) < 1e-12:
            value = v
            break
        value = v
    q = rewards+gamma*value[nxt]
    residual = np.max(np.abs(value-q.max(axis=1)))
    assert residual < 1e-10
    policy = q.argmax(axis=1)
    state, trajectory, actions = 0, [], []
    for _ in range(4096):
        action = policy[state]
        trajectory.append(rewards[state,action])
        actions.append(action)
        state = nxt[state,action]
    return {"iterations": iteration+1,"bellman_residual": residual, "q": q,
            "policy_by_state": policy, "expected_tail_reward": np.mean(trajectory[-2048:]),
            "maintenance_tail_rate": np.mean(np.array(actions[-2048:]) >= 3),
            "state_order": STATES}


def main():
    out = ROOT/"results"
    if (out/"diagnostic.json").exists():
        raise SystemExit("Diagnostic exists; refusing to overwrite.")
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),"post_main": True,
                "hashes": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                           ("SUPPLEMENT_01.md","diagnose.py","wuyu02/workshop.py")}}
    write_json(out/"diagnostic_manifest.json",manifest)
    definitions = {"development_A":((.95,.55),(False,False)),
                   "development_B":((.55,.95),(False,False)),
                   "common":((.9,.9),(False,False)),
                   "broken_A":((.5,.9),(False,False)),
                   "broken_B":((.9,.5),(False,False)),
                   "supported_A":((.9,.9),(True,False)),
                   "supported_B":((.9,.9),(False,True))}
    solved = {}
    for name,(quality,support) in definitions.items():
        rewards,nxt = model(quality,support)
        solved[name] = solve(rewards,nxt)
        solved[name]["world_implementation_max_error"] = verify_model(quality,support,rewards,nxt)
    extra = {}
    with np.load(out/"development_raw.npz") as data:
        q = data["development__q__reward"][-512:].mean(axis=0).reshape(2,128)
        myopic = data["development__myopic__reward"][-512:].mean(axis=0).reshape(2,128)
        extra["q_minus_myopic_pair_average"] = interval((q-myopic).mean(axis=0),9302)
        action,cue = data["common__action"][-256:],data["common__cue"][-256:]
        valid_a = ((action == 1)&(cue >= 0)).sum(axis=0)
        valid_b = ((action == 2)&(cue >= 0)).sum(axis=0)
        share = np.divide(valid_a,valid_a+valid_b,out=np.full(len(valid_a),.5),where=valid_a+valid_b>0).reshape(2,128)
        extra["history_valid_use_share_difference"] = interval(share[0]-share[1],9302)
    write_json(out/"diagnostic.json",{"exact_references":solved,"post_main_descriptions":extra,
                                     "is_learning_result":False})
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out/"diagnostic_manifest.json",manifest)
    print({k:{"return":v["expected_tail_reward"],"maintenance":v["maintenance_tail_rate"],
              "residual":v["bellman_residual"]} for k,v in solved.items()})
    print(extra)


if __name__ == "__main__":
    main()
