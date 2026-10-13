"""Hidden-state prediction with exactly computable Bayesian references."""
from dataclasses import dataclass
import numpy as np

KINDS = ("iid", "stable", "switching", "no_signal")
HAZARDS = (0., .001, .005, .01, .02, .05, .10, .15, .25, .35, .50)


@dataclass
class Sequence:
    kind: str
    truth: np.ndarray
    cue: np.ndarray
    regime: np.ndarray
    seeds: np.ndarray
    reliability: float


def transition(kind):
    """Four states indexed by 2 * regime + target."""
    m = np.zeros((4, 4))
    for r in range(2):
        for z in range(2):
            for rr in range(2):
                for zz in range(2):
                    rp = .985 if r == rr else .015
                    h = (.005, .25)[rr]
                    if kind == "iid":
                        h = .5
                    if kind == "stable":
                        h = 0.
                    m[2*r+z, 2*rr+zz] = rp * ((1-h) if z == zz else h)
    return m


def generate(kind, count, horizon, seed_base):
    truth = np.zeros((horizon, count), dtype=np.int8)
    cue = truth.copy()
    regimes = truth.copy()
    reliability = .5 if kind == "no_signal" else .7
    for j in range(count):
        rng = np.random.default_rng(seed_base+j)
        z, r = rng.integers(0, 2, 2)
        for t in range(horizon):
            if t:
                r ^= int(rng.random() < .015)
                h = .5 if kind == "iid" else 0. if kind == "stable" else (.005, .25)[r]
                z ^= int(rng.random() < h)
            truth[t, j], regimes[t, j] = z, r
            cue[t, j] = z if rng.random() < reliability else 1-z
    return Sequence(kind, truth, cue, regimes, np.arange(seed_base, seed_base+count), reliability)


def fixed_filter(cues, reliability, hazard):
    # Keep both masses: reconstructing the small mass as 1-p causes an
    # artificial absorbing belief after p rounds to exactly one.
    belief = np.full((cues.shape[1], 2), .5)
    out = np.empty(cues.shape)
    for t, y in enumerate(cues):
        prior = (1-hazard)*belief+hazard*belief[:, ::-1]
        likelihood = np.where(y[:, None] == np.array([0, 1]), reliability, 1-reliability)
        belief = prior*likelihood
        belief /= belief.sum(axis=1, keepdims=True)
        out[t] = belief[:, 1]
    return out


def reference_filter(cues, reliability, matrix):
    belief = np.full((cues.shape[1], 4), .25)
    out = np.empty(cues.shape)
    for t, y in enumerate(cues):
        if t:
            belief = belief @ matrix
        likelihood = np.where(y[:, None] == np.array([0, 1, 0, 1]), reliability, 1-reliability)
        belief *= likelihood
        belief /= belief.sum(axis=1, keepdims=True)
        out[t] = belief[:, [1, 3]].sum(axis=1)
    return out


def predictions(seq, hazard):
    return {
        "frozen": np.full(seq.cue.shape, .5),
        "current_only": np.where(seq.cue == 1, seq.reliability, 1-seq.reliability),
        "cumulative": fixed_filter(seq.cue, seq.reliability, 0.),
        "fixed_hazard": fixed_filter(seq.cue, seq.reliability, hazard),
        "bayes_reference": reference_filter(seq.cue, seq.reliability, transition(seq.kind)),
    }


def score(probability, truth, burn=64):
    p = np.clip(probability, 1e-9, 1-1e-9)
    loss = -(truth*np.log(p)+(1-truth)*np.log1p(-p))
    acc = np.where(probability == .5, .5, (probability > .5) == truth)
    brier = (probability-truth)**2
    return {"log_loss": loss[burn:].mean(axis=0), "accuracy": acc[burn:].mean(axis=0),
            "brier": brier[burn:].mean(axis=0), "loss_curve": loss.mean(axis=1)}
