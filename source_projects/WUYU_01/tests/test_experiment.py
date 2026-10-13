import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu.core import Agent, World, make_scenario, rollout, save_checkpoint, load_checkpoint


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.scenario = make_scenario(("fast", "noisy"), 2, 100, 444000)
        self.theta = np.array([-1., 2., -1., 1., .5, -2., .4, 1., -.2])

    def test_action_changes_future_and_hidden_state_not_observed(self):
        a, b = World(self.scenario), World(self.scenario)
        self.assertEqual(set(a.observe()), {"stock", "position"})
        a.step(np.zeros(4, dtype=int))
        b.step(np.ones(4, dtype=int))
        self.assertFalse(np.array_equal(a.stock, b.stock))
        self.assertFalse(np.array_equal(a.position, b.position))

    def test_reproducible_and_state_equivalent(self):
        a = rollout(self.scenario, "adaptive", self.theta, capture=True)
        b = rollout(self.scenario, "adaptive", self.theta, capture=True)
        c = rollout(self.scenario, "state_equivalent", self.theta, capture=True)
        np.testing.assert_array_equal(a["trajectory"], b["trajectory"])
        np.testing.assert_array_equal(a["trajectory"], c["trajectory"])
        np.testing.assert_array_equal(a["agent"].q, c["agent"].q)

    def test_complete_checkpoint_resumes_exactly(self):
        full = rollout(self.scenario, "adaptive", self.theta, capture=True)
        prefix = rollout(self.scenario, "adaptive", self.theta, stop=43)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.npz"
            save_checkpoint(path, prefix["world"], prefix["agent"])
            w, a = load_checkpoint(path)
            tail = rollout(w.scenario, "adaptive", self.theta, initial=(w, a), start=43, capture=True)
        np.testing.assert_array_equal(full["trajectory"][43:], tail["trajectory"])
        np.testing.assert_array_equal(full["agent"].q, tail["agent"].q)

    def test_intervention_changes_only_specified_mediators(self):
        x = rollout(self.scenario, "adaptive", self.theta)["agent"]
        y = x.copy()
        y.erase_history_statistics()
        for key in ("q", "count", "age", "last", "parameter"):
            np.testing.assert_array_equal(getattr(x, key), getattr(y, key))
        for key in ("fast", "slow", "change"):
            self.assertTrue(np.all(getattr(y, key) == 0))
        action = np.zeros(4, dtype=int)
        feedback = np.full(4, .8)
        self.assertGreater(np.max(np.abs(x.learn(action, feedback)-y.learn(action, feedback))), 0)

    def test_single_candidate_agrees_with_batched_candidates(self):
        a = rollout(self.scenario, "adaptive", self.theta)
        b = rollout(self.scenario, "adaptive", np.stack([self.theta, self.theta+.1]))
        np.testing.assert_array_equal(a["metrics"]["reward"][0], b["metrics"]["reward"][0])

    def test_bounds_and_finite_outputs(self):
        for mode, parameter in (("memory", 4), ("fixed", .2), ("adaptive", self.theta)):
            result = rollout(self.scenario, mode, parameter)
            self.assertTrue(np.isfinite(result["agent"].q).all())
            self.assertTrue(((result["agent"].q >= 0) & (result["agent"].q <= 1)).all())
            self.assertTrue((result["metrics"]["oracle_gap"] >= -1e-12).all())


if __name__ == "__main__":
    unittest.main()
