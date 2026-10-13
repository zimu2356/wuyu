import itertools
import tempfile
from pathlib import Path
import unittest
import numpy as np
from wuyu02 import calibration as c
from wuyu02 import workshop as w


class MechanismTests(unittest.TestCase):
    def test_strong_then_opposite_evidence_recovers(self):
        cue = np.r_[np.ones(80,dtype=int),np.zeros(80,dtype=int)][:,None]
        fixed = c.fixed_filter(cue,.7,0.)
        reference = c.reference_filter(cue,.7,c.transition("stable"))
        self.assertAlmostEqual(fixed[-1,0],.5,places=12)
        np.testing.assert_allclose(fixed,reference,atol=1e-13)

    def test_bayes_matches_enumeration(self):
        cue = np.array([1, 0, 1, 1, 0])[:, None]
        matrix = c.transition("switching")
        computed = c.reference_filter(cue, .7, matrix)[:, 0]
        for t in range(1, 6):
            joint = np.zeros(2)
            for path in itertools.product(range(4), repeat=t):
                p = .25
                for i, state in enumerate(path):
                    if i:
                        p *= matrix[path[i-1], state]
                    p *= .7 if state % 2 == cue[i, 0] else .3
                joint[path[-1] % 2] += p
            self.assertAlmostEqual(computed[t-1], joint[1]/joint.sum(), places=12)

    def test_analytic_no_history_and_no_signal_controls(self):
        cue = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])
        ref = c.reference_filter(cue, .7, c.transition("iid"))
        np.testing.assert_allclose(ref, np.where(cue == 1, .7, .3), atol=1e-14)
        ref = c.reference_filter(cue, .5, c.transition("no_signal"))
        np.testing.assert_allclose(ref, .5, atol=1e-14)

    def test_tool_is_a_real_observation_channel(self):
        world = w.World(1, (.95, .55))
        np.testing.assert_array_equal(world.observe(), [[0, 0]])
        reward, cue = world.step(np.array([1]), np.array([1]), np.array([[.1, .1, .1]]))
        self.assertEqual(cue[0], -1)
        self.assertEqual(reward[0], -.2)
        world.step(np.array([3]), np.array([1]), np.array([[.1, .1, .1]]))
        self.assertEqual(world.charge[0, 0], 4)
        reward, cue = world.step(np.array([1]), np.array([1]), np.array([[.1, .1, .1]]))
        self.assertEqual(cue[0], 1)
        self.assertEqual(reward[0], 1.)
        self.assertEqual(world.charge[0, 0], 3)
        self.assertEqual(world.observe().shape, (1, 2))

    def test_delayed_credit_assignment(self):
        a = w.Learner(1, alpha=.5, gamma=.95)
        a.q[0, 4, 0, 1] = 2.
        a.update(np.array([[0, 0]]), np.array([3]), np.array([-.2]), np.array([[4, 0]]))
        self.assertAlmostEqual(a.q[0, 0, 0, 3], .5*(-.2+.95*2))
        b = w.Learner(1, alpha=.5, gamma=0.)
        b.q[0, 4, 0, 1] = 2.
        b.update(np.array([[0, 0]]), np.array([3]), np.array([-.2]), np.array([[4, 0]]))
        self.assertAlmostEqual(b.q[0, 0, 0, 3], -.1)

    def test_checkpoint_and_intervention_isolation(self):
        ev = w.events(3, 120, 930000)
        world, a = w.World(3), w.Learner(3)
        w.run(ev, world, a, stop=60)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"state.npz"
            w.checkpoint(path, world, a)
            ww, aa = w.restore(path)
            b1, b2 = w.run(ev, world, a, start=60), w.run(ev, ww, aa, start=60)
            for key in b1:
                np.testing.assert_array_equal(b1[key], b2[key])
            np.testing.assert_array_equal(a.q, aa.q)
        altered, erased = w.branch(world, a, "erase_values", np.zeros(3, int))
        np.testing.assert_array_equal(altered.charge, world.charge)
        np.testing.assert_array_equal(altered.quality, world.quality)
        np.testing.assert_array_equal(erased.q, 0.)
        changed, same = w.branch(world, a, "external_support", np.zeros(3, int))
        np.testing.assert_array_equal(same.q, a.q)
        np.testing.assert_array_equal(changed.quality, world.quality)
        self.assertTrue(changed.support[:, 0].all())
        self.assertFalse(world.support.any())

    def test_batch_matches_individual_and_no_input_mutation(self):
        ev = w.events(4, 100, 940000)
        world, a = w.World(4), w.Learner(4)
        saved = ev.target.copy()
        batch = w.run(ev, world, a)
        for i in range(4):
            one_world, one = w.World(1), w.Learner(1)
            out = w.run(ev.columns([i]), one_world, one)
            np.testing.assert_array_equal(out["reward"][:, 0], batch["reward"][:, i])
            np.testing.assert_array_equal(out["action"][:, 0], batch["action"][:, i])
            np.testing.assert_array_equal(one.q[0], a.q[i])
        np.testing.assert_array_equal(ev.target, saved)


if __name__ == "__main__":
    unittest.main()
