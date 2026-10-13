import unittest

from wuyu_goal_study.core import Learner, World, training_experiences
from run_study import episode


class MechanismChecks(unittest.TestCase):
    def make(self, variant="learned", seed=713, swapped=False):
        world = World(seed, noise=0)
        agent = Learner(variant, seed, .01, world.outcomes[0] if variant == "fixed_target" else None)
        for e in training_experiences(seed, 2, 0, (.75, .05), swapped):
            agent.observe(e)
        return agent, world

    def test_no_unobserved_goals_or_transitions(self):
        agent = Learner("learned", 1, .01)
        self.assertEqual(agent.catalog, set())
        self.assertIsNone(agent.decide("unseen", (.75, .05))[0])
        self.assertIsNone(agent.path("unseen", "unseen_outcome"))

    def test_one_actual_event_discovers_one_goal_only(self):
        world = World(2, 0)
        agent = Learner("learned", 2, .01)
        event = world.step(world.states[5], world.action_maps[5][0], (.75, .05))
        agent.observe(event)
        self.assertEqual(agent.catalog, {world.outcomes[0]})
        self.assertAlmostEqual(agent.effects[world.outcomes[0]][0], -.6)

    def test_block_is_unknown_until_feedback_then_replanned(self):
        agent, world = self.make()
        old = agent.decide(world.start, (.75, .05))[0]
        world.block_short_a()
        self.assertEqual(agent.decide(world.start, (.75, .05))[0], old)
        agent.observe(world.step(world.start, old, (.75, .05)))
        action, goal, _ = agent.decide(world.start, (.75, .05))
        self.assertNotEqual(action, old)
        self.assertEqual(goal, world.outcomes[0])
        self.assertEqual(len(agent.path(world.start, goal)), 4)

    def test_need_shift_changes_goal_without_new_learning(self):
        agent, world = self.make()
        self.assertEqual(agent.decide(world.start, (.75, .05))[1], world.outcomes[0])
        self.assertEqual(agent.decide(world.start, (.05, .75))[1], world.outcomes[1])

    def test_cached_value_is_not_silently_given_revaluation(self):
        agent, world = self.make("cached_value")
        self.assertEqual(agent.decide(world.start, (.05, .75))[1], world.outcomes[0])

    def test_no_goal_buffer_really_has_no_goal_buffer(self):
        agent, world = self.make("no_goal_buffer")
        agent.decide(world.start, (.75, .05))
        self.assertIsNone(agent.goal)

    def test_erased_effects_keeps_navigation_but_not_valuation(self):
        agent, world = self.make("erased_effects")
        self.assertEqual(agent.effects, {})
        self.assertEqual(agent.values, {})
        self.assertEqual(len(agent.catalog), 2)
        self.assertTrue(agent.path(world.start, world.outcomes[0]))

    def test_post_completion_rest_is_a_separate_decision(self):
        agent, world = self.make()
        result = episode(agent, world, (.75, .05), 12)
        self.assertEqual(result["outcome"], world.outcomes[0])
        self.assertTrue(result["post_completion_rest"])
        self.assertIn("post_completion_decision", result["trace"][-1])
        cached, world = self.make("cached_value")
        self.assertFalse(episode(cached, world, (.75, .05), 12)["post_completion_rest"])

    def test_same_current_input_different_effect_history(self):
        first, world = self.make()
        second, _ = self.make(swapped=True)
        body = (.75, .05)
        self.assertEqual(first.decide(world.start, body)[1], world.outcomes[0])
        self.assertEqual(second.decide(world.start, body)[1], world.outcomes[1])

    def test_opaque_identifiers_and_action_permutations(self):
        for seed in [817, 823, 829, 839]:
            agent, world = self.make(seed=seed)
            result = episode(agent, world, (.75, .05), 12)
            self.assertEqual(result["outcome"], world.outcomes[0])
            self.assertEqual(result["steps"], 3)

    def test_training_pairs_have_identical_sampling_and_noise(self):
        left = training_experiences(701, 2, .025, (.75, .05), False)
        right = training_experiences(701, 2, .025, (.75, .05), True)
        for a, b in zip(left, right):
            self.assertEqual((a.state, a.action, a.next_state, a.outcome),
                             (b.state, b.action, b.next_state, b.outcome))
            if a.outcome:
                # Swapping the mean effect leaves the noise sum unchanged.
                self.assertAlmostEqual(sum(a.after), sum(b.after))


if __name__ == "__main__":
    unittest.main()
