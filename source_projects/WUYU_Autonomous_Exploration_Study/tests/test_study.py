import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from analyze import paired_bootstrap, wilson
from baseline.wuyu_goal_study.core import World, training_experiences
from run_study import ROOT, new_output, verify_freeze
from study import Collector, collect, evaluate, fit, history_hash, knowledge_metrics


CONFIG = json.loads((ROOT/'protocol.json').read_text())
TEST_SEED = 9101  # Separate from development and holdout cohorts.


class StudyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = training_experiences(TEST_SEED, 12, CONFIG['effect_noise_sd'], tuple(CONFIG['train_body']))

    def test_empty_collector_can_start_without_a_goal(self):
        for kind in ('random','least_visited'):
            collector = Collector(kind, TEST_SEED, CONFIG)
            self.assertIn(collector.choose('opaque_current_state'), range(4))
            self.assertEqual(collector.transitions, {})
            self.assertEqual(collector.counts, {})

    def test_directed_collector_uses_observed_route_to_untried_actions(self):
        collector = Collector('least_visited', TEST_SEED, CONFIG)
        for _ in range(2):
            for action in range(4):
                collector.observe('s', action, 'observed_x' if action==2 else 's')
        self.assertEqual(collector.choose('s'), 2)
        # Unknown state labels are never invented and no hidden map is available.
        self.assertEqual(set(collector.transitions.values()), {'s', 'observed_x'})

    def test_autonomous_position_continuity_and_explicit_body_reset(self):
        for kind in ('random','least_visited'):
            events, _ = collect(TEST_SEED, kind, 1728, CONFIG)
            self.assertEqual(events[0].state, World(TEST_SEED).start)
            self.assertTrue(all(a.next_state == b.state for a,b in zip(events,events[1:])))
            self.assertTrue(all(e.before == tuple(CONFIG['train_body']) for e in events))
            self.assertTrue(any(e.outcome is not None for e in events))
            self.assertEqual(events[:108], collect(TEST_SEED,kind,108,CONFIG)[0])
            self.assertEqual(events[:432], collect(TEST_SEED,kind,432,CONFIG)[0])

    def test_teacher_is_exact_inherited_generator(self):
        events, _ = collect(TEST_SEED,'balanced_teacher',432,CONFIG)
        self.assertEqual(events, self.teacher)
        self.assertEqual(len({(e.state,e.action) for e in events}),36)

    def test_missing_outcomes_remain_failures(self):
        for scenario in ('quiet','blocked','need_shift'):
            row, trace, initial = evaluate([],TEST_SEED,'learned',scenario,CONFIG)
            self.assertEqual(row['success'],0)
            self.assertEqual(row['failure_reason'],'target_not_discovered')
            self.assertEqual(initial['catalog'],[])
            self.assertIsNone(trace[0]['action'])
        row, _, _ = evaluate([],TEST_SEED,'learned','satiated',CONFIG)
        self.assertEqual(row['success'],1)

    def test_observing_an_outcome_does_not_supply_hidden_route(self):
        world = World(TEST_SEED,CONFIG['effect_noise_sd'])
        event = next(e for e in self.teacher if e.outcome==world.outcomes[0])
        row, _, initial = evaluate([event],TEST_SEED,'learned','quiet',CONFIG)
        self.assertEqual(len(initial['transitions']),1)
        self.assertEqual(row['initial_target_discovered'],True)
        self.assertEqual(row['initial_known_route'],False)
        self.assertEqual(row['failure_reason'],'no_initial_observed_route')

    def test_full_history_positive_control_and_body_revaluation(self):
        for scenario in CONFIG['scenarios']:
            row, _, _ = evaluate(self.teacher,TEST_SEED,'learned',scenario,CONFIG)
            self.assertEqual(row['success'],1,scenario)
        row, _, _ = evaluate(self.teacher,TEST_SEED,'cached_value','need_shift',CONFIG)
        self.assertEqual(row['success'],0)

    def test_unannounced_block_detected_only_after_action(self):
        quiet, qt, qi = evaluate(self.teacher,TEST_SEED,'learned','quiet',CONFIG)
        blocked, bt, bi = evaluate(self.teacher,TEST_SEED,'learned','blocked',CONFIG)
        self.assertEqual(qi,bi)
        self.assertEqual(qt[0]['action'],bt[0]['action'])
        self.assertEqual(qt[0]['decision'],bt[0]['decision'])
        self.assertNotEqual(qt[0]['experience']['state'],qt[0]['experience']['next_state'])
        self.assertEqual(bt[0]['experience']['state'],bt[0]['experience']['next_state'])
        self.assertEqual(blocked['success'],1)
        self.assertGreater(blocked['steps'],quiet['steps'])

    def test_snapshots_precede_feedback_and_branches_are_isolated(self):
        original = copy.deepcopy(self.teacher)
        digest = history_hash(self.teacher)
        for variant in CONFIG['variants']:
            expected = copy.deepcopy(fit(self.teacher,variant,TEST_SEED,CONFIG,World(TEST_SEED).outcomes[0]).snapshot())
            for scenario in CONFIG['scenarios']:
                _, _, initial = evaluate(self.teacher,TEST_SEED,variant,scenario,CONFIG)
                self.assertEqual(initial, expected)
        self.assertEqual(self.teacher, original)
        self.assertEqual(history_hash(self.teacher), digest)
        self.assertEqual(evaluate(self.teacher,TEST_SEED,'learned','blocked',CONFIG),
                         evaluate(self.teacher,TEST_SEED,'learned','blocked',CONFIG))

    def test_same_history_except_declared_mechanism_ablation(self):
        models = {v:fit(self.teacher,v,TEST_SEED,CONFIG).snapshot() for v in CONFIG['variants']}
        for model in models.values():
            for field in ('catalog','transitions','emits','outcome_samples'):
                self.assertEqual(model[field],models['learned'][field])
        self.assertEqual(models['no_goal_buffer']['effects'],models['learned']['effects'])
        self.assertEqual(models['erased_effects']['effects'],{})
        self.assertEqual(models['erased_effects']['cached_values'],{})

    def test_test_noise_is_independent_declared_stream(self):
        for scenario in ('quiet','blocked','need_shift'):
            _, trace, _ = evaluate(self.teacher,TEST_SEED,'learned',scenario,CONFIG)
            world = World(TEST_SEED,CONFIG['effect_noise_sd'])
            world.effect_rng = random.Random(TEST_SEED+CONFIG['test_effect_seed_offset']+CONFIG['scenarios'].index(scenario))
            for record in trace:
                if 'experience' in record and record['experience']['outcome'] is not None:
                    e = record['experience']
                    replay = world.step(e['state'],e['action'],tuple(e['before']))
                    self.assertEqual(replay.after,tuple(e['after']))

    def test_rmse_missingness_is_not_reported_as_zero(self):
        world = World(TEST_SEED)
        subset = [e for e in self.teacher if e.outcome != world.outcomes[1]]
        partial = knowledge_metrics(subset,TEST_SEED,CONFIG)
        full = knowledge_metrics(self.teacher,TEST_SEED,CONFIG)
        self.assertEqual(partial['both_discovered'],0)
        self.assertIsNone(partial['effect_rmse'])
        self.assertEqual(full['both_discovered'],1)
        self.assertGreater(full['effect_rmse'],0)

    def test_paired_counts_and_boundary_intervals(self):
        result = paired_bootstrap([(1,0),(1,1),(0,1),(0,0)],5000,12)
        self.assertEqual(result['difference'],0)
        self.assertEqual(result['only_directed'],1)
        self.assertEqual(result['only_random'],1)
        self.assertEqual(result['both'],1)
        self.assertEqual(result['neither'],1)
        self.assertLess(result['percentile95'][0],0)
        self.assertGreater(result['percentile95'][1],0)
        self.assertLess(wilson(128,128)[0],1)
        self.assertGreater(wilson(0,128)[1],0)

    def test_frozen_protocol_and_output_overwrite_guard(self):
        verify_freeze(ROOT/'PROTOCOL_FREEZE.json')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fresh'
            new_output(path)
            with self.assertRaises(FileExistsError):
                new_output(path)
            false_freeze=Path(temporary)/'false.json'
            false_freeze.write_text(json.dumps({'files':{'protocol.json':'0'*64}}))
            with self.assertRaises(RuntimeError):
                verify_freeze(false_freeze)


if __name__ == '__main__':
    unittest.main()
