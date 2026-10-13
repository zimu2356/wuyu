import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
import numpy as np

from wuyu_unified import codec
from wuyu_unified.adapters import ContinuousBackend,GoalBackend,ExplorationBackend
from wuyu_unified.audit import verify_sources
from wuyu_unified.evidence_model import WindowedEvidenceModel
from wuyu_unified.migration import import_native,migrate08
from wuyu_unified.paths import ROOT,project,source_module,config
from wuyu_unified.registry import DESCRIPTIONS
from wuyu_unified.runtime import Session

def same(a,b):return codec.canonical(codec.jsonable(a))==codec.canonical(codec.jsonable(b))

class IntegrationTests(unittest.TestCase):
    def test_all_combined_mechanisms_resume_exact_future(self):
        for mode in ('online','evidence','update','revision','combined'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                s=Session.create('workshop_bridge',seed=920001,mode=mode,horizon=512)
                s.run(128);p=Path(d)/'state.wuyu';s.save(p);other=Session.load(p)
                self.assertTrue(same(s.run(64),other.run(64)))
                self.assertEqual(codec.digest(s.state(False)),codec.digest(other.state(False)))
                if mode=='combined':
                    self.assertGreater(s.backend.native.individual.predictor.replay_updates,0)
                    self.assertGreater(sum(s.backend.visits),0)
                    self.assertGreater(s.backend.native.individual.memory.updates,0)

    def test_combined_decisions_and_probes_do_not_train_or_advance_rng(self):
        s=Session.create('workshop_bridge',seed=920003,mode='combined',horizon=512)
        s.run(90);b=s.backend;obs=b.native.world.observation()[0]
        b.native.individual.ingest(obs[None]);old=codec.digest(b.state())
        first=b.candidate_decision(obs);second=b.candidate_decision(obs)
        self.assertTrue(same(first,second));self.assertEqual(old,codec.digest(b.state()))
        self.assertIsNotNone(first['neural_predictions']);self.assertTrue(first['use'])

    def test_native_history_import_rejoins_exact_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'native'
            subprocess.run([sys.executable,'-B',str(project('history')/'tools/continue_life.py'),
                            '--seed','51000','--ticks','80','--output',str(output)],check=True,stdout=subprocess.DEVNULL)
            s=import_native('history',output/'state.json')
            raw=json.loads((output/'state.json').read_text());m=source_module('history');c=config('history')
            w=m.World.restore(raw['world'],c);a=m.Agent.restore(raw['agent'],c)
            for _ in range(32):
                o=w.observation();action,decision=a.decide(o);after=w.step(action)
                a.observe(o,action,after,f'{w.seed}:life:{o["tick"]}')
                q=s.step();self.assertEqual(q['action'],action);self.assertTrue(same(q['after'],after))
            self.assertEqual(m.digest(a.snapshot()),m.digest(s.backend.agent.snapshot()))

    def test_goal_horizon_cannot_silently_replace_native_episode_semantics(self):
        with self.assertRaisesRegex(ValueError,'episodic action budget'):
            Session.create('goals',horizon=99)

    def test_original_scientific_sources_unchanged(self):
        result=verify_sources()
        self.assertTrue(result['passed'],result)

    def test_complete_typed_state_preserves_rng_and_tuple_keys(self):
        rng=random.Random(77)
        value={'array':np.arange(18,dtype=np.int16).reshape(3,6),'table':{('s',2):'next'},
               'set':{'a','b'},'rng':rng.getstate(),'literal':{'type':'dict','items':[]}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.wuyu';codec.save(p,value);restored=codec.load(p)
            self.assertEqual(codec.digest(value),codec.digest(restored))
            other=random.Random();other.setstate(restored['rng'])
            self.assertEqual([rng.random() for _ in range(10)],[other.random() for _ in range(10)])
            with self.assertRaises(FileExistsError):codec.save(p,value)

    def test_checkpoint_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.wuyu';codec.save(p,{'hello':1})
            data=json.loads(gzip.decompress(p.read_bytes()));data['sha256']='0'*64
            p.write_bytes(gzip.compress(json.dumps(data).encode()))
            with self.assertRaisesRegex(ValueError,'checksum'):codec.load(p)

    def test_all_profiles_resume_same_future(self):
        with tempfile.TemporaryDirectory() as d:
            for profile in DESCRIPTIONS:
                with self.subTest(profile=profile):
                    horizon=64 if profile in ('01','02','03','exploration','history','revision','integrated','workshop_bridge') else None
                    s=Session.create(profile,seed=71000,horizon=horizon)
                    s.run(2);p=Path(d)/(profile+'.wuyu');s.save(p)
                    restored=Session.load(p)
                    self.assertTrue(same(s.run(5),restored.run(5)))
                    self.assertEqual(codec.digest(s.state(False)),codec.digest(restored.state(False)))

    def test_legacy_adapters_match_all_eight_native_futures(self):
        source_module('integrated');legacy=importlib.import_module('wuyu_core.legacy')
        for version in range(1,9):
            with self.subTest(version=version):
                native=legacy.LegacySession.create(version,71003,64)
                wrapped=Session.create(f'{version:02}',71003,horizon=64 if version<=3 else None)
                for _ in range(8):self.assertTrue(same(native.step(),wrapped.step()['payload']))

    def test_declared_horizon_is_enforced_without_reset(self):
        for profile in ('history','revision','exploration','integrated','workshop_bridge'):
            with self.subTest(profile=profile):
                s=Session.create(profile,seed=71001,horizon=3)
                self.assertEqual(len(s.run(10)),3)
                before=codec.digest(s.state(False))
                self.assertEqual(s.run(10),[])
                self.assertEqual(codec.digest(s.state(False)),before)

    def test_source_lock_rejects_silent_implementation_change(self):
        s=Session.create('revision');state=s.state()
        first=next(iter(state['source_locks']));state['source_locks'][first]='0'*64
        with self.assertRaisesRegex(ValueError,'source/configuration'):Session.from_state(state)

    def test_fork_erases_only_selected_model_and_parent_is_unchanged(self):
        for profile in ('history','revision'):
            with self.subTest(profile=profile):
                s=Session.create(profile,seed=71000);s.run(100)
                parent=codec.digest(s.state(False));before=s.backend.agent.snapshot()
                child=s.fork('erase_model',2)
                self.assertEqual(len(child.backend.agent.models[2].samples),0)
                after=child.backend.agent.snapshot()
                for i in (0,1,3,4):self.assertTrue(same(before['models'][i],after['models'][i]))
                self.assertTrue(same(s.backend.world.snapshot(),child.backend.world.snapshot()))
                child.run(20)
                self.assertEqual(codec.digest(s.state(False)),parent)
                frozen=s.fork('freeze_learning');weights=codec.digest(frozen.backend.agent.snapshot()['models'])
                frozen.run(20)
                self.assertEqual(codec.digest(frozen.backend.agent.snapshot()['models']),weights)

    def test_continuing_history_adapter_matches_original_feedback_order(self):
        m=source_module('history');c=config('history');seed=41000
        w=m.World(seed,c);a=m.Agent(seed,c);s=Session.create('history',seed)
        for tick in range(1800):
            o=w.observation();action,d=a.decide(o);after=w.step(action)
            a.observe(o,action,after,f'{seed}:main:{tick}')
            r=s.step();self.assertEqual(r['action'],action)
            self.assertTrue(same(r['decision'],d));self.assertTrue(same(r['after'],after))
        self.assertEqual(m.digest(a.snapshot()),m.digest(s.backend.agent.snapshot()))

    def test_generic_evidence_model_matches_original_including_revisions(self):
        m=source_module('revision');c=config('revision');seed=71000
        w=m.World(seed,'local_swap',c);a=m.Agent(seed,c,'BEU',w.n_actions)
        models=[WindowedEvidenceModel(6,3,c) for _ in range(w.n_actions)]
        triggers=0
        for tick in range(1700):
            o=w.observation();action,d=a.decide(o);x=m.features(o)
            pred=models[action].predict(x)
            np.testing.assert_allclose(pred,d['pred'][action],rtol=0,atol=1e-12)
            after=w.step(action);event_id=f'{seed}:local_swap:BEU:{tick}'
            expected=a.observe(o,action,after,event_id,d['pred'][action])
            y=np.array([after['body'][i]-o['body'][i] for i in range(2)]+[after['fatigue']-o['fatigue']])
            actual=models[action].update(x,y,tick,event_id,pred,True)
            self.assertEqual(expected[1],actual[1]);self.assertAlmostEqual(expected[0],actual[0],places=12)
            np.testing.assert_allclose(models[action].coef,a.models[action].coef,rtol=0,atol=1e-12)
            self.assertEqual(models[action].samples,a.models[action].samples)
            triggers+=int(actual[1])
        self.assertGreater(triggers,0)

    def test_exploration_preserves_actual_history_and_body_reset(self):
        m=source_module('exploration');c=config('exploration')
        for mode in ('random','least_visited'):
            s=Session.create('exploration',20000,mode,horizon=108)
            rows=s.run(108);original,_=m.collect(20000,mode,108,c)
            self.assertTrue(same([r['experience'] for r in rows],[asdict(e) for e in original]))
            self.assertTrue(all(r['body_reset_before_action'] for r in rows))
            parent=codec.digest(s.state(False))
            for variant in ('learned','no_goal_buffer','cached_value','erased_effects','fixed_target'):
                for case in c['scenarios']:
                    _,trace,_=m.evaluate(original,20000,variant,case,c)
                    branch=s.evaluation(case,variant);actual=branch.run(12)
                    self.assertEqual([r['action'] for r in actual],[r['action'] for r in trace])
                    self.assertTrue(same([r['decision'] for r in actual],[r['decision'] for r in trace]))
            self.assertEqual(codec.digest(s.state(False)),parent)

    def test_goal_profile_matches_original_all_variants_and_cases(self):
        directory=project('goals');sys.path.insert(0,str(directory))
        spec=importlib.util.spec_from_file_location('_goal_original_runner',directory/'run_study.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        m=source_module('goals');c=config('goals');seed=7
        events=m.training_experiences(seed,c['training_repeats_per_state_action'],c['effect_noise_sd'],tuple(c['train_body']))
        for variant in c['variants']:
            for case in c['scenarios']:
                w=m.World(seed,c['effect_noise_sd']);a=m.Learner(variant,seed,c['action_cost'],w.outcomes[0] if variant=='fixed_target' else None)
                for e in events:a.observe(e)
                if case=='blocked':w.block_short_a()
                body=c['shift_body'] if case=='need_shift' else c['satiated_body'] if case=='satiated' else c['train_body']
                expected=runner.episode(a,w,tuple(body),c['max_steps'])['trace']
                s=Session.create('goals',seed,variant,case);actual=s.run(12)
                self.assertEqual([r['action'] for r in actual],[r['action'] for r in expected])
                self.assertTrue(same([r['decision'] for r in actual],[r['decision'] for r in expected]))

    def test_workshop_shadow_keeps_native_policy_body_and_memory(self):
        native=Session.create('integrated',71004,mode='prior',horizon=384)
        bridge=Session.create('workshop_bridge',71004,mode='shadow',horizon=384)
        for _ in range(384):
            a=native.step();b=bridge.step()
            self.assertEqual(a['action'],b['action']);np.testing.assert_array_equal(a['after'],b['after'])
        self.assertEqual(codec.digest(native.backend.native.state()),codec.digest(bridge.backend.native.state()))
        self.assertGreater(sum(m.version for m in bridge.backend.models),0)

    def test_workshop_only_actual_action_receives_evidence_after_feedback(self):
        s=Session.create('workshop_bridge',71005,mode='revision',horizon=128)
        for tick in range(80):
            versions=[m.version for m in s.backend.models]
            r=s.step();action=r['action']
            for i,m in enumerate(s.backend.models):
                self.assertEqual(m.version,versions[i]+int(i==action and r['learning']['updated']))
                self.assertTrue(all(sample['tick']<=tick for sample in m.samples))
            if r['learning']['updated']:
                np.testing.assert_array_equal(s.backend.models[action].samples[-1]['y'],r['learning']['target'])
        self.assertTrue(r['decision']['use'])

    def test_workshop_visible_prediction_is_pure_and_schema_checked(self):
        s=Session.create('workshop_bridge',71008);s.run(40)
        before=codec.digest(s.backend.state());obs=s.backend.native.world.observation()[0]
        x=s.backend.candidate_decision(obs);y=s.backend.candidate_decision(obs)
        self.assertTrue(same(x,y));self.assertEqual(codec.digest(s.backend.state()),before)
        state=s.backend.state();state['feature_schema']='wrong-world'
        with self.assertRaisesRegex(ValueError,'incompatible'):type(s.backend).restore(state)

    def test_native_revision_import_matches_original_continuation(self):
        p=project('revision')/'holdout/71000_local_swap_BEU.json.gz'
        with gzip.open(p,'rt') as f:data=json.load(f)
        imported=import_native('revision',p);m=source_module('revision');c=config('revision')
        w=m.World.restore(data['final_world'],c);a=m.Agent.restore(data['final_agent'],c)
        for _ in range(30):
            o=w.observation();act,d=a.decide(o);after=w.step(act)
            a.observe(o,act,after,f'continuation:{o["tick"]}',d['pred'][act])
            r=imported.step();self.assertEqual(r['action'],act);self.assertTrue(same(r['after'],after))
        self.assertEqual(m.digest(a.snapshot()),m.digest(imported.backend.agent.snapshot()))

    def test_migrate08_preserves_declared_content_and_records_rng_change(self):
        p=project('08')/'results/checkpoint_graph_varied_0_384.npz'
        s=migrate08(p,0)
        self.assertEqual(s.backend.native.world.t,384)
        self.assertFalse(s.lineage['stochastic_future_equivalent'])
        self.assertEqual(s.backend.native.individual.predictor.seen,0)
        clone=s.fork();self.assertTrue(same(s.run(10),clone.run(10)))

    def test_cli_cross_process_resume_is_byte_identical(self):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            calls=[['run','--profile','revision','--case','local_swap','--mode','BEU','--seed','71000','--steps','1100','--out',str(root/'whole')],
                   ['run','--profile','revision','--case','local_swap','--mode','BEU','--seed','71000','--steps','1000','--out',str(root/'first')],
                   ['resume',str(root/'first/state.wuyu'),'--steps','100','--out',str(root/'second')]]
            for args in calls:
                result=subprocess.run([sys.executable,'-B','-m','wuyu_unified',*args],cwd=ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'whole/state.wuyu').read_bytes(),(root/'second/state.wuyu').read_bytes())
            self.assertEqual((root/'whole/events.jsonl').read_bytes(),(root/'first/events.jsonl').read_bytes()+(root/'second/events.jsonl').read_bytes())

if __name__=='__main__':unittest.main()
