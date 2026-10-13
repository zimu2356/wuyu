import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu_core.runtime import Session, Individual, load_network
from wuyu_core.features import encode
from wuyu_core.memory import ExperienceMemory
from wuyu_core.predictor import Predictor
from wuyu_core.legacy import LegacySession, module
from wuyu_core.state import read_tree, write_tree


def equal(a,b):
    if isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b,(a,b)


class IntegrationTests(unittest.TestCase):
    def make(self,seed=991011,mode='replay'): return Session.create(seed,0,mode,total=256,shocks=False)

    def test_legacy_all_eight_resume(self):
        for v in range(1,9):
            with self.subTest(version=v), tempfile.TemporaryDirectory() as d:
                a=LegacySession.create(v,horizon=64)
                for _ in range(3):a.step()
                p=Path(d)/'state.npz';a.save(p);b=LegacySession.load(p)
                for _ in range(6):equal(a.step(),b.step())
                # The complete envelope, including old world streams, is retained.
                equal(a.state(),b.state())

    def test_round_trip_complete_future(self):
        a=self.make()
        for _ in range(41):a.step()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.npz';a.save(p);b=Session.load(p)
            for _ in range(55):equal(a.step(),b.step())
            equal(a.state(),b.state())

    def test_pure_features_and_idempotent_observation(self):
        a=self.make().individual;o=self.make().world.observation()
        a.ingest(o);s=a.state()
        for _ in range(4):
            a.ingest(o);encode(o,a.prev,[a.memory])
        equal(s,a.state())
        changed=o.copy();changed[0,14]-=.1
        with self.assertRaises(ValueError):a.ingest(changed)

    def test_no_learning_before_feedback_or_duplicate_decision(self):
        s=self.make();a=s.individual;before=copy.deepcopy(a.predictor.p)
        a.decide(s.world.observation())
        equal(before,a.predictor.p);self.assertEqual(a.predictor.seen,0)
        with self.assertRaises(RuntimeError):a.decide(s.world.observation())
        with self.assertRaises(RuntimeError):self.make().individual.feedback(s.world.observation())

    def test_no_planner_or_hidden_world_argument(self):
        for name in ('act','plan','_route'):self.assertFalse(hasattr(ExperienceMemory,name))
        import inspect
        self.assertEqual(list(inspect.signature(Individual.decide).parameters),['self','obs','greedy','forced'])
        s=self.make();a=s.individual;b=Individual.from_state(a.state())
        world=s.world.one(0);world.codes[:]=(world.codes+1)%3
        self.assertEqual(a.decide(s.world.observation()),b.decide(world.observation()))

    def test_individual_extraction_and_reordering(self):
        batch=[self.make(991020+i) for i in range(3)]
        for _ in range(17):
            for a in batch:a.step()
        solo=batch[1].fork();reordered=[a.fork() for a in batch[::-1]]
        for _ in range(45):
            records=[a.step() for a in batch]
            equal(records[1],solo.step())
            for i,a in enumerate(reordered):equal(records[2-i],a.step())
        equal(batch[1].state(),solo.state())

    def test_branch_no_shared_mutable_parameters(self):
        a=self.make();b=a.fork();before=b.state()
        for _ in range(8):a.step()
        equal(before,b.state())

    def test_predictor_gradient(self):
        p=Predictor(111,input_size=5,hidden=3,outputs=2)
        rng=np.random.default_rng(47);xs=rng.normal(size=(4,5));y=np.array([.2,-.3])
        _,g=p.gradients(xs,y);eps=1e-6
        for key,index in [('W',(2,1)),('R',(1,2)),('V',(1,0)),('b',(2,)),('c',(1,))]:
            old=p.p[key][index];p.p[key][index]=old+eps;hi=p.gradients(xs,y)[0]
            p.p[key][index]=old-eps;lo=p.gradients(xs,y)[0];p.p[key][index]=old
            self.assertAlmostEqual((hi-lo)/(2*eps),g[key][index],places=8)

    def test_replay_has_actual_prefix_and_no_future(self):
        p=Predictor(11,input_size=5,hidden=3,outputs=2,capacity=64,burn=4)
        history=[]
        for t in range(12):
            x=np.full(5,t/12);history.append(x)
            p.feedback(x,np.zeros(2),replay=True)
            for m in p.memory:
                at=m['event'];equal(m['xs'],np.array(history[max(0,at-4):at+1]));self.assertLessEqual(at,t)
        self.assertEqual(p.replay_updates,11)
        # Replay recomputation depends on current recurrent weights, not a saved h.
        before=p.sequence(p.memory[-1]['xs'])[0];p.p['R']+=.7
        self.assertGreater(np.max(np.abs(before-p.sequence(p.memory[-1]['xs'])[0])),1e-6)

    def test_only_predictor_weights_plastic(self):
        a=self.make();net=copy.deepcopy(a.individual.net.p);pred=copy.deepcopy(a.individual.predictor.p)
        for _ in range(48):a.step()
        equal(net,a.individual.net.p)
        self.assertGreater(sum(np.linalg.norm(pred[k]-a.individual.predictor.p[k]) for k in pred),0)

    def test_predictor_has_causal_action_path(self):
        s=self.make();a=s.individual;a.warmup=0;b=Individual.from_state(a.state());b.mode='shadow'
        def predictions(context):
            pred=np.zeros((7,6));pred[0,0]=1;pred[1,0]=-1
            return pred,np.zeros((7,68))
        a.predictor.predict=predictions;b.predictor.predict=predictions
        a.decide(s.world.observation());b.decide(s.world.observation())
        self.assertGreater(a.pending['probability'][0]/a.pending['probability'][1],b.pending['probability'][0]/b.pending['probability'][1])

    def test_shadow_preserves_prior_behavior(self):
        a=self.make(mode='prior');b=self.make(mode='shadow')
        for _ in range(75):
            x,y=a.step(),b.step();self.assertEqual(x['action'],y['action']);equal(a.world.state(),b.world.state())

    def test_invalid_schema_and_tamper_rejected(self):
        s=self.make().state();s['schema']='future/99'
        with self.assertRaises(ValueError):Session.from_state(s)
        a=LegacySession.create(2,horizon=4).state();a['inner'][0]^=1
        with self.assertRaises(ValueError):LegacySession.from_state(a)

    def test_old08_neural_path_preserved_under_greedy_actions(self):
        # Same observed history must produce identical prior logits, without
        # relying on old shared-batch RNG behaviour.
        module(8)
        from wuyu08.individual import Individual as Old
        from wuyu08.network import Network as OldNetwork
        a=self.make(mode='prior');new=a.individual
        old=Old(OldNetwork.from_state(new.net.state()),[new.seed],new.memory.initialization)
        for _ in range(40):
            obs=a.world.observation();act=new.decide(obs,greedy=True);r=old.act(obs,greedy=True)
            self.assertEqual(act,int(r['action'][0]));np.testing.assert_allclose(new.pending['probability'],r['ap'][0],rtol=1e-12)
            a.world.step(np.array([act]));no=a.world.observation();new.feedback(no);old.feedback(r,no)

    def test_migration_preserves_experience_and_greedy_future(self):
        from wuyu_core.migration import migrate_08, ROOT
        old_module=module(8);path=ROOT/'legacy/WUYU_08/results/checkpoint_graph_varied_0_384.npz'
        w,a=old_module.restore(path);w=w.one(1);a=a.one(1)
        new=migrate_08(path,1)
        equal(w.state(),new.world.state());equal(a.memories[0].state(),new.individual.memory.state())
        for _ in range(30):
            old=old_module.neural_step(w,a,True);r=new.step(greedy=True)
            self.assertEqual(int(old['action'][0]),r['action']);equal(w.state(),new.world.state())
        self.assertFalse(new.lineage['stochastic_future_equivalent'])
        equal(new.lineage,new.fork().lineage)


if __name__=='__main__':unittest.main()
