import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu_unified import codec
from wuyu_unified.runtime import Session
from wuyu_unified.action_learning import ActionLearner,softmax
from wuyu_unified.relations import MovementRelations
from wuyu_unified.relation_world import RelationWorld,CASES


class RelationTests(unittest.TestCase):
    def test_gate_gradient_against_finite_difference(self):
        rng=np.random.default_rng(109)
        a=ActionLearner(5,108,'C',dict(hidden=3,allocate_gate=True,prior_gate='learned'))
        a.p['O'][:]=rng.normal(0,.1,a.p['O'].shape)
        initial=rng.normal(0,.2,3)
        samples=[(rng.normal(size=5),softmax(rng.normal(size=7)),i) for i in range(4)]
        adv=rng.normal(size=4);returns=rng.normal(size=4)
        def compute():
            a.buffer=[];h=initial.copy()
            for x,base,act in samples:
                q=a.forward(x,base,h);h=q['h'];q['action']=act;a.buffer.append(q)
            return a.gradients(adv,returns)
        _,grads,_=compute()
        for name in a.p:
            indices={0,a.p[name].size//2,a.p[name].size-1}
            if name=='O':indices.update(range(8,a.p[name].size,9))
            for ix in indices:
                flat=a.p[name].flat;old=flat[ix];eps=1e-6
                flat[ix]=old+eps;plus=compute()[0]
                flat[ix]=old-eps;minus=compute()[0];flat[ix]=old
                self.assertAlmostEqual((plus-minus)/(2*eps),grads[name].flat[ix],delta=3e-6,msg=(name,ix))

    def test_r00_preserves_old_c_behavior_with_shadow_relations(self):
        a=Session.create('whole',1500991,'C','r_stable',160)
        b=Session.create('whole',1500991,'R00','r_stable',160)
        for _ in range(160):
            x,y=a.step(),b.step()
            self.assertEqual(x['action'],y['action'])
            np.testing.assert_array_equal(x['after'],y['after'])
            np.testing.assert_allclose(x['decision']['final_probability'],y['decision']['final_probability'],atol=1e-13)
        for key in ('network','memory','rng','h','prev','predictor'):
            self.assertEqual(codec.digest(a.backend.native.individual.state()[key]),codec.digest(b.backend.native.individual.state()[key]))

    def test_relation_learning_is_observed_revisable_and_pure(self):
        m=MovementRelations();obs=np.zeros(21);obs[0]=1;obs[14:16]=.8
        good=obs.copy();good[0]=0;good[1]=1
        self.assertEqual(m.count.sum(),0);self.assertEqual(m.success(0)[0,1],1.)
        m.update(obs,1,obs,0);m.update(obs,1,obs,1)
        low=m.success(2)[0,1];self.assertLess(low,.4)
        before=codec.digest(m.state());m.distances(0,2);m.distribution(1000)
        self.assertEqual(before,codec.digest(m.state()))
        self.assertGreater(m.success(1000)[0,1],low)
        m.update(obs,1,good,2);self.assertGreater(m.success(3)[0,1],low)
        self.assertEqual(m.count[0,0],0)
        np.testing.assert_allclose(m.distribution(3).sum(2),1.)
        obs[14]=0;before=codec.digest(m.state());m.update(obs,1,good,4)
        self.assertEqual(before,codec.digest(m.state()))

    def test_world_truth_cannot_affect_pre_feedback_decision(self):
        a=Session.create('whole',1500992,'R11','r_block',2560);b=Session.from_state(a.state())
        w=b.backend.native.world;p=int(w.pos[0]);w.blocked_edge=[p,(p+1)%6]
        for s in (a,b):s.backend.native.individual.ingest(s.backend.native.world.observation())
        obs=w.observation()[0]
        self.assertEqual(codec.digest(a.backend.decision(obs)),codec.digest(b.backend.decision(obs)))
        a.backend.step(forced=1);b.backend.step(forced=1)
        self.assertNotEqual(codec.digest(a.backend.relations.state()),codec.digest(b.backend.relations.state()))

    def test_relational_encoding_is_used_by_hidden_commit(self):
        s=Session.create('whole',1500993,'R10','r_block',2560);b=s.backend
        obs=b.native.world.observation()[0];b.native.individual.ingest(obs[None])
        pos=int(obs[:6].argmax());b.relations.update(obs,1,obs,0)
        enc=b.encode_observation(obs)
        expected=b.native.individual.net.forward(enc,b.native.individual.h)['h'].copy()
        b.step(forced=1)
        np.testing.assert_array_equal(expected,b.native.individual.h)

    def test_complete_resume_mid_buffer_and_pure_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            for mode in ('R00','R10','R01','R11','RT'):
                s=Session.create('whole',1500994,mode,'r_noise',2560);s.run(73)
                p=Path(td)/(mode+'.wuyu');s.save(p);r=Session.load(p)
                obs=s.backend.native.world.observation()[0]
                s.backend.native.individual.ingest(obs[None]);r.backend.native.individual.ingest(obs[None])
                before=codec.digest(s.state(False));s.backend.decision(obs);s.backend.decision(obs)
                self.assertEqual(before,codec.digest(s.state(False)))
                self.assertEqual(codec.digest(s.run(42)),codec.digest(r.run(42)))
                self.assertEqual(codec.digest(s.state(False)),codec.digest(r.state(False)))

    def test_reopening_noise_and_world_restore(self):
        for case in CASES:
            w=RelationWorld(1500995,2560,case);r=RelationWorld.from_state(w.state())
            for i in range(40):
                a=np.array([i%7]);self.assertEqual(codec.digest(w.step(a)),codec.digest(r.step(a)))
        w=RelationWorld(1500995,2560,'r_reopen');w.t=w.recovery_tick-1
        p=int(w.pos[0]);w.blocked_edge=[p,(p+1)%6]
        old=w.pos.copy();o=w.step(np.array([1]));np.testing.assert_array_equal(w.pos,old)
        self.assertTrue(o['structural_denied']);self.assertIsNone(w.blocked_edge)
        w.step(np.array([1]));self.assertNotEqual(int(w.pos[0]),p)

    def test_gating_is_real_and_fixed_control_has_equal_allocated_parameters(self):
        methods=[Session.create('whole',1500996,m,'r_stable',40) for m in ('R00','R10','R01','R11','RT')]
        self.assertEqual(len(set(s.backend.actor.count for s in methods)),1)
        for s in methods:
            r=s.step();tau=r['movement']['prior_tau']
            self.assertAlmostEqual(tau,1. if s.backend.condition in ('R00','R10') else .9)
            self.assertGreater(tau,0);self.assertLessEqual(tau,1)


if __name__=='__main__':unittest.main()
