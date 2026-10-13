import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu_unified import codec
from wuyu_unified.runtime import Session
from wuyu_unified.whole import target,corrected_feedback
from wuyu_unified.workshop_bridge import target as old_target
from wuyu_unified.action_learning import ActionLearner,softmax
from wuyu_unified.whole_world import WholeWorld
from wuyu_unified.whole_migration import migrate_whole
from wuyu_unified.paths import ROOT

class WholeTests(unittest.TestCase):
    def test_cross_device_stage_is_not_progress_in_either_predictor(self):
        s=Session.create('whole',1300990,'A',horizon=40);b=s.backend
        w=b.native.world;pos=int(w.pos[0]);w.stage[0,pos]=0;w.stage[0,(pos+1)%6]=2
        before=w.observation()[0].copy();r=b.step(forced=1)
        self.assertEqual(old_target(before,r['after'])[4],1.)
        self.assertEqual(r['learning']['target'][4],0.)
        self.assertEqual(r['native_payload']['target'][4],0.)
        self.assertEqual(b.native.individual.predictor.memory[-1]['y'][4],0.)
        self.assertEqual(b.models[1].samples[-1]['y'][4],0.)

    def test_zero_residual_recovers_base_and_counts_match(self):
        sessions=[Session.create('whole',1300991,m,horizon=96) for m in 'ABCD']
        counts=[s.backend.actor.count for s in sessions[1:]];self.assertEqual(len(set(counts)),1)
        for _ in range(32):
            rs=[s.step() for s in sessions]
            for r in rs[1:]:
                self.assertEqual(rs[0]['action'],r['action'])
                np.testing.assert_array_equal(rs[0]['after'],r['after'])
                np.testing.assert_allclose(r['decision']['probability'],r['decision']['final_probability'],atol=1e-15)

    def test_actor_gradients_all_cells_against_finite_differences(self):
        rng=np.random.default_rng(93)
        for mode in 'BCD':
            a=ActionLearner(5,91,mode,dict(hidden=3));a.p['O'][:]=rng.normal(0,.2,a.p['O'].shape)
            data=[(rng.normal(size=5),softmax(rng.normal(size=7)),i%7) for i in range(5)]
            initial=rng.normal(0,.2,3);adv=rng.normal(size=5);ret=rng.normal(size=5)
            def compute():
                a.buffer=[];h=initial.copy()
                for x,base,action in data:
                    q=a.forward(x,base,h);h=q['h'].copy();q.update(action=action)
                    a.buffer.append(q)
                return a.gradients(adv,ret)
            _,g,_=compute()
            for name in a.p:
                for flat in (0,a.p[name].size//2,a.p[name].size-1):
                    ix=np.unravel_index(flat,a.p[name].shape);old=a.p[name][ix];eps=1e-6
                    a.p[name][ix]=old+eps;plus=compute()[0]
                    a.p[name][ix]=old-eps;minus=compute()[0];a.p[name][ix]=old
                    self.assertAlmostEqual((plus-minus)/(2*eps),g[name][ix],delta=3e-6,msg=f'{mode} {name} {ix}')

    def test_full_resume_mid_multistep_buffer_and_diagnostic_parent(self):
        with tempfile.TemporaryDirectory() as td:
            for mode in 'BCD':
                s=Session.create('whole',1300992,mode,horizon=160);s.run(73)
                self.assertEqual(len(s.backend.actor.buffer),9)
                p=Path(td)/(mode+'.wuyu');s.save(p);r=Session.load(p)
                before=codec.digest(s.state(False));child=s.fork('suppress_work_state');child.run(4)
                self.assertEqual(before,codec.digest(s.state(False)))
                self.assertEqual(codec.digest(s.run(63)),codec.digest(r.run(63)))
                self.assertEqual(codec.digest(s.state(False)),codec.digest(r.state(False)))

    def test_decision_reads_are_pure(self):
        s=Session.create('whole',1300993,'D',horizon=128);s.run(70);b=s.backend
        obs=b.native.world.observation()[0];b.native.individual.ingest(obs[None])
        before=codec.digest(b.state());x=b.decision(obs);y=b.decision(obs)
        self.assertEqual(codec.digest(x),codec.digest(y));self.assertEqual(before,codec.digest(b.state()))

    def test_value_prior_uses_current_body_and_does_not_change_reward(self):
        s=Session.create('whole',1300998,'D',horizon=96);b=s.backend
        before=b.native.world.observation()[0].copy();expected=-np.sum((1-before[14:16])**2)/(1-b.actor.c['gamma'])
        r=s.step();self.assertAlmostEqual(r['decision']['value_prediction'],expected)
        self.assertEqual(b.actor.buffer[-1]['reward'],-r['body_burden'])
        self.assertEqual(r['decision']['chosen'],r['decision']['base_choice'])

    def test_world_route_cost_yield_and_continuous_debits(self):
        w=WholeWorld(1300994,2560);w.schedule=[];old=int(w.pos[0]);w.blocked_edge=[old,(old+1)%6]
        body=w.body.copy();out=w.step(np.array([1]))
        self.assertEqual(int(w.pos[0]),old);self.assertTrue(out['route_denied'])
        np.testing.assert_allclose(w.body,body-[.003*.55,.0018])
        site=int(np.flatnonzero(w.products[0]==0)[0]);w.pos[0]=site;w.stage[0,site]=2
        w.inactive_site=site;body=w.body.copy();out=w.step(np.array([5]))
        self.assertEqual(w.last[0],0);np.testing.assert_allclose(w.body,body-[.004*.55,.0018])
        w=WholeWorld(1300995,2560);w.t=191;w.body[0]=[.31,.34]
        before=w.body.copy();out=w.step(np.array([6]))
        np.testing.assert_allclose(w.body,before-[.0018*.55,.0018]-out['shock'])
        self.assertEqual(len(out['physical_events']),1)
        for seed in range(1300000,1300040):
            for split in ('development','holdout'):
                self.assertTrue(WholeWorld(seed,2560,split=split).feasibility['acyclic'])

    def test_dead_is_absorbing_and_actor_stops_training(self):
        s=Session.create('whole',1300996,'D',horizon=40);w=s.backend.native.world
        w.body[0]=[.0001,.4];w.alive[:]=False
        old=s.backend.actor.state();s.run(40)
        self.assertEqual(codec.digest(old),codec.digest(s.backend.actor.state()))
        self.assertFalse(w.alive[0]);np.testing.assert_array_equal(w.body[0],[.0001,.4])

    def test_explicit_old_migration_relabels_real_history(self):
        old=Session.create('workshop_bridge',1300997,'combined',horizon=160);old.run(70)
        state=old.state();locks=state['source_locks']
        import hashlib
        # Build the exact frozen-v1 provenance, not the new runtime hashes.
        for rel in list(locks):
            if rel.startswith('wuyu_unified/'):
                p=ROOT/'reference_runtime_v1'/rel.split('/',1)[1]
                if p.is_file():locks[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
                else:del locks[rel]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'old.wuyu';codec.save(p,state);new=migrate_whole(p,'D')
        self.assertEqual(codec.digest(old.backend.native.world.state()),codec.digest(new.backend.native.world.state()))
        for key in ('network','memory','rng','h','prev'):
            self.assertEqual(codec.digest(old.backend.native.individual.state()[key]),codec.digest(new.backend.native.individual.state()[key]))
        self.assertTrue(new.backend.migration['neural_reinitialized'])
        for m in new.backend.models[:2]:self.assertTrue(all(x['y'][4]==0 for x in m.samples))
        clone=Session.from_state(new.state());self.assertEqual(codec.digest(new.run(30)),codec.digest(clone.run(30)))

if __name__=='__main__':unittest.main()
