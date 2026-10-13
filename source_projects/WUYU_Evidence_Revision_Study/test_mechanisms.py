import importlib.util, math, tempfile, unittest
from pathlib import Path
import numpy as np
from core import Agent, World, LocalModel, features, digest
from protocol import fresh_config
from storage import write_events,read_events,DTYPE

class Mechanisms(unittest.TestCase):
    def setUp(self): self.c=fresh_config()
    def test_agent_cannot_see_schedule_or_tools(self):
        a=Agent(9,self.c,'BEU',5)
        self.assertNotIn('change_low',a.c);self.assertNotIn('body_drift',a.c)
        self.assertFalse(hasattr(a,'world'));self.assertEqual(set(World(9,'local_swap',self.c).observation()),{'tick','body','fatigue','material'})
    def test_swap_preserves_both_directions_and_other_axis(self):
        w=World(8,'local_swap',self.c);o=w.observation();o['tick']=w.tau-1
        before=[w.mean_delta(o,a) for a in range(1,5)];o['tick']=w.tau
        after=[w.mean_delta(o,a) for a in range(1,5)]
        for a in range(1,5):
            axis=w.tools[a]['axis']
            if axis!=w.changed_axis: np.testing.assert_array_equal(before[a-1],after[a-1])
        self.assertEqual(sorted(w.tools[a]['sign'] for a in range(1,5) if w.tools[a]['axis']==w.changed_axis),[-1,1])
    def test_silent_opportunity_common_actions_identical(self):
        w=World(12,'opportunity',self.c);o=w.observation();p=dict(o,tick=w.tau)
        for a in range(5): np.testing.assert_array_equal(w.mean_delta(o,a),w.mean_delta(p,a))
        self.assertGreater(np.linalg.norm(w.mean_delta(p,5)-w.mean_delta(o,5)),.1)
    def test_recurrence_returns_same_mapping(self):
        w=World(12,'recurrence',self.c);o=w.observation();p=dict(o,tick=w.tau+self.c['recurrence_gap'])
        for a in range(5): np.testing.assert_array_equal(w.mean_delta(o,a),w.mean_delta(p,a))
    def test_noise_does_not_change_mapping(self):
        w=World(12,'noise_pulse',self.c);s=World(12,'stable',self.c);o=w.observation();o['tick']=w.tau
        for a in range(5): np.testing.assert_array_equal(w.mean_delta(o,a),s.mean_delta(o,a))
        self.assertGreater(w.noise_scale(w.tau),w.noise_scale(w.tau+128))
    def test_visible_shock_preserves_conditional_tools(self):
        w=World(12,'visible_shock',self.c);s=World(12,'stable',self.c);w.tick=w.tau
        after=w.step(0);self.assertEqual(after['fatigue'],.95)
        for a in range(5): np.testing.assert_array_equal(w.mean_delta(after,a),s.mean_delta(after,a))
    def test_last_use_not_total_count_drives_E(self):
        a=Agent(1,self.c,'BE',5);a.visits=[100]*5;a.last_use=[99,0,99,99,99]
        o=dict(tick=100,body=[0.,0.],fatigue=.2,material=.2)
        _,_,_,bonus=a.candidates(o);self.assertGreater(bonus[1],bonus[0]);self.assertLessEqual(max(bonus),self.c['recency_cap'])
    def test_three_aligned_contradictions_local_revision(self):
        a=Agent(1,self.c,'BU',5);o=dict(tick=0,body=[0.,0.],fatigue=.2,material=.2)
        for t in range(10):
            o['tick']=t;a.observe(o,1,dict(o,tick=t+1),str(t),np.zeros(3))
        unrelated=digest(a.models[2].snapshot())
        for t in range(3):
            o['tick']=20+t;after=dict(o,tick=21+t,body=[.2,0.])
            _,trigger=a.observe(o,1,after,str(20+t),np.zeros(3));self.assertEqual(trigger,t==2)
        self.assertEqual(a.models[1].revisions,1);self.assertLess(a.models[1].mass,len(a.models[1].samples))
        self.assertEqual(digest(a.models[2].snapshot()),unrelated)
    def test_isolated_error_does_not_trigger(self):
        m=LocalModel(self.c);o=dict(tick=0,body=[0.,0.],fatigue=.2,material=.2)
        for t in range(8): m.observe(o,o,str(t),np.zeros(3),True)
        self.assertFalse(m.observe(o,dict(o,body=[.8,0]),'large',np.zeros(3),True)[1])
        self.assertFalse(m.observe(o,o,'quiet',np.zeros(3),True)[1]);self.assertEqual(m.streak,0)
    def test_snapshot_exact_continuation(self):
        w=World(17,'local_swap',self.c);a=Agent(17,self.c,'BEU',5)
        for _ in range(70):
            o=w.observation();ac,d=a.decide(o);a.observe(o,ac,w.step(ac),str(o['tick']),d['pred'][ac])
        wc=World.restore(w.snapshot(),self.c);ac=Agent.restore(a.snapshot(),self.c)
        for _ in range(90):
            for world,agent in [(w,a),(wc,ac)]:
                o=world.observation();act,d=agent.decide(o);agent.observe(o,act,world.step(act),str(o['tick']),d['pred'][act])
        self.assertEqual(digest(w.snapshot()),digest(wc.snapshot()));self.assertEqual(digest(a.snapshot()),digest(ac.snapshot()))
    def test_capacity_and_weighted_normal_equations(self):
        w=World(1,'stable',self.c);a=Agent(1,self.c,'BEU',5)
        for t in range(250):
            o=w.observation();act=t%5;after=w.step(act);a.observe(o,act,after,str(t))
        for m in a.models:
            self.assertEqual(len(m.samples),24)
            x=np.array([s['x'] for s in m.samples]);y=np.array([s['y'] for s in m.samples]);weights=np.array([s['w'] for s in m.samples])
            np.testing.assert_allclose((x.T*weights)@x,m.gram,atol=1e-10)
            np.testing.assert_allclose((m.gram+.002*np.eye(6))@m.coef,m.rhs,atol=1e-10)
    def test_binary_lossless(self):
        arr=np.zeros(3,dtype=DTYPE);arr['after'][0]=[math.pi,.2,.3]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.npy.gz';write_events(p,arr);np.testing.assert_array_equal(arr,read_events(p))
    def test_B0_legacy_policy_parity(self):
        p=Path(__file__).parent/'references/previous_frozen_core.py';spec=importlib.util.spec_from_file_location('legacy',p)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        c=dict(self.c,change_tick=100000);w=World(60091,'stable',c);wo=old.World(60091,c)
        a=Agent(60091,c,'B0',5);ao=old.Agent(60091,c)
        for t in range(600):
            o=w.observation();act,d=a.decide(o);acto,do=ao.decide(wo.observation())
            self.assertEqual(act,acto);np.testing.assert_allclose(d['pred'][act],do['candidates'][act]['predicted_delta'],atol=1e-10)
            after=w.step(act);aftero=wo.step(acto);self.assertEqual(after,aftero)
            a.observe(o,act,after,str(t),d['pred'][act]);ao.observe(o,acto,aftero,str(t))

if __name__=='__main__': unittest.main(verbosity=2)
