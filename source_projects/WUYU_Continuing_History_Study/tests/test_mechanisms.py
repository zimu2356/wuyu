import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Agent,World,LocalModel,canonical,digest,features,intervene,predicted_observation
from experiment import checkpoint_diagnostics,execute
from analyze import selective_effect

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'development_protocol.json').read_text())


class MechanismTests(unittest.TestCase):
    def trained(self,n=240):
        w=World(9103,CONFIG);a=Agent(9103,CONFIG)
        for _ in range(n):execute(w,a,9103,'unit',lambda *args:None)
        return w,a

    def test_wait_is_a_transition_and_has_recovery(self):
        w=World(9103,CONFIG);before=w.observation();after=w.step(0)
        self.assertEqual(after['tick'],1);self.assertNotEqual(after['body'],before['body'])
        self.assertAlmostEqual(after['fatigue'],.08)
        for _ in range(20):w.step(0)
        self.assertEqual(w.tick,21);self.assertEqual(w.fatigue,0)

    def test_state_changes_tool_causal_effect(self):
        w=World(9103,CONFIG);low=w.observation();low['fatigue']=0;high=dict(low,fatigue=.8)
        axis=w.tools[1]['axis'];base=CONFIG['body_drift'][axis]-CONFIG['body_relaxation']*low['body'][axis]
        self.assertAlmostEqual((w.mean_delta(high,1)[axis]-base)/(w.mean_delta(low,1)[axis]-base),.44)

    def test_noise_is_shared_despite_different_actions(self):
        a=World(9103,CONFIG);b=World(9103,CONFIG)
        for _ in range(60):
            self.assertEqual(a.noise(a.tick),b.noise(b.tick));a.step(1);b.step(0)
        self.assertNotEqual(a.body,b.body)

    def test_save_resume_exact_future(self):
        w,a=self.trained();saved=json.loads(canonical(dict(world=w.snapshot(),agent=a.snapshot())))
        w2=World.restore(saved['world'],CONFIG);a2=Agent.restore(saved['agent'],CONFIG)
        for _ in range(140):
            one=execute(w,a,9103,'unit',lambda *args:None);two=execute(w2,a2,9103,'unit',lambda *args:None)
            self.assertEqual(one,two)
        self.assertEqual(digest(a.snapshot()),digest(a2.snapshot()))

    def test_conditional_fit_predicts_held_between_materials(self):
        c=dict(CONFIG,model_window=100,ridge=1e-8);model=LocalModel(c);rng=random.Random(6108)
        def rule(o):return [.07+.02*o['body'][0]-.06*o['fatigue']+.08*o['material']-.05*o['fatigue']*o['material'],.002,.02]
        for i in range(100):
            obs=dict(tick=i,body=[rng.uniform(-.3,.3),rng.uniform(-.3,.3)],fatigue=rng.uniform(.1,.7),material=.2 if i%2 else .8)
            model.observe(obs,predicted_observation(obs,rule(obs)),str(i))
        for f in (.1,.7):
            o=dict(tick=0,body=[.1,-.1],fatigue=f,material=.5)
            self.assertLess(max(abs(x-y) for x,y in zip(model.predict(o),rule(o))),1e-6)
        self.assertNotEqual(model.predict(dict(o,fatigue=.1)),model.predict(dict(o,fatigue=.7)))
        self.assertEqual(model.predict(dict(o,fatigue=.1),'action_mean'),model.predict(dict(o,fatigue=.7),'action_mean'))

    def test_mean_and_conditional_share_history(self):
        _,a=self.trained();model=next(m for m in a.models if m.samples);before=digest(model.snapshot())
        obs=dict(tick=999,body=[.6,0],fatigue=.25,material=.5)
        expected=[sum(s['y'][i] for s in model.samples)/len(model.samples) for i in range(3)]
        for x,y in zip(model.predict(obs,'action_mean'),expected):self.assertAlmostEqual(x,y,12)
        model.predict(obs);self.assertEqual(before,digest(model.snapshot()))

    def test_agent_has_no_hidden_world_specification(self):
        a=Agent(9103,CONFIG)
        for key in ('change_tick','tools','checkpoints','probe_tasks','body_drift'):self.assertNotIn(key,a.config)
        self.assertFalse(hasattr(a,'world'))

    def test_erasure_is_selective_and_restore_exact(self):
        w,a=self.trained();state=a.snapshot()
        erased,log=intervene(state,'erase_axis0',w.tools,CONFIG)
        self.assertEqual(len(log['removed']),2)
        for act in range(5):
            if act and w.tools[act]['axis']==0:self.assertEqual(erased['models'][act],LocalModel(CONFIG).snapshot())
            else:self.assertEqual(erased['models'][act],state['models'][act])
        for key in ('visits','rng_state','learn'):self.assertEqual(erased[key],state[key])
        restored,_=intervene(state,'restore_axis0',w.tools,CONFIG)
        self.assertEqual(digest(restored),digest(state));self.assertEqual(a.snapshot(),state)

    def test_diagnostic_branches_are_isolated_and_restoration_matches(self):
        w,a=self.trained();before=digest(dict(w=w.snapshot(),a=a.snapshot()));events=[]
        rows,_=checkpoint_diagnostics(9103,w,a,CONFIG,lambda name,row:events.append(row) if name=='events' else None)
        self.assertEqual(before,digest(dict(w=w.snapshot(),a=a.snapshot())))
        for task in (0,1):
            def behavior(iv):return [{k:r[k] for k in ('before','action','decision','after')} for r in events if r['branch']==f'probe_{w.tick}_{iv}_{task}']
            self.assertEqual(behavior('retain'),behavior('restore_axis0'))
        self.assertEqual(len(rows),8)

    def test_frozen_model_keeps_parameters_but_counts_actions(self):
        w,a=self.trained();a.learn=False;models=[m.snapshot() for m in a.models];visits=sum(a.visits)
        for _ in range(50):execute(w,a,9103,'frozen',lambda *args:None)
        self.assertEqual([m.snapshot() for m in a.models],models);self.assertEqual(sum(a.visits),visits+50)

    def test_decision_prediction_precedes_feedback(self):
        w,a=self.trained();obs=w.observation();pred=[m.predict(obs) for m in a.models]
        row,_=execute(w,a,9103,'unit',lambda *args:None)
        for cand,p in zip(row['decision']['candidates'],pred):self.assertEqual(cand['predicted_delta'],p)
        self.assertEqual(row['decision']['visits'][row['action']]+1,a.visits[row['action']])

    def test_hidden_change_is_not_a_state_reset(self):
        w=World(9103,CONFIG);w.tick=1599;w.body=[.1,.1];w.fatigue=.3
        before=w.observation();w.step(0);after=w.observation()
        self.assertEqual(after['tick'],1600);self.assertNotEqual(after['body'],[.55,.45]);self.assertNotEqual(after['fatigue'],.2)
        a=dict(before,tick=1599);b=dict(before,tick=1600);axis=w.tools[1]['axis']
        background=CONFIG['body_drift'][axis]-CONFIG['body_relaxation']*before['body'][axis]
        self.assertAlmostEqual(w.mean_delta(a,1)[axis]-background,-(w.mean_delta(b,1)[axis]-background))

    def test_primary_interaction_hand_computed(self):
        values={('erase_axis0',0):.4,('erase_axis1',0):.1,('erase_axis1',1):.5,('erase_axis0',1):.2}
        rows=[dict(intervention=iv,task=task,body_loss=value) for (iv,task),value in values.items()]
        self.assertAlmostEqual(selective_effect(rows),.3)


if __name__=='__main__':unittest.main()
