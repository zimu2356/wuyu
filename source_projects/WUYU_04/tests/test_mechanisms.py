import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu04 import latent as l
from wuyu04.diagnostic import Replay,input_events,Decoder


class Mechanisms(unittest.TestCase):
    def test_replay_contains_only_actual_preceding_events(self):
        m=Replay(2,24,5,capacity=4,burn=3)
        for t in range(4):m.add(np.full((4,2,17),t+1.),np.zeros((2,24)),np.full(2,t%2))
        np.testing.assert_array_equal(m.past[0,3,:,0,0],[1,2,3])
        self.assertEqual(m.x[0,3,0,0],4)
        self.assertEqual(m.length[0,0],0)

    def test_gating_does_not_update_other_module(self):
        a=Decoder([1,2],'gated',.03)
        before=a.nets[1].w.copy();ctx=np.zeros(2,int);x=input_events(np.array([0,1]),ctx)
        p,h=a.predict(x,ctx);a.update(x,ctx,h,np.array([1,0]))
        np.testing.assert_array_equal(before,a.nets[1].w)
        self.assertEqual(a.nets[1].updates.sum(),0)

    def test_unrequested_probe_has_no_label(self):
        w=l.World([1,2]);w.begin()
        labels=w.probe(np.array([2,3]),np.array([True,False]))
        self.assertIn(labels[0],(0,1));self.assertEqual(labels[1],-1)
        with self.assertRaises(RuntimeError):w.probe(np.array([2,3]),np.array([True,False]))

    def test_feedback_is_sufficient_only_after_answer(self):
        w=l.World([1,2]);w.begin();target=w.target.copy();answer=1-target
        reward=w.finish(answer)
        np.testing.assert_array_equal(np.where(reward>0,answer,1-answer),target)

    def test_frozen_bank_and_learnable_novel_slot(self):
        a=l.Agent(2,'passive');a.w[:,:2]=.8;a.freeze_apprenticeship();before=a.w.copy()
        a.observe(np.array([1,2]),np.array([1,0]))
        np.testing.assert_array_equal(before[:,:2],a.w[:,:2])
        self.assertFalse(np.array_equal(before[:,2],a.w[:,2]))

    def test_complete_resume_contains_environment_rng(self):
        w=l.World([91,92]);a=l.Agent(2,'active')
        for _ in range(3712):l.step(w,a)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.npz';l.save(path,w,a);v,b=l.load(path)
        for _ in range(128):
            one=l.step(w,a);two=l.step(v,b)
            for key in one:np.testing.assert_array_equal(one[key],two[key])
        for one,two in ((w,v),(a,b)):
            for key,value in vars(one).items():
                if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(two,key))
        self.assertEqual(w.event_rng.bit_generator.state,v.event_rng.bit_generator.state)
        self.assertEqual(w.probe_rng.bit_generator.state,v.probe_rng.bit_generator.state)


if __name__=='__main__':unittest.main()
