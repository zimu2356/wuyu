import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu03.neural import RNN
from wuyu03.core import Agent,World,events,step,save_checkpoint,load_checkpoint
from wuyu03.control import Control


class Mechanisms(unittest.TestCase):
    def test_four_step_gradient(self):
        net=RNN([2],inputs=3,hidden=4)
        rng=np.random.default_rng(5)
        x=rng.normal(size=(4,1,3));h=rng.normal(size=(1,4))*.2;y=np.array([1.])
        g=net.gradients(x,h,y,np.ones(1))
        for name in ['w','r','b','v','c']:
            arr=getattr(net,name)
            for flat in np.unique(np.linspace(0,arr.size-1,min(8,arr.size)).astype(int)):
                ix=np.unravel_index(flat,arr.shape);orig=arr[ix];eps=1e-6
                arr[ix]=orig+eps;p=net.forward(x,h)[0];lp=-np.log(p[0])
                arr[ix]=orig-eps;p=net.forward(x,h)[0];lm=-np.log(p[0])
                arr[ix]=orig
                self.assertAlmostEqual((lp-lm)/(2*eps),g[name][ix],places=6)

    def test_environment_does_not_answer(self):
        w=World([1,2]);w.charge[:]=4
        cue,valid=w.acquire(np.array([1,2]),np.array([0,0]),False,np.array([0,1]),np.zeros((2,3)))
        np.testing.assert_array_equal(w.charge,[[3,4],[4,3]])
        np.testing.assert_array_equal(cue,np.array([0,1])^w.polarity)
        np.testing.assert_array_equal(w.feedback(np.array([1,0]),np.array([0,1]),valid),[-1,-1])

    def test_no_memory_cannot_read_earlier_cue(self):
        a=Agent([1,2],mode='no_memory');x=np.zeros((4,2,17));x[-1,:,14]=1
        p=a.net.forward(x,a.net.h)[0];x[1,:,9]=[1,-1];x[0,:,:]=12
        np.testing.assert_array_equal(a.net.forward(x,a.net.h)[0],p)

    def test_label_is_reconstructed_after_feedback(self):
        a=Agent([1,2]);x=np.zeros((4,2,17));h=a.net.h.copy()
        a.feedback(x,h,np.array([0,1]),np.array([-1.,-1.]),np.ones(2,bool),
                   np.zeros(2,int),np.zeros(2,int),np.zeros(2,int))
        np.testing.assert_array_equal(a.memory.y[:,0],[1,0])

    def test_model_only_observed_keys(self):
        c=Control(2,mode='dyna');c.update(np.array([0,2]),np.array([1,4]),np.array([.3,-.2]),np.array([3,1]))
        self.assertEqual(int(c.seen.sum()),2)
        np.testing.assert_array_equal(c.mr[[0,1],[1,14]],[.3,-.2])
        np.testing.assert_array_equal(c.mn[[0,1],[1,14]],[3,1])

    def test_restore_exact_online_future(self):
        seeds=np.array([41,42]);a=Agent(seeds,'replay');w=World(seeds);ev=events(seeds,40)
        for t in range(20):step(w,a,ev,t,t//10%2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';save_checkpoint(path,w,a);w2,a2=load_checkpoint(path)
        for t in range(20,40):
            one=step(w,a,ev,t,t//10%2);two=step(w2,a2,ev,t,t//10%2)
            for k in one:np.testing.assert_array_equal(one[k],two[k])
        for obj1,obj2 in [(w,w2),(a.net,a2.net),(a.ctrl,a2.ctrl),(a.memory,a2.memory)]:
            for k,v in vars(obj1).items():
                if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,getattr(obj2,k))


if __name__=='__main__':unittest.main()
