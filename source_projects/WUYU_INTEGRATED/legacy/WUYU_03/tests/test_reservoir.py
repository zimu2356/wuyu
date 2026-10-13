import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu03 import reservoir,core


class ReservoirTests(unittest.TestCase):
    def test_capacity_and_past_only(self):
        m=reservoir.ReservoirMemory(2,capacity=8,seed=7)
        for t in range(200):
            x=np.full((4,2,17),t,dtype=float)
            m.add(x,np.zeros((2,24)),np.zeros(2),np.ones(2,bool))
            self.assertLessEqual(m.x.max(),t)
        np.testing.assert_array_equal(m.count,[200,200])
        self.assertEqual(m.x.shape[1],8)
        self.assertLess(m.x.min(),100)

    def test_restore_reservoir_rng(self):
        seeds=np.array([17,18]);a=core.Agent(seeds,'replay');a.memory=reservoir.ReservoirMemory(2,seed=22)
        w=core.World(seeds);ev=core.events(seeds,300)
        for t in range(270):core.step(w,a,ev,t,0,forced=0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.npz';reservoir.save(p,w,a);w2,a2=reservoir.load(p)
        for t in range(270,300):
            one=core.step(w,a,ev,t,1);two=core.step(w2,a2,ev,t,1)
            for k in one:np.testing.assert_array_equal(one[k],two[k])
        for k in ['x','h','y','count']:
            np.testing.assert_array_equal(getattr(a.memory,k),getattr(a2.memory,k))


if __name__=='__main__':unittest.main()
