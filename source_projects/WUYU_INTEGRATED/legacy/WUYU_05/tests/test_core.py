import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wuyu05.core import Agent,World,Replay,VectorRNG,step,save,load,MAX_RATE


class Mechanisms(unittest.TestCase):
    def test_vector_rng_is_individual_and_batch_independent(self):
        single=VectorRNG([849001],10);batched=VectorRNG([849002,849001,849001],10)
        for _ in range(8):
            a=single.random()[0];b=batched.random()
            self.assertEqual(a,b[1]);self.assertEqual(a,b[2]);self.assertNotEqual(a,b[0])

    def test_vector_update_matches_original_scalar_update(self):
        a=Agent([849003,849004],'fixed',np.zeros((2,2)))
        old=copy.deepcopy(a.net);rng=np.random.default_rng(9)
        x=rng.normal(size=(4,2,17));h=rng.normal(size=(2,24));y=np.array([1.,0.]);valid=np.array([True,False])
        old.learn(x,h,y,valid,.03);a.learn(x,h,y,valid,np.full(2,.03))
        for name in ('w','r','b','v','c'):np.testing.assert_allclose(getattr(old,name),getattr(a.net,name),rtol=1e-14,atol=1e-14)

    def test_original_gradients_finite_difference(self):
        a=Agent([849005],'fixed',np.zeros((1,2)));rng=np.random.default_rng(3)
        x=rng.normal(size=(4,1,17))*.1;h=np.zeros((1,24));y=np.ones(1);valid=np.ones(1,bool)
        g=a.net.gradients(x,h,y,valid)
        for name,idx in [('w',(0,9,2)),('r',(0,3,4)),('b',(0,2)),('v',(0,5)),('c',(0,))]:
            v=getattr(a.net,name);original=v[idx];eps=1e-6
            v[idx]=original+eps;plus=-np.log(a.net.forward(x,h)[0][0])
            v[idx]=original-eps;minus=-np.log(a.net.forward(x,h)[0][0])
            v[idx]=original;self.assertAlmostEqual(g[name][idx],(plus-minus)/(2*eps),places=7)

    def test_replay_preserves_preceding_invalid_events_without_future(self):
        m=Replay([849006],capacity=256,burn=8)
        for t in range(20):
            x=np.full((4,1,17),t,dtype=float)
            m.add(x,np.array([t%2]),np.array([t%3!=0]))
        self.assertEqual(m.count[0],13)
        for j in range(13):
            event=int(m.x[0,j,0,0]);length=m.length[0,j]
            np.testing.assert_array_equal(m.past[0,j,:length,0,0],np.arange(max(0,event-8),event))
            self.assertEqual(length,min(event,8))

    def test_prediction_does_not_update_weights(self):
        a=Agent([849007],'learned',np.zeros((1,2,8)));w=World([849007])
        ctx=w.conditions()[0];before=w.charge.copy();cue,valid=w.acquire(np.zeros(1,int))
        weights={k:getattr(a.net,k).copy() for k in ('w','r','b','v','c')}
        a.net.observe(a.inputs(ctx,before,np.zeros(1,int),cue,valid))
        for k,v in weights.items():np.testing.assert_array_equal(v,getattr(a.net,k))

    def test_rates_bounded_and_statistics_use_past(self):
        a=Agent([849008],'learned',np.full((1,2,8),100.))
        f,e,re=a.features(np.array([.99]),np.array([0.]),np.ones(1,bool),np.array([.2]),np.array([1.]),np.ones(1,bool))
        self.assertEqual(f[0,2],0.);self.assertEqual(f[0,3],0.)
        rates=a.rates(f,e,re);self.assertTrue(np.all((rates>=0)&(rates<=MAX_RATE)))

    def test_frozen_keeps_weights_but_history_advances(self):
        a=Agent([849009],'frozen',np.zeros((1,2)));a.branch=True;w=World([849009],'change')
        weights={k:getattr(a.net,k).copy() for k in ('w','r','b','v','c')}
        for _ in range(12):step(w,a)
        for k,v in weights.items():np.testing.assert_array_equal(v,getattr(a.net,k))
        self.assertEqual(a.t,12);self.assertGreater(a.memory.count[0],0)

    def test_restore_reproduces_outputs_and_all_serialized_state(self):
        seeds=[849010,849011];a=Agent(seeds,'learned',np.zeros((2,2,8)));w=World(seeds,'noise')
        for _ in range(12):step(w,a)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mid.npz';save(path,w,a);v,b=load(path)
            for _ in range(16):
                first,second=step(w,a),step(v,b)
                for k in first:np.testing.assert_array_equal(first[k],second[k])
            p=Path(directory)/'a.npz';q=Path(directory)/'b.npz';save(p,w,a);save(q,v,b)
            with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
                self.assertEqual(x.files,y.files)
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])


if __name__=='__main__':unittest.main()
