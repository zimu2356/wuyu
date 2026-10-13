import copy,unittest,tempfile
from pathlib import Path
import numpy as np
from wuyu08.network import Network
from wuyu08.features import F,G,N,encode
from wuyu08.individual import Individual
from wuyu08.memory import Agent
from wuyu08.world import World
from wuyu08.experiment import neural_step,checkpoint,restore,common_history,replay_history,probe

class CoreTests(unittest.TestCase):
    def test_recurrent_and_iterative_gradients(self):
        rng=np.random.default_rng(8)
        for kind in ['flat','graph']:
            net=Network(9,kind);xs=[dict(node=rng.normal(size=(2,N,F)),glob=rng.normal(size=(2,G)),edge=rng.uniform(0,.2,(2,N,N)),local=np.eye(6)[[0,3]]) for _ in range(3)]
            h0=net.zero(2)+.01;y=[rng.integers(0,6,(2,3)) for _ in xs]
            def forward():
                h=h0.copy();rs=[]
                for x in xs:
                    r=net.forward(x,h);r['alive']=np.ones(2,bool);h=r['h'];rs.append(r)
                return rs
            gr=net.gradients(forward(),y)
            for k,v in net.p.items():
                for ix in rng.choice(v.size,min(4,v.size),replace=False):
                    old=v.flat[ix];eps=1e-5;v.flat[ix]=old+eps;lp=net.loss(forward(),y);v.flat[ix]=old-eps;lm=net.loss(forward(),y);v.flat[ix]=old
                    num=(lp-lm)/(2*eps);self.assertAlmostEqual(num,gr[k].flat[ix],delta=2e-6,msg=f'{kind}:{k}:{ix}')
    def test_no_teacher_at_deployment(self):
        from unittest.mock import patch
        w=World([29000001,29000002]);a=Individual(Network(3,'graph'),w.seeds)
        with patch.object(Agent,'act',side_effect=AssertionError('teacher')),patch.object(Agent,'plan',side_effect=AssertionError('plan')),patch.object(Agent,'_route',side_effect=AssertionError('route')):
            for _ in range(5):neural_step(w,a)
    def test_hidden_codes_not_observed(self):
        w=World([2,3]);o=w.observation();w.codes=(w.codes+1)%3;w.products=w.products[:,::-1]
        np.testing.assert_array_equal(o,w.observation())
    def test_no_feedback_no_model_learning(self):
        w=World([3]);a=Individual(Network(3),w.seeds);before=copy.deepcopy(a.memories[0].state());a.act(w.observation())
        for k in ['transition','motor','effect']:np.testing.assert_array_equal(before[k],getattr(a.memories[0],k))
    def test_dead_absorbing_and_shock(self):
        w=World([4],training_varied=True);w.body[:]=[.9,.9];w.t=47;z=w.step([6]);self.assertGreater(z['shock'].sum(),0)
        w.body[:]=0;w.alive[:]=False;b=w.body.copy()
        for _ in range(60):w.step([5])
        np.testing.assert_array_equal(w.body,b);self.assertFalse(w.alive.any())
    def test_prerequisite_consumption(self):
        w=World([4]);w.regime_override=1;site=int(np.flatnonzero(w.products[0]<2)[0]);w.pos[:]=site;w.stage[0,site]=2;req=w.requirements()[0,site]
        w.step([5]);self.assertEqual(w.last[0],4);w.inventory[0,req]=1;w.step([5]);self.assertLess(w.last[0],2);self.assertEqual(w.inventory[0,req],0)
    def test_resume(self):
        for kind in ['flat','graph']:
            w=World([1,2],training_varied=True);a=Individual(Network(7,kind),w.seeds)
            for _ in range(13):neural_step(w,a)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'s.npz';checkpoint(path,w,a);v,b=restore(path)
                for _ in range(20):
                    x=neural_step(w,a);y=neural_step(v,b)
                    np.testing.assert_array_equal(x['action'],y['action']);np.testing.assert_array_equal(w.body,v.body);np.testing.assert_array_equal(a.h,b.h)
    def test_probe_freeze_and_no_writeback(self):
        hist=common_history(np.array([29000002,29000003]),801,total=64);w,a=replay_history(Network(5,'graph'),hist,801)
        before=copy.deepcopy(a.state());pos=w.pos.copy();q=probe(w,a,'dependency',steps=5)
        np.testing.assert_array_equal(a.h,before['h']);np.testing.assert_array_equal(w.pos,pos)
        for s,t in zip(q['initial']['memories'],q['final']['memories']):
            for k in ['transition','motor','effect']:np.testing.assert_array_equal(s[k],t[k])
    def test_common_feature_information(self):
        w=World([5]);a=Individual(Network(5),w.seeds);x=encode(w.observation(),a.prev,a.memories)
        self.assertEqual(x['node'].shape,(1,6,F));self.assertEqual(x['glob'].shape,(1,G));self.assertFalse(x['edge'].any())
        self.assertLess(abs(Network(1,'flat').count-Network(1,'graph').count)/Network(1,'flat').count,.01)
if __name__=='__main__':unittest.main()
