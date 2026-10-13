import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wuyu06.network import Network,H,D,G
from wuyu06.world import World
from wuyu06.agent import Agent,Stream
from wuyu06.experiment import transition
from wuyu06.state import save,load,write_tree,read_tree


class CoreTests(unittest.TestCase):
    def test_full_recurrent_gradient(self):
        n=3;tmax=4;rng=np.random.default_rng(300);net=Network(40)
        xs=rng.normal(0,.1,(tmax,n,D));cands=rng.normal(0,.2,(n,8,G));valid=np.ones((n,8),bool)
        fast=rng.normal(0,.01,(n,H,21));fb=np.zeros((n,21));ug=rng.random((tmax,n));ua=rng.random((tmax,n))
        adv=rng.normal(0,.5,(tmax,n));ret=rng.normal(0,.3,(tmax,n));targets=rng.normal(0,.1,(tmax,n,3))
        def records():
            h=np.zeros((n,H));out=[]
            for t in range(tmax):
                q=net.forward(xs[t],h,fast,fb,cands,valid,np.zeros((n,G)),np.ones(n,bool),ug[t],ua[t]);h=q['h']
                q.update(alive=np.ones(n,bool),target=targets[t]);out.append(q)
            return out
        def loss():
            out=records();v=0
            for t,q in enumerate(out):
                ix=np.arange(n);v+=(-np.log(q['ap'][ix,q['action']])*adv[t]).sum()
                v+=(-np.log(q['gp'][ix,q['gi']])*adv[t]).sum()
                v+=.012*(q['ap']*np.log(q['ap'])).sum()+.012*(q['gp']*np.log(q['gp'])).sum()
                v+=.25*((q['value']-ret[t])**2).sum()
                error=q['pred'].reshape(n,7,3)[ix,q['action']]-targets[t]
                v+=.15*(error**2).mean(1).sum()
            return v/(n*tmax)
        grad,_=net.gradients(records(),ret,adv)
        for key,ix in [('W',(3,5)),('R',(2,8)),('b',(7,)),('M',(4,5)),('mb',(9,)),('A',(5,2)),('GA',(3,1)),('PA',(2,4)),('Q',(2,3)),('V',(2,)),('vb',(0,))]:
            before=net.p[key][ix];eps=1e-5;net.p[key][ix]=before+eps;p=loss();net.p[key][ix]=before-eps;m=loss();net.p[key][ix]=before
            self.assertAlmostEqual((p-m)/(2*eps),grad[key][ix],places=6,msg=key)

    def test_rng_batch_invariance(self):
        one=Stream([1],22);many=Stream([2,1,5],22)
        for _ in range(10):self.assertEqual(one.draw()[0],many.draw()[1])

    def test_prediction_precedes_feedback(self):
        w=World([2]);a=Agent(Network(1),[2]);p=copy.deepcopy(a.net.p);fast=a.fast.copy();a.act(w.observation())
        for k,v in p.items():np.testing.assert_array_equal(v,a.net.p[k])
        np.testing.assert_array_equal(fast,a.fast)

    def test_budget_stops_replay_and_world_action(self):
        w=World([2]);a=Agent(Network(1),[2]);transition(w,a);w.body[0,0]=.005
        r=a.act(w.observation());r['action'][:]=4;oldpos=w.pos.copy();out=w.step(r['action']);a.feedback(r,w.observation(),out['reward'])
        self.assertFalse(out['allowed'][0]);self.assertFalse(r['replay_applied'][0]);np.testing.assert_array_equal(oldpos,w.pos)

    def test_goal_provenance_and_replay_past(self):
        w=World([7,8,9],256);a=Agent(Network(4),[7,8,9]);positive={i:{} for i in range(3)}
        for t in range(256):
            r=transition(w,a)
            for i in range(3):
                if np.any(r['goal'][i]):
                    origin=r['goal_origin'][i];self.assertLess(origin,t);np.testing.assert_allclose(r['goal'][i],positive[i][origin])
                if r['replay_applied'][i]:self.assertLess(r['replay_origin'][i],t)
                if r['positive_outcome'][i]:positive[i][t]=np.r_[np.eye(4)[w.pos[i]],np.maximum(0,r['target'][i,:2])]

    def test_no_hidden_world_inputs(self):
        w=World([7]);a=Agent(Network(3),[7]);obs=w.observation();x=a.inputs(obs)
        self.assertEqual(obs.shape,(1,9));self.assertEqual(x.shape,(1,D))
        self.assertEqual(set(Agent.feedback.__code__.co_varnames[:4]),{'self','r','next_obs','reward'})

    def test_frozen_fast_still_has_history(self):
        w=World([3,4],128);a=Agent(Network(3),[3,4],'frozen_fast')
        for _ in range(100):transition(w,a)
        self.assertTrue((a.fast==0).all());self.assertTrue((a.fast_b==0).all());self.assertGreater(a.count.sum(),0);self.assertGreater(np.abs(a.h).sum(),0)

    def test_checkpoint_exact_future(self):
        w=World([21,22],256);a=Agent(Network(5),[21,22])
        for _ in range(30):transition(w,a)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.npz';save(p,w,a);v,b=load(p)
            for _ in range(32):
                x=transition(w,a);y=transition(v,b)
                for k in ('action','ap','pred','goal','reward','target'):np.testing.assert_array_equal(x[k],y[k])
            save(Path(td)/'a.npz',w,a);save(Path(td)/'b.npz',v,b)
            with np.load(Path(td)/'a.npz',allow_pickle=False) as az,np.load(Path(td)/'b.npz',allow_pickle=False) as bz:
                self.assertEqual(set(az.files),set(bz.files))
                for k in az.files:np.testing.assert_array_equal(az[k],bz[k])


if __name__=='__main__':unittest.main()
