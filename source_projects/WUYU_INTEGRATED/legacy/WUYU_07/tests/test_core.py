import copy, tempfile, unittest
from pathlib import Path
import numpy as np
from wuyu07.world import World,OBS
from wuyu07.agent import Agent,softmax
from wuyu07.rnn import Network
from wuyu07.experiment import step,checkpoint,restore
from wuyu07.student import Student,features,FEATURES


class Tests(unittest.TestCase):
    def test_observation_boundary(self):
        w=World([710001]);o=w.observation().copy()
        w.codes=(w.codes+1)%3;w.products=np.roll(w.products,1,axis=1)
        np.testing.assert_array_equal(o,w.observation());self.assertEqual(o.shape,(1,OBS))

    def test_no_learning_before_feedback(self):
        a=Agent(710001);w=World([710001]);before=[x.copy() for x in [a.transition,a.motor,a.effect]]
        a.act(w.observation()[0])
        for x,y in zip(before,[a.transition,a.motor,a.effect]):np.testing.assert_array_equal(x,y)

    def test_semantic_gradient(self):
        x=np.array([.2,-.3,.1]);target=2;g=softmax(x);g[target]-=1
        for i in range(3):
            plus=x.copy();minus=x.copy();plus[i]+=1e-5;minus[i]-=1e-5
            fd=(-np.log(softmax(plus)[target])+np.log(softmax(minus)[target]))/2e-5
            self.assertAlmostEqual(fd,g[i],places=8)

    def test_skill_comes_from_completed_past_actions(self):
        w=World([710002]);a=Agent(710002);site=int(w.pos[0]);codes=w.codes[0,site].copy()
        for stage in range(2):
            obs=w.observation()[0];r=a.act(obs);r['action']=int(codes[stage])+2
            out=w.step([r['action']]);a.feedback(r,w.observation()[0],out['reward'][0])
        self.assertTrue(a.valid[site]);self.assertEqual(a.skill_events[-1]['origins'],[0,1])
        self.assertTrue(np.all(a.skill_origin[site]<a.tick))
        self.assertEqual(a.skill_events[-1]['actions'],codes.tolist())

    def test_no_product_without_preparation_or_key(self):
        w=World([710003]);before=w.body.copy();w.step([5]);self.assertEqual(w.last[0],4)
        w.regime_override=1;site=int(np.flatnonzero(w.products[0]<2)[0]);w.pos[0]=site;w.stage[0,site]=2
        w.inventory[:]=0;w.step([5]);self.assertEqual(w.last[0],4);self.assertEqual(w.stage[0,site],2)

    def test_dead_is_absorbing(self):
        w=World([710004]);w.body[0,0]=.0001;w.step([2]);self.assertFalse(w.alive[0])
        before=[w.body.copy(),w.stage.copy(),w.inventory.copy(),w.pos.copy()]
        for _ in range(3):w.step([5])
        for x,y in zip(before,[w.body,w.stage,w.inventory,w.pos]):np.testing.assert_array_equal(x,y)

    def test_frozen_weights(self):
        a=Agent(710005,mode='frozen');w=World([710005]);p=[x.copy() for x in [a.transition,a.motor,a.effect]]
        for _ in range(80):step(w,[a])
        for x,y in zip(p,[a.transition,a.motor,a.effect]):np.testing.assert_array_equal(x,y)

    def test_restore_exact_future(self):
        w=World([710006,710007]);agents=[Agent(s) for s in w.seeds]
        for _ in range(73):step(w,agents)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.npz';checkpoint(p,w,agents);w2,a2=restore(p)
            for _ in range(60):
                r1=step(w,agents);r2=step(w2,a2)
                np.testing.assert_array_equal(r1['action'],r2['action']);np.testing.assert_array_equal(w.body,w2.body)
            for x,y in zip(agents,a2):
                for k in ['transition','motor','effect','skill_origin']:np.testing.assert_array_equal(getattr(x,k),getattr(y,k))
                self.assertEqual(x.memory,y.memory)

    def test_rnn_recurrent_gradient(self):
        net=Network(19,hidden=4);rng=np.random.default_rng(5);records=[];h=np.zeros((2,4))
        for _ in range(3):
            r=net.forward(rng.normal(0,.2,(2,30)),h,rng.random(2));h=r['h'];r['alive']=np.ones(2,bool);records.append(r)
        returns=rng.normal(size=(3,2));adv=rng.normal(size=(3,2));gr=net.gradients(records,returns,adv,entropy=0)
        def loss():
            h=np.zeros((2,4));value=0.
            for t,r in enumerate(records):
                rr=net.forward(r['x'],h,np.zeros(2));h=rr['h']
                value+=(-np.log(rr['ap'][np.arange(2),r['action']])*adv[t]+.25*(rr['value']-returns[t])**2).sum()/6
            return value
        for k in net.p:
            ix=tuple(0 for _ in net.p[k].shape);old=net.p[k][ix]
            net.p[k][ix]=old+1e-5;plus=loss();net.p[k][ix]=old-1e-5;minus=loss();net.p[k][ix]=old
            self.assertAlmostEqual((plus-minus)/2e-5,gr[k][ix],places=7,msg=k)

    def test_student_no_runtime_planner(self):
        w=World([710001]);a=Student(Network(7,32,FEATURES),w.seeds)
        def fail(*args,**kwargs):raise AssertionError('runtime teacher/planner called')
        a.memories[0].plan=fail;a.memories[0]._route=fail;a.memories[0].act=fail
        for _ in range(15):step(w,a)

    def test_student_restore_exact(self):
        w=World([710006,710007]);a=Student(Network(7,32,FEATURES),w.seeds)
        for _ in range(15):step(w,a)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.npz';checkpoint(p,w,a);w2,a2=restore(p)
            for _ in range(30):
                r=step(w,a);rr=step(w2,a2);np.testing.assert_array_equal(r['action'],rr['action'])
                np.testing.assert_array_equal(w.body,w2.body)
                np.testing.assert_array_equal(a.memories[0].motor,a2.memories[0].motor)

    def test_imitation_gradient(self):
        net=Network(8,4,FEATURES);rng=np.random.default_rng(7);records=[];h=np.zeros((2,4));labels=[]
        for _ in range(3):
            r=net.forward(rng.normal(0,.1,(2,FEATURES)),h,np.zeros(2));h=r['h'];r['alive']=np.ones(2,bool)
            records.append(r);labels.append(rng.integers(0,7,2))
        gr=net.imitation_gradients(records,labels)
        def loss():
            h=np.zeros((2,4));v=0.
            for r,y in zip(records,labels):
                rr=net.forward(r['x'],h,np.zeros(2));h=rr['h'];v-=np.log(rr['ap'][np.arange(2),y]).sum()/6
            return v
        for k in ['W','R','A','b','ab']:
            ix=tuple(0 for _ in net.p[k].shape);old=net.p[k][ix]
            net.p[k][ix]=old+1e-5;plus=loss();net.p[k][ix]=old-1e-5;minus=loss();net.p[k][ix]=old
            self.assertAlmostEqual((plus-minus)/2e-5,gr[k][ix],places=7,msg=k)

    def test_probe_frozen_and_no_main_writeback(self):
        from run_probes import operate
        from audit import same
        w=World([710008]);a=Agent(710008)
        for _ in range(80):step(w,[a])
        ws=copy.deepcopy(w.state());aa=copy.deepcopy(a.state())
        _,_,_,clone,_=operate(w,a,int(w.pos[0]),0)
        same(ws,w.state());same(aa,a.state())
        for k in ['transition','motor','effect']:np.testing.assert_array_equal(getattr(a,k),getattr(clone,k))

if __name__=='__main__':unittest.main()
