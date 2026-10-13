"""Independent physics ledger + real checkpoint replay + stored-file checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib,zipfile,copy
from pathlib import Path
import numpy as np
from wuyu07.world import World
from wuyu07.agent import Agent
from wuyu07.student import Student
from wuyu07.experiment import restore,step,metrics
from wuyu07.state import read_tree
ROOT=Path(__file__).resolve().parent


def same(x,y,path='root'):
    if isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y,err_msg=path)
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:same(x[k],y[k],path+'/'+str(k))
    elif isinstance(x,list):
        assert len(x)==len(y),path
        for i,(a,b) in enumerate(zip(x,y)):same(a,b,path+'/'+str(i))
    else:assert x==y,(path,x,y)


def ledger(raw):
    # Only reuse the seeded world constructor, never its transition or observation methods.
    w=World(raw['seeds']);n=w.n;rows=np.arange(n)
    pos=w.pos.copy();body=w.body.copy();stage=w.stage.copy();inv=w.inventory.copy();alive=w.alive.copy();last=w.last.copy()
    checked=0;origins=0
    for t in range(len(raw['action'])):
        phase=min(3,t//384);req=np.full((n,6),-1,int) if phase==0 else w.base_req.copy()
        if phase==3:
            for i in range(n):
                ii=np.flatnonzero(w.products[i]<2)[:2];req[i,ii]=1-req[i,ii]
        obs=np.concatenate([np.eye(6)[pos],np.eye(3)[stage[rows,pos]],np.eye(3)[req[rows,pos]+1],inv,body,np.eye(5)[last]],1)
        np.testing.assert_array_equal(obs.astype(np.float32),raw['obs'][t])
        np.testing.assert_array_equal(alive,raw['alive'][t]);before=body.copy();was=alive.copy();last[:]=4
        codes=w.codes.copy()
        if phase==3:codes[rows,w.changed]=w.replacement
        for i,a in enumerate(raw['action'][t]):
            if not was[i]:continue
            price=[.003,.003,.004,.004,.004,.004,.0018][a]*(.55 if phase>=2 else 1.)
            allowed=before[i,0]>=price and before[i,1]>.001
            body[i,0]=max(0,before[i,0]-price);body[i,1]=max(0,before[i,1]-(.0018 if phase>=2 else .0007))
            if allowed:
                if a<2:pos[i]=(pos[i]+(-1 if a==0 else 1))%6
                elif a<5:
                    s=stage[i,pos[i]]
                    if s<2:stage[i,pos[i]]=s+1 if codes[i,pos[i],s]==a-2 else 0
                elif a==5 and stage[i,pos[i]]==2:
                    need=req[i,pos[i]]
                    if need<0 or inv[i,need]>0:
                        if need>=0:inv[i,need]-=1
                        product=w.products[i,pos[i]];last[i]=product;stage[i,pos[i]]=0
                        if product<2:body[i,product]=min(1,body[i,product]+.28)
                        else:inv[i,product-2]=1
            alive[i]=allowed and min(body[i])>.001
        reward=np.where(was,(body-before).sum(1)+.006*alive-(was&~alive).astype(float),0.)
        for key,value in [('body',body),('position',pos),('stage',stage),('inventory',inv),('product',last),('reward',reward)]:
            np.testing.assert_array_equal(value,raw[key][t],err_msg=f'{key}@{t}')
        for i in np.flatnonzero(raw['source'][t]==2):
            origin=int(raw['skill_origin'][t,i]);assert 0<=origin<t
            site=int(obs[i,:6].argmax());s=int(obs[i,6:9].argmax())
            assert 2<=raw['action'][origin,i]<=4
            assert raw['obs'][origin,i,:6].argmax()==site
            assert raw['stage'][origin,i,site]==s+1
            origins+=1
        checked+=n
    return checked,origins


if __name__=='__main__':
    out=ROOT/'results';summary=dict(event_slots=0,skill_origins=0,npz_files=0,checkpoint_replays=[],probe_steps=0,probe_traces=0)
    for p in sorted(out.glob('raw_*.npz')):
        with np.load(p,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        count,origins=ledger(raw);summary['event_slots']+=count;summary['skill_origins']+=origins
        saved=json.loads((out/('metrics_'+p.stem[4:]+'.json')).read_text())['metrics']
        for k,v in metrics(raw).items():np.testing.assert_allclose(v,saved[k],rtol=0,atol=0)
        print('ledger',p.name,count,flush=True)
    for family in ['full','neural_student']:
        for i in range(3):
            w,a=restore(out/f'checkpoint_{family}_{i}_384.npz')
            with np.load(out/f'raw_{family}_{i}.npz',allow_pickle=False) as z:actions=z['action'];bodies=z['body']
            for t in range(384,768):
                r=step(w,a);np.testing.assert_array_equal(r['action'],actions[t]);np.testing.assert_array_equal(w.body,bodies[t])
            ww,aa=restore(out/f'checkpoint_{family}_{i}_768.npz');same(w.state(),ww.state())
            same(a.state(),aa.state()) if isinstance(a,Student) else same([x.state() for x in a],[x.state() for x in aa])
            summary['checkpoint_replays'].append(dict(family=family,initialization=i,events=384,worlds=32,all_state_equal=True))
    for i,initialization in enumerate([701,702,703]):
        w,a=restore(out/f'checkpoint_frozen_{i}_1536.npz')
        for seed,x in zip(w.seeds,a):
            born=Agent(seed,initialization,'frozen')
            for k in ['transition','motor','effect']:same(getattr(x,k),getattr(born,k))
        w,a=restore(out/f'checkpoint_neural_student_{i}_384.npz')
        def fail(*args,**kwargs):raise AssertionError('Runtime planning/teacher queried')
        for m in a.memories:m.plan=fail;m._route=fail;m.act=fail
        for _ in range(16):step(w,a)
        s0=read_tree(out/f'checkpoint_neural_student_{i}_0.npz')['agents']['network']
        sf=read_tree(out/f'checkpoint_neural_student_{i}_1536.npz')['agents']['network'];same(s0,sf)
    for trace in read_tree(out/'probes_raw.npz')['traces']:
        if 'actions' not in trace:continue
        w=World([trace['seed']],trace['total']);w.t=trace['start_t'];w.regime_override=trace['phase'];w.pos[:]=trace['position']
        w.body[:]=trace['body'];w.inventory[:]=trace['inventory'];w.alive[:]=trace['alive'];outcomes=[]
        for j,action in enumerate(trace['actions']):
            w.step([action]);assert int(w.last[0])==trace['products'][j];np.testing.assert_array_equal(w.body[0],trace['bodies'][j]);outcomes.append((int(w.pos[0]),int(w.last[0])))
        if 'target_site' in trace:
            hits=[j+1 for j,(site,product) in enumerate(outcomes) if product<4 and site==trace['target_site']]
            success=int(bool(hits));cost=hits[0] if hits else trace['budget']
        else:
            hits=[(j+1,product) for j,(_,product) in enumerate(outcomes) if product<2]
            success=int(bool(hits) and hits[0][1]==trace['target_need']);cost=hits[0][0] if success else trace['budget']
        assert success==trace['success'] and cost==trace['cost'],trace['label']
        summary['probe_steps']+=len(outcomes);summary['probe_traces']+=1
    for p in sorted(out.glob('*.npz')):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None,p.name
        summary['npz_files']+=1
    frozen=json.loads((out/'frozen_before_formal.json').read_text())
    for name,digest in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    training=json.loads((out/'frozen_before_training.json').read_text())
    summary['changed_since_rnn_training']=[name for name,d in training.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=d]
    summary.update(formal_source_unchanged=True,neural_runtime_planner_absent=True,student_global_weights_frozen=True,physics_and_metrics_verified=True)
    (out/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
