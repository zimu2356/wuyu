"""Independent scalar-rule physics ledger, separate from World.step().

Seed-to-world generation is reused via saved initial states or the constructor;
transition/observation and metric recomputation below are independently expressed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from wuyu08.state import read_tree
from wuyu08.world import World
from wuyu08.experiment import restore,neural_step
ROOT=Path(__file__).resolve().parent

def equal(a,b,path='root'):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:equal(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)

def laws(s):
    phase=min(3,s['t']//max(1,s['total']//4)) if s['regime_override'] is None else s['regime_override']
    if s['req_override'] is not None:req=s['req_override'].copy()
    elif phase==0:req=np.full_like(s['base_req'],-1)
    else:
        req=s['base_req'].copy()
        if phase==3:
            for j in range(s['n']):
                ii=np.flatnonzero(s['products'][j]<2)[:2];req[j,ii]=1-req[j,ii]
    codes=s['codes'].copy()
    if phase==3:
        for j in range(s['n']):codes[j,s['changed'][j]]=s['replacement'][j]
    return phase,req,codes

def observed(s):
    _,req,_=laws(s);o=np.zeros((s['n'],21))
    for j in range(s['n']):
        p=s['pos'][j];o[j,p]=1;o[j,6+s['stage'][j,p]]=1;o[j,9+req[j,p]+1]=1
        o[j,12:14]=s['inventory'][j];o[j,14:16]=s['body'][j];o[j,16+s['last'][j]]=1
    return o

def independent_step(s,acts):
    phase,req,codes=laws(s);product=np.full(s['n'],4);shock=np.zeros_like(s['body']);alive=s['alive'].copy()
    for j,action in enumerate(acts):
        if not alive[j]:continue
        cost=(.003 if action<2 else (.0018 if action==6 else .004))*(.55 if phase>=2 else 1.)
        available=s['body'][j,0]>=cost and s['body'][j,1]>.001
        s['body'][j,0]=max(0,s['body'][j,0]-cost)
        s['body'][j,1]=max(0,s['body'][j,1]-(.0018 if phase>=2 else .0007))
        site=s['pos'][j];stage=s['stage'][j,site]
        if available:
            if action==0:s['pos'][j]=(site-1)%6
            elif action==1:s['pos'][j]=(site+1)%6
            elif 2<=action<=4 and stage<2:s['stage'][j,site]=stage+1 if codes[j,site,stage]==action-2 else 0
            elif action==5 and stage==2:
                need=req[j,site]
                if need<0 or s['inventory'][j,need]>0:
                    if need>=0:s['inventory'][j,need]-=1
                    kind=s['products'][j,site];product[j]=kind
                    if kind<2:s['body'][j,kind]=min(1,s['body'][j,kind]+.28)
                    else:s['inventory'][j,kind-2]=1
                    s['stage'][j,site]=0
        s['alive'][j]=available and s['body'][j].min()>.001
        if s['alive'][j]:
            if s['training_varied'] and (s['t']+1)%48==0:
                d=int((s['seeds'][j]+s['t']//48)%2);desired=.25+(s['seeds'][j]*17+s['t'])%19/100
                shock[j,d]=max(0,s['body'][j,d]-desired)
            elif s['shocks'] and s['t']+1 in (480,640,960,1120,1280,1440):
                d=int((s['seeds'][j]+s['t']//160)%2);shock[j,d]=min(.35,s['body'][j,d])
        s['body'][j]-=shock[j];s['alive'][j]&=s['body'][j].min()>.001
    s['last']=product;s['t']+=1
    return dict(alive=alive,body=s['body'].copy(),product=product,position=s['pos'].copy(),stage=s['stage'].copy(),inventory=s['inventory'].copy(),shock=shock)

def replay(start,raw):
    s=copy.deepcopy(start)
    for t,acts in enumerate(raw['action']):
        np.testing.assert_allclose(observed(s),raw['obs'][t],atol=1e-12,rtol=0)
        out=independent_step(s,acts)
        for k in out:np.testing.assert_allclose(out[k],raw[k][t],atol=1e-12,rtol=0,err_msg=f'{k}@{t}')
    return len(raw['action'])*s['n']

def recalc_probe(q):
    raw=q['raw'];need=q['need'];m=q['metrics'];n=len(need);steps=len(raw['action'])
    for j in range(n):
        first=-1;ft=steps;wt=steps;wanted=False
        for t,x in enumerate(raw['product'][:,j]):
            if first<0 and x<2:first=int(x);ft=t+1
            if not wanted and x==need[j]:wanted=True;wt=t+1
        expected=dict(first_success=first==need[j],any_success=wanted,no_body=first<0,first_product=first,
                      first_cost=ft if first==need[j] else steps,any_cost=wt,body_gain=raw['body'][-1,j,need[j]]-q['start']['body'][j,need[j]],
                      final_min=raw['body'][-1,j].min(),survival=raw['alive'][:,j].mean(),
                      goal_need=bool(q['start']['products'][j,raw['goal'][0,j]]==need[j] and q['start']['alive'][j]))
        for k,v in expected.items():assert abs(float(m[k][j])-float(v))<1e-12,(k,j)

if __name__=='__main__':
    out=ROOT/'results';cfg=json.loads((ROOT/'CONFIG.json').read_text());result={};slots=0;life_files=sorted(out.glob('life_*.npz'))
    for p in life_files:
        with np.load(p,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        start=World(raw['seeds'],1536,shocks=True).state();slots+=replay(start,raw)
        saved=json.loads((out/(p.stem.replace('life_','life_metrics_',1)+'.json')).read_text())
        recomputed=dict(survival=raw['alive'].mean(0),body_area=(raw['body'].sum(2)*raw['alive']).mean(0),deliveries=(raw['product']<4).sum(0))
        for k,v in recomputed.items():np.testing.assert_allclose(saved[k],v,atol=1e-12,rtol=0)
        print('life audited',p.name,flush=True)
    result.update(life_files=len(life_files),life_event_slots=slots)
    slots=0;manifest=json.loads((out/'probe_manifest.json').read_text());facts={}
    for item in manifest:
        q=read_tree(out/item['file']);slots+=replay(q['start'],q['raw']);recalc_probe(q)
        for x,y in zip(q['initial']['memories'],q['final']['memories']):
            for k in ['transition','motor','effect']:np.testing.assert_array_equal(x[k],y[k])
        for k,v in q['initial']['network']['p'].items():np.testing.assert_array_equal(v,q['final']['network']['p'][k])
        if item['history']=='common' and item['intervention']=='none':
            key=(item['initialization'],item['condition']);s=[{k:m[k] for k in ['transition','effect','motor','requirement','effect_count']} for m in q['initial']['memories']]
            if key in facts:equal(facts[key],s)
            else:facts[key]=s
    result.update(probe_files=len(manifest),probe_event_slots=slots,shared_initial_facts_equal=True,frozen_probe_parameters=True,probe_metrics_recomputed=True)
    slots=0
    for i,init in enumerate(cfg['initializations']):
        h=read_tree(out/f'common_history_{i}.npz');s=World(h['world']['seeds'],1536).state()
        for t,act in enumerate(h['action']):
            np.testing.assert_allclose(observed(s),h['obs'][t],atol=1e-12,rtol=0);independent_step(s,act);slots+=len(act)
        np.testing.assert_allclose(observed(s),h['last_obs'],atol=1e-12,rtol=0)
        equal(s,h['world'])
    result['common_history_event_slots']=slots
    resumes=[]
    for cell in ['flat_regular','flat_varied','graph_regular','graph_varied']:
        w,a=restore(out/f'checkpoint_{cell}_0_384.npz')
        while w.t<768:neural_step(w,a)
        tw,ta=restore(out/f'checkpoint_{cell}_0_768.npz');equal(w.state(),tw.state());equal(a.state(),ta.state());resumes.append(cell)
    result['actual_trained_resume_384_to_768']=resumes
    for cell in ['flat_regular','flat_varied','graph_regular','graph_varied']:
        for i in range(5):
            birth=read_tree(out/f'checkpoint_{cell}_{i}_0.npz');end=read_tree(out/f'checkpoint_{cell}_{i}_1536.npz')
            equal(birth['individual']['network'],end['individual']['network'])
    result['all_shared_networks_fixed_during_formal_life']=True
    # Runtime teacher exclusion on an actually trained graph checkpoint.
    from unittest.mock import patch
    from wuyu08.memory import Agent
    w,a=restore(out/'checkpoint_graph_varied_0_384.npz')
    with patch.object(Agent,'act',side_effect=AssertionError('teacher')),patch.object(Agent,'plan',side_effect=AssertionError('plan')),patch.object(Agent,'_route',side_effect=AssertionError('route')):
        for _ in range(32):neural_step(w,a)
    result['trained_runtime_no_teacher']=True
    frozen=json.loads((out/'frozen_before_formal.json').read_text())
    for path,h in frozen['hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    train=json.loads((out/'frozen_before_training.json').read_text())
    for path,h in train.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    result['training_and_formal_sources_unchanged']=True
    for p in out.glob('*.npz'):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
    result['npz_crc_files']=len(list(out.glob('*.npz')))
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
