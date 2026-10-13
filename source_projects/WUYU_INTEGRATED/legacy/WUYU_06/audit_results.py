"""Independent physical ledger, provenance and real checkpoint replay audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib
import numpy as np
from wuyu06.state import read_tree,load,save
from wuyu06.experiment import transition

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def check_physics(raw,world):
    n=len(world['seeds']);body=np.tile([.68,.78],(n,1));pos=world['seeds']%4;cool=np.zeros((n,4),int);alive=np.ones(n,bool);inspected=np.zeros(n,bool)
    rows=np.arange(n);costs=np.array([.022,.022,.017,.025,.027,.009,.012]);max_error=0
    # Vectorize provenance assertions without dropping any recorded event.
    tt,ii=np.nonzero(np.any(raw['goal']!=0,axis=2));origin=raw['goal_origin'][tt,ii]
    assert np.all((origin>=0)&(origin<tt))
    expected=np.concatenate([np.eye(4)[raw['position'][origin,ii]],np.maximum(0,raw['target'][origin,ii,:2])],axis=1)
    np.testing.assert_allclose(raw['goal'][tt,ii],expected,rtol=0,atol=1e-12)
    tt,ii=np.nonzero(raw['replay_origin']>=0);origin=raw['replay_origin'][tt,ii]
    assert np.all((origin<tt)&(raw['action'][tt,ii]==4))
    for t in range(len(raw['action'])):
        a=raw['action'][t];old=body.copy();oldpos=pos.copy();was=alive.copy();p=(t>=world['bounds']).sum(1)
        idx=np.where(p==0,0,np.where(p<3,1,2));mapping=world['maps'][rows,idx];kind=mapping[rows,pos];u=world['random'][t]
        reliability=(.55+.43*body[:,1])*np.select([p==2,p==3],[.68,.82],default=1.);reliability[inspected]=.99
        cue=np.where(u[:,0]<reliability,kind,(kind+1+(u[:,1]>.5))%3)
        assert np.array_equal(raw['cue'][t],cue);assert np.array_equal(raw['alive'][t],alive)
        spend=costs[a]*np.where(p==3,1.18,1.);permitted=alive & (body[:,0]>=spend)
        body[:,0]=np.clip(body[:,0]-spend*alive,0,None);body[:,1]=np.maximum(.02,body[:,1]-.0014*alive);cool=np.maximum(0,cool-1)
        moved=permitted & (a<2) & (u[:,2]<(.72+.28*old[:,1])*np.select([p==2,p==3],[.62,.82],default=1.))
        pos=(pos+np.where(moved,1-2*(a==0),0))%4;kind=mapping[rows,pos]
        collect=permitted & (a==2) & (cool[rows,pos]==0) & (kind<2)
        for resource,increase in [(0,.36),(1,.32)]:
            take=collect & (kind==resource);body[take,resource]=np.minimum(1,body[take,resource]+increase)
        cool[rows[collect],pos[collect]]=14;alive=was & permitted & (body[:,0]>.001);body[~was]=old[~was];inspected=permitted & (a==3)
        delta=body-old;reward=np.where(was,delta.sum(1)+.02*alive-1.*(was & ~alive),0.)
        target=np.column_stack([delta/.36,(pos!=oldpos).astype(float)])
        assert np.max(np.abs(raw['body'][t]-body))<1e-12;assert np.array_equal(raw['position'][t],pos)
        assert np.max(np.abs(raw['reward'][t]-reward))<1e-12;assert np.max(np.abs(raw['target'][t]-target))<1e-12
        max_error=max(max_error,float(np.max(np.abs(raw['body'][t]-body))))
        assert np.all(alive[raw['replay_origin'][t]>=0])
    return max_error,int(alive.sum())


if __name__=='__main__':
    audit={'physical_ledgers':{},'checkpoint_replays':[],'frozen_files':{},'original_w05_unchanged':True,'w05_reference_matches':[]}
    metrics=json.loads((OUT/'formal_metrics.json').read_text())
    for raw_path in sorted(OUT.glob('raw_*.npz')):
        name=raw_path.stem[4:]
        with np.load(raw_path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        state=read_tree(OUT/f'final_{name}.npz');error,finish=check_physics(raw,state['world'])
        survival=raw['alive'].mean(0);area=(raw['body'].sum(2)*raw['alive']).mean(0)
        np.testing.assert_array_equal(survival,np.asarray(metrics[name]['survival']));np.testing.assert_allclose(area,metrics[name]['body_area'],atol=1e-12,rtol=0)
        family='plain_rnn' if name.startswith('plain_rnn') else 'full';initialization=int(name[-1]);model=json.loads((OUT/f'training_{family}_{initialization}.json').read_text())['selected']['checkpoint']
        trained=read_tree(OUT/model)
        for k in trained['p']:np.testing.assert_array_equal(trained['p'][k],state['network']['p'][k])
        audit['physical_ledgers'][name]={'max_body_error':error,'finished_individuals':finish,'events_checked':int(np.prod(raw['alive'].shape)),'global_weights_unchanged':True}
    for i in range(3):
        with np.load(OUT/f'raw_full_{i}.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        w,a=load(OUT/f'checkpoint_full_{i}_128.npz')
        for t in range(128,512):
            r=transition(w,a)
            np.testing.assert_array_equal(r['action'],raw['action'][t]);np.testing.assert_array_equal(r['ap'],raw['probabilities'][t]);np.testing.assert_array_equal(r['goal'],raw['goal'][t])
            np.testing.assert_array_equal(w.body,raw['body'][t])
        tmp=OUT/f'audit_resumed_{i}.npz';save(tmp,w,a)
        with np.load(tmp,allow_pickle=False) as x,np.load(OUT/f'checkpoint_full_{i}_512.npz',allow_pickle=False) as y:
            assert set(x.files)==set(y.files)
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        tmp.unlink();audit['checkpoint_replays'].append({'initialization':i,'steps':384,'all_serialized_fields_equal':True})
    for name,digest in json.loads((OUT/'frozen_before_training.json').read_text()).items():
        equal=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest;audit['frozen_files'][name]=equal;assert equal,name
    for name,digest in json.loads((ROOT/'source_provenance.json').read_text()).items():
        path=ROOT.parent/name
        if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    d=np.load(OUT/'w05_diagnostic_raw.npz',allow_pickle=False)
    for origin in ['fixed','learned']:
        for branch in ['return','change','noise']:
            baseline=d[f'{origin}_{branch}_closed_FF__action']
            for tag in ['FF','LF','FL','LL']:np.testing.assert_array_equal(d[f'{origin}_{branch}_yoked_{tag}__action'],baseline)
    previous=ROOT.parent/'WUYU_05/results/raw_fixed.npz'
    if previous.exists():
        with np.load(previous,allow_pickle=False) as old:
            for branch in ['return','change','noise']:
                for key in ['action','reward','valid']:
                    np.testing.assert_array_equal(d[f'fixed_{branch}_closed_FF__{key}'],old[f'{branch}__{key}'][:512])
                audit['w05_reference_matches'].append(branch)
    audit['all_checks_passed']=True;(OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
