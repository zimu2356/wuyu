"""Independent vectorized physics and sufficient-statistics audit, plus every-file SHA."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, hashlib, json, math, random
from pathlib import Path
import numpy as np
from storage import sha,read_events,read_gz_json,write_json
from core import digest

def audit(folder,c):
    folder=Path(folder);manifest=json.loads((folder/'MANIFEST.json').read_text());rows=json.loads((folder/'summaries.json').read_text())
    count=0;max_resid=0.;models=0;causal_pairs=0
    for name,expected in manifest.items(): assert sha(folder/name)==expected,(name,'SHA mismatch')
    for r in rows:
        seed=r['seed'];case=r['case'];method=r['method'];stem=f'{seed}_{case}_{method}'
        e=read_events(folder/(stem+'.npy.gz'));meta=read_gz_json(folder/(stem+'.json.gz'));n=c['life_ticks'];assert len(e)==n
        assert e.dtype.names==('action','purpose','revision','streak','base_choice','after','prediction','score','residual','age','n','mass','ess','version')
        assert np.all(np.isfinite(e['after']));assert np.max(np.abs(e['after'][:,:2]))<=1 and np.min(e['after'][:,2])>=0 and np.max(e['after'][:,2])<=1
        assert np.max(e['action'])<r['n_actions'];assert meta['summary']==r
        rng=random.Random(seed);roles=[(0,-1),(0,1),(1,-1),(1,1)];rng.shuffle(roles)
        tools=[(axis,sign,rng.uniform(.14,.20),rng.uniform(.06,.10)) for axis,sign in roles]
        previous=np.vstack(([.55,.45,.2],e['after'][:-1]));t=np.arange(n);mat=np.asarray(c['training_materials'])[(t//c['material_period']+seed%2)%2]
        delta=np.zeros((n,3));delta[:,:2]=np.asarray(c['body_drift'])-c['body_relaxation']*previous[:,:2]
        active=e['action']>0;delta[:,2]=np.where(active,np.minimum(c['work_fatigue'],1-previous[:,2]),-np.minimum(c['wait_recovery'],previous[:,2]))
        for a,(axis,sign,base,slope) in enumerate(tools,1):
            use=e['action']==a;polarity=np.ones(n)
            if case in ('local_swap','recurrence') and axis==r['changed_axis']:
                sw=t>=r['tau']
                if case=='recurrence': sw &=t<r['tau']+c['recurrence_gap']
                polarity[sw]=-1
            delta[use,axis]+=polarity[use]*sign*(base+slope*mat[use])*(1-.7*previous[use,2])
        if case=='opportunity':
            use=e['action']==5;strength=np.where(t>=r['tau'],.4,.005)
            delta[use,:2]-=strength[use,None]*previous[use,:2]
        noise=[]
        for tick in range(n):
            h=hashlib.blake2b(f'{seed}:{tick}:body'.encode(),digest_size=16).digest()
            u=(int.from_bytes(h[:8],'big')+.5)/18446744073709551616;v=(int.from_bytes(h[8:],'big')+.5)/18446744073709551616
            radius=math.sqrt(-2*math.log(max(u,1e-18)));noise.append([radius*math.cos(2*math.pi*v),radius*math.sin(2*math.pi*v)])
        sd=np.full(n,c['noise_sd'])
        if case=='noise_pulse': sd[(t>=r['tau'])&(t<r['tau']+c['noise_duration'])]=c['noisy_sd']
        expected=previous+delta;expected[:,:2]+=sd[:,None]*np.array(noise)
        expected[:,:2]=np.clip(expected[:,:2],-1,1);expected[:,2]=np.clip(expected[:,2],0,1)
        if case=='visible_shock':
            expected[r['tau'],:2]=np.clip(expected[r['tau'],:2]+[.45,-.35],-1,1);expected[r['tau'],2]=.95
        np.testing.assert_allclose(expected,e['after'],rtol=0,atol=1e-14)
        residual=np.linalg.norm(e['after'][:,:2]-previous[:,:2]-e['prediction'][:,:2],axis=1)
        np.testing.assert_allclose(residual,e['residual'],rtol=0,atol=1e-14)
        loss=np.sum(e['after'][:,:2]**2,axis=1);assert abs(float(loss[r['tau']:r['tau']+512].mean())-r['post_loss'])<1e-14
        visits=np.zeros(r['n_actions'],dtype=int);last=np.full(r['n_actions'],-1)
        for tick,row in enumerate(e):
            a=int(row['action']);assert row['version']==visits[a];assert row['n']==min(visits[a],24);assert row['age']==tick-last[a]
            visits[a]+=1;last[a]=tick
        assert visits.tolist()==meta['final_agent']['visits'] and last.tolist()==meta['final_agent']['last_use']
        for action,m in enumerate(meta['final_agent']['models']):
            samples=m['samples'];assert len(samples)<=24
            if not samples: continue
            for s in samples:
                tick=s['tick'];assert e['action'][tick]==action;assert s['event_id']==f'{seed}:{case}:{method}:{tick}'
                np.testing.assert_allclose(s['y'],e['after'][tick]-previous[tick],rtol=0,atol=1e-14)
            x=np.array([s['x'] for s in samples]);y=np.array([s['y'] for s in samples]);weights=np.array([s['w'] for s in samples])
            gram=(x.T*weights)@x;rhs=(x.T*weights)@y
            err=max(float(np.max(np.abs(gram-m['gram']))),float(np.max(np.abs(rhs-m['rhs']))))
            assert err<1e-9;max_resid=max(max_resid,err)
            np.testing.assert_allclose((gram+c['ridge']*np.eye(6))@m['coef'],rhs,rtol=0,atol=1e-9);models+=1
        assert digest(dict(world=meta['final_world'],agent=meta['final_agent']))==r['final_state_sha']
        causal=meta['causal']
        if causal:
            assert digest(causal['post_agents']['normal'])==digest(causal['post_agents']['restore'])
            assert digest(causal['branches']['normal'])==digest(causal['branches']['restore'])==digest(causal['branches']['block_unrelated_sham'])
            ev=causal['event'];assert int(e['action'][ev['tick']])==ev['action'];assert e['revision'][ev['tick']]==1
            causal_pairs+=1
        count+=n
    result=dict(passed=True,lives=len(rows),events=count,files=len(manifest),final_models=models,max_sufficient_statistics_residual=max_resid,causal_restore_pairs=causal_pairs,
                boundaries='Independent physics expressions and final weighted normal equations; same declared noise recipe. All policies separately replayed in reproduction.')
    write_json(folder/'AUDIT.json',result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);args=ap.parse_args()
    print(json.dumps(audit(args.data,json.loads(Path(args.config).read_text())),indent=2))
