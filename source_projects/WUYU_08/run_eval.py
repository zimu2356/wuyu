import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib,copy,time
from pathlib import Path
import numpy as np
from wuyu08.network import Network
from wuyu08.state import read_tree,write_tree
from wuyu08.experiment import common_history,replay_history,probe,life,restore,checkpoint,neural_step
from wuyu08.world import World
from wuyu08.individual import Individual
ROOT=Path(__file__).resolve().parent
CELLS=['flat_regular','flat_varied','graph_regular','graph_varied']
CONDITIONS=['independent','dependency','long_chain','changed']

def source_hashes():
    paths=list((ROOT/'wuyu08').glob('*.py'))+[ROOT/k for k in ['CONFIG.json','PROTOCOL.md','EVALUATION_DETAILS.md','run_train.py','run_eval.py','analyze.py']]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

if __name__=='__main__':
    cfg=json.loads((ROOT/'CONFIG.json').read_text());out=ROOT/'results';seeds=cfg['formal_start']+np.arange(cfg['formal_worlds']);start=time.perf_counter()
    selections={}
    for cell in CELLS:
        for i in range(len(cfg['initializations'])):
            q=json.loads((out/f'training_{cell}_{i}.json').read_text());selections[f'{cell}_{i}']=q['selected']
    hashes=source_hashes();frozen=out/'frozen_before_formal.json'
    if frozen.exists():assert json.loads(frozen.read_text())['hashes']==hashes
    else:frozen.write_text(json.dumps(dict(hashes=hashes,selections=selections,config=cfg),indent=2))
    manifest=[]
    for i,init in enumerate(cfg['initializations']):
        hp=out/f'common_history_{i}.npz'
        if not hp.exists():write_tree(hp,common_history(seeds,init))
        history=read_tree(hp)
        for cell in CELLS:
            name=f'{cell}_{i}';net=Network.from_state(read_tree(out/selections[name]['file']));w,a=replay_history(net,history,init)
            for c in CONDITIONS:
                dest=out/f'probe_{name}_{c}.npz'
                if not dest.exists():write_tree(dest,probe(w,a,c))
                manifest.append(dict(file=dest.name,cell=cell,initialization=i,condition=c,intervention='none',history='common'))
            for c in ['independent','dependency']:
                dest=out/f'probe_{name}_{c}_body_clamp.npz'
                if not dest.exists():write_tree(dest,probe(w,a,c,'body_clamp'))
                manifest.append(dict(file=dest.name,cell=cell,initialization=i,condition=c,intervention='body_clamp',history='common'))
            if cell=='graph_varied':
                for c in CONDITIONS:
                    modes=['planner']+(['one_round','zero_round','cut_edges'] if c in ['dependency','long_chain'] else [])+(['hidden_reset'] if c=='dependency' else [])
                    for mode in modes:
                        dest=out/f'probe_{name}_{c}_{mode}.npz'
                        if not dest.exists():write_tree(dest,probe(w,a,c,mode))
                        manifest.append(dict(file=dest.name,cell=cell,initialization=i,condition=c,intervention=mode,history='common'))
            if not (out/f'life_{name}.npz').exists():life(net,seeds,init,out,name)
            ow,oa=restore(out/f'checkpoint_{name}_384.npz')
            for c in ['independent','dependency']:
                dest=out/f'own_probe_{name}_{c}.npz'
                if not dest.exists():write_tree(dest,probe(ow,oa,c))
                manifest.append(dict(file=dest.name,cell=cell,initialization=i,condition=c,intervention='none',history='own'))
            print(json.dumps(dict(evaluated=name,seconds=time.perf_counter()-start)),flush=True)
        # Online associative learning ablation of the proposed graph/varied cell.
        cell='graph_varied';name=f'{cell}_frozen_{i}';net=Network.from_state(read_tree(out/selections[f'{cell}_{i}']['file']))
        if not (out/f'life_{name}.npz').exists():
            w=World(seeds,1536,shocks=True);a=Individual(net,seeds,init)
            for m in a.memories:m.mode='frozen'
            rows=[]
            while w.t<w.total:
                r=neural_step(w,a);z=r['out'];rows.append(dict(obs=r['obs'],action=r['action'],alive=r['alive'],body=z['body'],product=z['product'],position=z['position'],stage=z['stage'],inventory=z['inventory'],shock=z['shock'],goal=r['gp'].argmax(1),target=r['tp'].argmax(1)))
            from wuyu08.state import save_raw
            raw={k:np.stack([r[k] for r in rows]) for k in rows[0]};save_raw(out/f'life_{name}.npz',seeds=seeds,**raw)
            met=dict(survival=raw['alive'].mean(0),body_area=(raw['body'].sum(2)*raw['alive']).mean(0),deliveries=(raw['product']<4).sum(0))
            (out/f'life_metrics_{name}.json').write_text(json.dumps({k:v.tolist() for k,v in met.items()},indent=2))
    (out/'probe_manifest.json').write_text(json.dumps(manifest,indent=2));assert hashes==source_hashes()
    print('FORMAL_COMPLETE',flush=True)
