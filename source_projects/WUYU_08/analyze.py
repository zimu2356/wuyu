import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from wuyu08.state import read_tree
ROOT=Path(__file__).resolve().parent

def stats(a):
    a=np.asarray(a,float);by=a.mean(axis=tuple(range(1,a.ndim)));world=a.mean(0)
    if world.ndim>1:world=world.mean(axis=tuple(range(1,world.ndim)))
    rng=np.random.default_rng(88008);boots=world[rng.integers(0,len(world),(4000,len(world)))].mean(1)
    return dict(mean=float(a.mean()),ci=np.quantile(boots,[.025,.975]).tolist(),by_initialization=by.tolist())

def analyze():
    out=ROOT/'results';cfg=json.loads((ROOT/'CONFIG.json').read_text());manifest=json.loads((out/'probe_manifest.json').read_text());groups={}
    for item in manifest:
        k='/'.join(item[x] for x in ['cell','history','condition','intervention']);q=read_tree(out/item['file'])
        groups.setdefault(k,{})[item['initialization']]={m:np.array(v,dtype=float).reshape(cfg['formal_worlds'],2) for m,v in q['metrics'].items()}
    arrays={};summ={}
    for key,rows in groups.items():
        arrays[key]={m:np.stack([rows[i][m] for i in sorted(rows)]) for m in rows[0]}
        arrays[key]['pair_success']=np.repeat(arrays[key]['first_success'].all(2)[:,:,None],2,axis=2).astype(float)
        summ[key]={m:stats(v) for m,v in arrays[key].items()}
    lives={};life_arrays={}
    for cell in ['flat_regular','flat_varied','graph_regular','graph_varied','graph_varied_frozen']:
        vals=[json.loads((out/f'life_metrics_{cell}_{i}.json').read_text()) for i in range(len(cfg['initializations']))]
        life_arrays[cell]={k:np.array([x[k] for x in vals]) for k in vals[0]};lives[cell]={k:stats(v) for k,v in life_arrays[cell].items()}
    effects={}
    for condition in ['independent','dependency','long_chain','changed']:
        suffix=f'/common/{condition}/none'
        for label,hi,lo in [('coverage_flat','flat_varied','flat_regular'),('coverage_graph','graph_varied','graph_regular'),('architecture_regular','graph_regular','flat_regular'),('architecture_varied','graph_varied','flat_varied')]:
            effects[f'{label}/{condition}']=stats(arrays[hi+suffix]['first_success']-arrays[lo+suffix]['first_success'])
        effects[f'interaction/{condition}']=stats((arrays['graph_varied'+suffix]['first_success']-arrays['graph_regular'+suffix]['first_success'])-(arrays['flat_varied'+suffix]['first_success']-arrays['flat_regular'+suffix]['first_success']))
    ablations={}
    for key in arrays:
        cell,history,c,mode=key.split('/')
        if mode not in ('none','planner'):
            ablations[key]=stats(arrays[f'{cell}/{history}/{c}/none']['first_success']-arrays[key]['first_success'])
    result=dict(probes=summ,lives=lives,effects=effects,ablations=ablations,online_learning_effect=stats(life_arrays['graph_varied']['survival']-life_arrays['graph_varied_frozen']['survival']),
                interval_scope='Paired world bootstrap conditional on five fixed trained initializations; exploratory multiple contrasts not familywise corrected.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    s=analyze();print(json.dumps(dict(lives=s['lives'],effects=s['effects']),indent=2))
