import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
from wuyu06.network import Network
from wuyu06.state import read_tree
from wuyu06.experiment import rollout

ROOT=Path(__file__).resolve().parent
MODES=['full','plain_rnn','rule','frozen_fast','no_replay','no_goal','no_forecast']

def summarize(data,config):
    rng=np.random.default_rng(config['bootstrap_seed']);ix=rng.integers(0,32,(config['bootstrap_samples'],32));result={'methods':{},'contrasts':{},'gates':{}}
    def est(x):
        # x = training initialization x matched world. Resample worlds jointly;
        # uncertainty is conditional on these three trained networks.
        avg=x.mean(0);lo,hi=np.quantile(avg[ix].mean(1),[.025,.975]);return {'mean':float(avg.mean()),'ci95':[float(lo),float(hi)],'by_initialization':x.mean(1).tolist()}
    for mode in MODES:
        result['methods'][mode]={k:est(np.stack([np.asarray(data[f'{mode}_{i}'][k]) for i in range(3)])) for k in data[f'{mode}_0']}
    for mode in MODES[1:]:
        result['contrasts'][mode]={k:est(np.stack([np.asarray(data[f'full_{i}'][k])-np.asarray(data[f'{mode}_{i}'][k]) for i in range(3)])) for k in data['full_0']}
    for gate,mode in [('G1','plain_rnn'),('G2','frozen_fast'),('G3_performance','no_goal'),('G4_performance','no_forecast'),('G5','no_replay')]:
        result['gates'][gate]=result['contrasts'][mode]['survival']['ci95'][0]>0
    result['interval_note']='World-paired bootstrap, averaged over 3 fixed training initializations; not a population interval over training seeds.'
    return result


if __name__=='__main__':
    config=json.loads((ROOT/'TRAINING_CONFIG.json').read_text());out=ROOT/'results';data={};start=time.perf_counter()
    for i in range(3):
        models={}
        for mode in ['full','plain_rnn']:
            selected=json.loads((out/f'training_{mode}_{i}.json').read_text())['selected']['checkpoint'];models[mode]=Network.from_state(read_tree(out/selected))
        for mode in MODES:
            name=f'{mode}_{i}';net=models['plain_rnn' if mode=='plain_rnn' else 'full']
            m,_,_,_,seconds=rollout(net,np.arange(980000,980032),config['formal_length'],mode,out,name,checkpoint=mode=='full')
            data[name]={k:v.tolist() for k,v in m.items()}
            print(json.dumps({'formal':name,'seconds':seconds,'survival':float(m['survival'].mean()),'body':float(m['body_area'].mean())}),flush=True)
            (out/'formal_metrics.json').write_text(json.dumps(data,indent=2))
    summary=summarize(data,config);summary['seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('FORMAL_COMPLETE',flush=True)
