"""Exploratory report diagnostics; not used for selection or agent feedback."""
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core import Agent,World,predicted_observation
from analyze import read_csv,estimate
from run import csv_write,write_json


def diagnose():
    output=ROOT/'outputs/holdout';c=json.loads((ROOT/'formal_protocol.json').read_text())
    out=ROOT/'reporting/derived';out.mkdir(exist_ok=True)
    rows=[];actual=[];groups={str(t):[] for t in c['checkpoints']}
    with gzip.open(output/'checkpoints.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);cp=r['checkpoint']
            if cp not in c['checkpoints']:continue
            w=World.restore(r['world'],c);a=Agent.restore(r['agent'],c);obs=w.observation();errors=[];perseed=[]
            for action in range(1,5):
                samples=r['agent']['models'][action]['samples'];times=[int(s['event_id'].rsplit(':',1)[1]) for s in samples]
                stale=sum(t<c['change_tick'] for t in times) if cp>c['change_tick'] else 0
                pred=predicted_observation(obs,a.models[action].predict(obs));target=predicted_observation(obs,w.mean_delta(obs,action))
                error=sum((pred['body'][i]-target['body'][i])**2 for i in range(2))/2;errors.append(error)
                rr=dict(seed=r['seed'],checkpoint=cp,action=action,axis=r['world']['tools'][action]['axis'],
                        n=len(samples),pre_change_samples=stale,oldest_age=cp-min(times) if times else '',
                        newest_age=cp-max(times) if times else '',actual_state_body_mse=error)
                rows.append(rr);perseed.append(rr)
            groups[str(cp)].append(dict(seed=r['seed'],stale_fraction=sum(x['pre_change_samples'] for x in perseed)/sum(x['n'] for x in perseed),
                                        never_revisited_after_change=sum(x['newest_age']>cp-c['change_tick'] for x in perseed) if cp>c['change_tick'] else 0,
                                        tool_prediction_mse=statistics.mean(errors)))
            action,decision=a.decide(obs);pred=predicted_observation(obs,a.models[action].predict(obs));target=predicted_observation(obs,w.mean_delta(obs,action))
            actual.append(dict(seed=r['seed'],checkpoint=cp,chosen_action=action,chosen_mse=sum((pred['body'][i]-target['body'][i])**2 for i in range(2))/2,
                               all_tool_mse=statistics.mean(errors)))
    result={'status':'exploratory','checkpoints':{}}
    for cp,rr in groups.items():
        aa=[r for r in actual if r['checkpoint']==int(cp)]
        result['checkpoints'][cp]=dict(mean_pre_change_sample_fraction=statistics.mean(r['stale_fraction'] for r in rr),
                                       total_tool_models_not_revisited=sum(r['never_revisited_after_change'] for r in rr),
                                       mean_tool_mse_actual_state=statistics.mean(r['all_tool_mse'] for r in aa),
                                       mean_chosen_mse_actual_state=statistics.mean(r['chosen_mse'] for r in aa),
                                       chooses_wait=sum(r['chosen_action']==0 for r in aa))
    life=read_csv(output/'life_windows.csv');index={(int(r['seed']),r['branch'],int(r['start']),int(r['end'])):r for r in life}
    paired=[]
    for seed in range(c['holdout_seeds']['start'],c['holdout_seeds']['start']+c['holdout_seeds']['count']):
        before=float(index[seed,'online',1200,1600]['mean_body_loss']);after=float(index[seed,'online',2000,2400]['mean_body_loss'])
        paired.append(dict(seed=seed,prechange_loss=before,late_loss=after,difference=after-before,ratio=after/before))
    result['late_minus_prechange']=estimate([r['difference'] for r in paired],c)
    result['prechange_mean']=statistics.mean(r['prechange_loss'] for r in paired);result['late_mean']=statistics.mean(r['late_loss'] for r in paired)
    result['ratio_of_means']=result['late_mean']/result['prechange_mean']
    csv_write(out/'memory_age_and_prediction.csv',rows);csv_write(out/'checkpoint_actual_state_predictions.csv',actual);csv_write(out/'recovery_by_seed.csv',paired)
    write_json(out/'exploratory_diagnostics.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':diagnose()
