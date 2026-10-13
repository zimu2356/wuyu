"""Seed-level estimates. Branches and time steps are never independent samples."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import random
import statistics

from run import csv_write,write_json


def read_csv(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))


def percentile(xs,p):
    ordered=sorted(xs);i=(len(ordered)-1)*p;lo=int(i);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(i-lo)


def estimate(values,config):
    values=list(values);n=len(values);rng=random.Random(config['bootstrap_seed'])
    samples=[sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(config['bootstrap_resamples'])]
    return dict(n=n,mean=statistics.mean(values),sd=statistics.stdev(values) if n>1 else 0,
                lower95=percentile(samples,.025),upper95=percentile(samples,.975),
                positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),negative=sum(v<0 for v in values),
                min=min(values),max=max(values))


def selective_effect(rows):
    j={(r['intervention'],int(r['task'])):float(r['body_loss']) for r in rows}
    return ((j['erase_axis0',0]-j['erase_axis1',0])+(j['erase_axis1',1]-j['erase_axis0',1]))/2


def analyze(output):
    output=Path(output);manifest=json.loads((output/'run_manifest.json').read_text());c=manifest['config']
    if manifest['status']!='complete':raise ValueError('Incomplete cohort')
    branches=read_csv(output/'branches.csv');probes=read_csv(output/'prediction_probes.csv');life=read_csv(output/'life_windows.csv')
    seeds=list(range(manifest['seeds']['start'],manifest['seeds']['start']+manifest['seeds']['count']))
    roles={}
    with gzip.open(output/'checkpoints.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            roles[r['seed']]={a:r['world']['tools'][a]['axis'] for a in range(1,5)}
    by_branch={(s,t):[] for s in seeds for t in c['checkpoints']}
    for r in branches:by_branch[int(r['seed']),int(r['checkpoint'])].append(r)
    by_prediction={(s,t):{} for s in seeds for t in c['checkpoints']}
    for r in probes:
        by_prediction[int(r['seed']),int(r['checkpoint'])][r['intervention'],int(r['task']),r['predictor'],int(r['action'])]=float(r['body_mse'])
    individual=[];checkpoints={}
    for t in c['checkpoints']:
        all_effects=[];matched=[];capacity_full=[];restoration_errors=[];prediction_selectivity=[];conditional_improvements=[]
        for s in seeds:
            rows=by_branch[s,t]
            if len(rows)!=8:raise AssertionError('Missing or duplicate branch')
            effect=selective_effect(rows);all_effects.append(effect)
            index={(r['intervention'],int(r['task'])):r for r in rows}
            full=bool(int(rows[0]['full_tool_buffers']));capacity_full.append(full)
            equally_removed=int(index['erase_axis0',0]['removed_samples'])==int(index['erase_axis1',0]['removed_samples']);matched.append(equally_removed)
            restoration_errors.append(max(abs(float(index['retain',k]['body_loss'])-float(index['restore_axis0',k]['body_loss'])) for k in (0,1)))
            p=by_prediction[s,t];increments=[]
            for axis in (0,1):
                erased=[a for a in range(1,5) if roles[s][a]==axis];other=[a for a in range(1,5) if roles[s][a]!=axis]
                def inc(actions):return statistics.mean(p[f'erase_axis{axis}',k,'conditional',a]-p['retain',k,'conditional',a] for k in (0,1) for a in actions)
                increments.append(inc(erased)-inc(other))
            ps=statistics.mean(increments);prediction_selectivity.append(ps)
            ci=statistics.mean(p['retain',k,'action_mean',a]-p['retain',k,'conditional',a] for k in (0,1) for a in range(1,5))
            conditional_improvements.append(ci)
            individual.append(dict(seed=s,checkpoint=t,selective_effect=effect,full_tool_buffers=int(full),equal_removed_samples=int(equally_removed),
                                   prediction_selectivity=ps,conditional_mse_improvement=ci))
        outcomes={}
        for iv in c['interventions']:
            for task in (0,1):
                rr=[r for r in branches if int(r['checkpoint'])==t and r['intervention']==iv and int(r['task'])==task]
                outcomes[f'{iv}/task{task}']={field:statistics.mean(float(r[field]) for r in rr) for field in ('body_loss','mean_fatigue','waits','first_action_regret','uniform_action_regret','predictor_body_mse','removed_samples')}
        paired={}
        for label,iv,task_index in [('related','erase_axis',True),('irrelevant','erase_axis',False)]:
            values=[]
            for s in seeds:
                idx={(r['intervention'],int(r['task'])):float(r['body_loss']) for r in by_branch[s,t]}
                values.append(statistics.mean(idx[f'erase_axis{k if task_index else 1-k}',k]-idx['retain',k] for k in (0,1)))
            paired[label+'_minus_retain']=estimate(values,c)
        checkpoints[str(t)]=dict(selective_effect=estimate(all_effects,c),full_buffer_count=sum(capacity_full),full_buffer_fraction=statistics.mean(capacity_full),
                                 equal_removed_count=sum(matched),equal_removed_fraction=statistics.mean(matched),
                                 max_restoration_loss_difference=max(restoration_errors),prediction_selectivity=estimate(prediction_selectivity,c),
                                 conditional_mse_improvement=estimate(conditional_improvements,c),absolute_outcomes=outcomes,**paired)
    adaptation={}
    for start,end in [(1600,1700),(1700,2000),(2000,2400)]:
        index={(int(r['seed']),r['branch']):r for r in life if int(r['start'])==start and int(r['end'])==end}
        adaptation[f'{start}-{end}']={}
        for field in ('mean_body_loss','prediction_mse','mean_fatigue','waits','saturated'):
            online=[float(index[s,'online'][field]) for s in seeds];frozen=[float(index[s,'frozen'][field]) for s in seeds]
            adaptation[f'{start}-{end}'][field]=dict(online_mean=statistics.mean(online),frozen_mean=statistics.mean(frozen),
                                                     frozen_minus_online=estimate([f-o for f,o in zip(frozen,online)],c))
    all_life={}
    for key in sorted({(r['branch'],int(r['start']),int(r['end'])) for r in life}):
        rr=[r for r in life if (r['branch'],int(r['start']),int(r['end']))==key]
        all_life[f'{key[0]}/{key[1]}-{key[2]}']={field:statistics.mean(float(r[field]) for r in rr) for field in ('mean_body_loss','mean_fatigue','prediction_mse','waits','saturated')}
    result=dict(cohort=manifest['cohort'],n=len(seeds),checkpoints=checkpoints,adaptation=adaptation,life_windows=all_life,
                primary_checkpoint=c.get('selected_primary_checkpoint'),interval_method='5000 percentile bootstrap resamples of independent seeds; all secondary intervals descriptive, no multiplicity correction',
                prediction_selectivity_definition='For each erased functional axis, MSE increase over retain on its two tool channels minus the increase on the other two channels, averaged across the two matched tasks and axes. WAIT excluded. This tests target-model error specificity, not task-value specificity.',
                loss_definition='J = arithmetic mean over all 12 post-action body squared norms; no selection of terminal/best step. Evaluation uses noiseless clipped reference, not the expectation under clipped noise.')
    write_json(output/'summary.json',result);csv_write(output/'individual_effects.csv',individual)
    print(json.dumps(dict(cohort=result['cohort'],n=result['n'],checkpoints={t:{k:r[k] for k in ('full_buffer_count','equal_removed_count','selective_effect')} for t,r in checkpoints.items()}),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();analyze(a.output)
