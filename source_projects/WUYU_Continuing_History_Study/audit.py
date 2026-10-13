"""Streaming record audit, independent normal-equation and summary checks.

Physics uses the public world implementation; this is a consistency audit, not
an independent second physics implementation. Unit tests supply analytic cases.
"""
import argparse
from collections import defaultdict,deque
from copy import deepcopy
import gzip
import itertools
import json
from pathlib import Path
import statistics

from core import Agent,World,LocalModel,digest,features,intervene,physical_loss,predicted_observation,evaluation_loss
from experiment import observation_probe
from run import sha,write_json
from analyze import read_csv


def grouped(path):
    with gzip.open(path,'rt') as f:
        for seed,rows in itertools.groupby((json.loads(line) for line in f),key=lambda r:r['seed']):yield seed,list(rows)


def close(a,b,tol=1e-9):
    if abs(a-b)>tol:raise AssertionError((a,b,abs(a-b)))


def verify_model(model):
    samples=model['samples'];gram=[[sum(s['x'][i]*s['x'][j] for s in samples) for j in range(6)] for i in range(6)]
    rhs=[[sum(s['x'][i]*s['y'][k] for s in samples) for k in range(3)] for i in range(6)]
    for i in range(6):
        for j in range(6):close(gram[i][j],model['gram'][i][j],1e-8)
        for k in range(3):
            close(rhs[i][k],model['rhs'][i][k],1e-8)
            residual=sum((gram[i][j]+(model['ridge'] if i==j else 0))*model['coefficients'][j][k] for j in range(6))
            close(residual,rhs[i][k],1e-8)
    for k in range(3):close(sum(s['y'][k] for s in samples),model['total_y'][k],1e-8)


def audit(output):
    output=Path(output);manifest=json.loads((output/'run_manifest.json').read_text());c=manifest['config']
    if manifest['status']!='complete':raise AssertionError('Incomplete run')
    for name,value in manifest['output_sha256'].items():
        if sha(output/name)!=value:raise AssertionError('Raw hash mismatch: '+name)
    streams=[grouped(output/(name+'.jsonl.gz')) for name in ('events','checkpoints','interventions','predictions','final_states')]
    branch_csv=read_csv(output/'branches.csv');window_csv=read_csv(output/'life_windows.csv');pred_csv=read_csv(output/'prediction_probes.csv')
    bindex={(int(r['seed']),int(r['checkpoint']),r['intervention'],int(r['task'])):r for r in branch_csv}
    pindex={(int(r['seed']),int(r['checkpoint']),r['intervention'],int(r['task']),r['predictor'],int(r['action'])):r for r in pred_csv}
    windex={(int(r['seed']),r['branch'],int(r['start']),int(r['end'])):r for r in window_csv}
    expected=list(range(manifest['seeds']['start'],manifest['seeds']['start']+manifest['seeds']['count']))
    counts=defaultdict(int);observed=[]
    for groups in itertools.zip_longest(*streams):
        if any(g is None for g in groups):raise AssertionError('Stream cohort differs')
        seed=groups[0][0];observed.append(seed)
        if any(g[0]!=seed for g in groups):raise AssertionError('Stream seeds differ')
        evs,cps,ivs,prs,finals=[g[1] for g in groups]
        checkpoints={r['checkpoint']:r for r in cps};interventions={(r['checkpoint'],r['intervention']):r for r in ivs}
        if set(checkpoints)!=set(c['checkpoints']+[c['change_tick']]):raise AssertionError('Checkpoint coverage')
        initial_world=World(seed,c);initial_agent=Agent(seed,c).snapshot()
        trackers={};events_by_branch=defaultdict(list);ids=set();metrics=defaultdict(list)
        for row in evs:
            branch=row['branch']
            if row['event_id'] in ids:raise AssertionError('Duplicate event')
            ids.add(row['event_id']);events_by_branch[branch].append(row)
            if branch not in trackers:
                if branch=='life':w=initial_world;state=initial_agent
                elif branch=='frozen_after_change':
                    origin=checkpoints[c['change_tick']];w=World.restore(origin['world'],c);state=deepcopy(origin['agent']);state['learn']=False
                else:
                    _,tick,rest=branch.split('_',2);iv,task=rest.rsplit('_',1);tick=int(tick);task=int(task)
                    w=World.restore(checkpoints[tick]['world'],c);observation_probe(w,task,c);state=interventions[tick,iv]['agent']
                trackers[branch]=dict(world=w,buffers=[deque(deepcopy(m['samples']),maxlen=c['model_window']) for m in state['models']],
                                      visits=list(state['visits']),learn=state['learn'],initial=state)
                # First choice in a fork must use its saved, intervened history.
                a=Agent.restore(state,c);act,decision=a.decide(w.observation())
                if act!=row['action'] or decision!=row['decision']:raise AssertionError('Fork did not use its initial model')
            tr=trackers[branch];w=tr['world'];before=w.observation();act=row['action'];decision=row['decision']
            if before!=row['before'] or w.step(act)!=row['after']:raise AssertionError('Continuity or physics mismatch')
            if decision['visits']!=tr['visits'] or decision['knowledge_samples']!=[len(b) for b in tr['buffers']]:raise AssertionError('History mismatch')
            if decision['chosen']!=act:raise AssertionError('Chosen action mismatch')
            prior=evaluation_loss(before,c)
            for candidate in decision['candidates']:
                a=candidate['action'];after=predicted_observation(before,candidate['predicted_delta'])
                close(candidate['benefit'],prior-evaluation_loss(after,c))
                close(candidate['exploration'],c['exploration_weight']/(1+tr['visits'][a])**.5)
                close(candidate['cost'],c['action_cost'] if a else 0)
                close(candidate['score'],candidate['benefit']+candidate['exploration']-candidate['cost'])
                if candidate['n']!=len(tr['buffers'][a]):raise AssertionError('Candidate buffer size')
            if decision['candidates'][act]['score']<max(x['score'] for x in decision['candidates'])-1e-12:raise AssertionError('Nonmaximal selection')
            tr['visits'][act]+=1
            if tr['learn']:
                tr['buffers'][act].append(dict(event_id=row['event_id'],x=features(before),
                                               y=[row['after']['body'][i]-before['body'][i] for i in range(2)]+[row['after']['fatigue']-before['fatigue']]))
            predicted=predicted_observation(before,decision['candidates'][act]['predicted_delta'])
            metrics[branch].append(dict(loss=physical_loss(row['after']),fatigue=row['after']['fatigue'],wait=int(act==0),
                                        mse=sum((predicted['body'][i]-row['after']['body'][i])**2 for i in range(2))/2,
                                        saturated=int(any(abs(b)>=.999999 for b in row['after']['body']))))
            if branch=='life' and w.tick in checkpoints:
                checkpoint=checkpoints[w.tick]
                if w.snapshot()!=checkpoint['world']:raise AssertionError('Checkpoint world mismatch')
                snapshot=checkpoint['agent']
                if tr['visits']!=snapshot['visits'] or [list(b) for b in tr['buffers']]!=[m['samples'] for m in snapshot['models']]:raise AssertionError('Checkpoint not actual recent history')
                if digest(snapshot)!=checkpoint['agent_sha256']:raise AssertionError('Checkpoint digest')
                for model in snapshot['models']:verify_model(model);counts['verified_models']+=1
            counts['events']+=1
        if len(events_by_branch['life'])!=c['life_ticks'] or len(events_by_branch['frozen_after_change'])!=c['life_ticks']-c['change_tick']:raise AssertionError('Life length')
        for cp in c['checkpoints']:
            state=checkpoints[cp]['agent'];world=checkpoints[cp]['world']
            for iv in c['interventions']:
                expected_state,expected_log=intervene(state,iv,world['tools'],c);actual=interventions[cp,iv]
                if expected_state!=actual['agent'] or any(actual[k]!=v for k,v in expected_log.items()):raise AssertionError('Intervention mismatch')
                counts['interventions']+=1
                for task in c['probe_tasks']:
                    name=f'probe_{cp}_{iv}_{task}';events=events_by_branch[name];mm=metrics[name];csvrow=bindex[seed,cp,iv,task]
                    if len(events)!=c['probe_ticks']:raise AssertionError('Probe duration')
                    close(statistics.mean(x['loss'] for x in mm),float(csvrow['body_loss']))
                    close(mm[-1]['loss'],float(csvrow['final_body_loss']))
                    if int(csvrow['waits'])!=sum(x['wait'] for x in mm):raise AssertionError('Wait tally')
                    if csvrow['initial_agent_sha256']!=digest(expected_state):raise AssertionError('Branch origin')
                    counts['branches']+=1
            for task in c['probe_tasks']:
                def behavior(iv):return [{k:r[k] for k in ('before','action','decision','after')} for r in events_by_branch[f'probe_{cp}_{iv}_{task}']]
                if behavior('retain')!=behavior('restore_axis0'):raise AssertionError('Restored behavior differs')
                counts['exact_restoration_pairs']+=1
        for r in prs:
            cp=r['checkpoint'];w=World.restore(checkpoints[cp]['world'],c);obs=observation_probe(w,r['task'],c)
            a=Agent.restore(interventions[cp,r['intervention']]['agent'],c)
            pred=predicted_observation(obs,a.models[r['action']].predict(obs,r['predictor']))
            target=predicted_observation(obs,w.mean_delta(obs,r['action']))
            if pred!=r['predicted'] or target!=r['target_noiseless']:raise AssertionError('Prediction provenance')
            mse=sum((pred['body'][i]-target['body'][i])**2 for i in range(2))/2
            close(mse,r['body_mse']);close(mse,float(pindex[seed,cp,r['intervention'],r['task'],r['predictor'],r['action']]['body_mse']))
            counts['prediction_probes']+=1
        for r in finals:
            tr=trackers[r['branch']]
            if tr['world'].snapshot()!=r['world'] or tr['visits']!=r['agent']['visits']:raise AssertionError('Final state')
            if [list(b) for b in tr['buffers']]!=[m['samples'] for m in r['agent']['models']]:raise AssertionError('Final history')
            for m in r['agent']['models']:verify_model(m);counts['verified_models']+=1
            if r['branch']=='frozen_after_change' and r['agent']['models']!=checkpoints[c['change_tick']]['agent']['models']:raise AssertionError('Frozen learned')
        for key,csvrow in windex.items():
            if key[0]!=seed:continue
            _,kind,start,end=key;name='life' if kind=='online' else 'frozen_after_change';offset=0 if kind=='online' else c['change_tick'];mm=metrics[name][start-offset:end-offset]
            for target,source in [('mean_body_loss','loss'),('mean_fatigue','fatigue'),('prediction_mse','mse')]:close(statistics.mean(x[source] for x in mm),float(csvrow[target]))
            for target,source in [('waits','wait'),('saturated','saturated')]:close(sum(x[source] for x in mm),float(csvrow[target]))
        first_online=events_by_branch['life'][c['change_tick']];first_frozen=events_by_branch['frozen_after_change'][0]
        for key in ('before','action','decision','after'):
            if first_online[key]!=first_frozen[key]:raise AssertionError('Changed-world origins not matched')
        counts['seeds']+=1
    if observed!=expected:raise AssertionError('Seed omissions')
    if counts['events']!=manifest['rows']['events'] or counts['branches']!=len(branch_csv) or counts['prediction_probes']!=len(pred_csv):raise AssertionError('Output counts')
    result=dict(status='passed',counts=dict(counts),checks=['raw hashes','all seeds retained','world continuity and replay','decision scores and visit history','actual recent-sample provenance','independent normal equations','selective erasure and unchanged controls','exact restore behavior','prediction targets and CSV agreement','main and frozen origin matching','frozen model unchanged','trajectory metric recomputation'],
                tolerance=1e-8,limitation='World transitions are replayed with the same implementation; this audit is not an independently coded simulator.')
    write_json(output/'audit.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();audit(a.output)
