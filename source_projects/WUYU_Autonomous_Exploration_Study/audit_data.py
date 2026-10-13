"""Check raw-to-summary identity, branch isolation, and complete denominators."""
import argparse
from collections import Counter
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path

from baseline.wuyu_goal_study.core import Experience, World
from run_study import sha256, write_json
from study import fit, history_hash, knowledge_metrics


def jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def audit(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    config=manifest['protocol']
    for name,digest in manifest['output_sha256'].items():
        assert sha256(directory/name)==digest, name
    with open(directory/'knowledge.csv',newline='') as stream:
        krows={(int(r['seed']),r['collector'],int(r['budget'])):r for r in csv.DictReader(stream)}
    expected_snapshots={}
    stream_count=0
    raw_count=0
    history_count=0
    seen_streams=set()
    for key,group in groupby(jsonl(directory/'acquisition.jsonl.gz'),lambda r:(r['seed'],r['collector'],r['stream_budget'])):
        assert key not in seen_streams, 'Noncontiguous or duplicate stream'
        seen_streams.add(key)
        seed,collector,stream_budget=key
        records=list(group)
        assert [r['step'] for r in records]==list(range(1,stream_budget+1))
        events=[]
        for r in records:
            e=r['experience']
            e['before'],e['after']=tuple(e['before']),tuple(e['after'])
            events.append(Experience(**e))
        assert all(e.before==tuple(config['train_body']) for e in events)
        if collector!='balanced_teacher':
            assert events[0].state==World(seed).start
            assert all(a.next_state==b.state for a,b in zip(events,events[1:]))
            budgets=config['budgets']
        else:
            budgets=[stream_budget]
        for budget in budgets:
            prefix=events[:budget]
            hkey=(seed,collector,budget)
            digest=history_hash(prefix)
            assert digest==krows[hkey]['history_sha256']
            metrics=knowledge_metrics(prefix,seed,config)
            for name,value in metrics.items():
                assert ('' if value is None else str(value))==krows[hkey][name], (hkey,name)
            for variant in config['variants']:
                model=fit(prefix,variant,seed,config,World(seed).outcomes[0])
                expected_snapshots[hkey+(variant,)]=canonical(model.snapshot())
            history_count+=1
        raw_count+=len(events)
        stream_count+=1
    seen_initial=set()
    for row in jsonl(directory/'initial_learners.jsonl.gz'):
        key=(row['seed'],row['collector'],row['budget'],row['variant'])
        assert key not in seen_initial
        seen_initial.add(key)
        assert canonical(row['initial'])==expected_snapshots[key], key
        assert row['history_sha256']==krows[key[:3]]['history_sha256']
    assert seen_initial==set(expected_snapshots)
    with open(directory/'episodes.csv',newline='') as stream:
        episode_rows=list(csv.DictReader(stream))
    episode_lookup={}
    for row in episode_rows:
        key=(int(row['seed']),row['collector'],int(row['budget']),row['variant'],row['scenario'])
        assert key not in episode_lookup
        episode_lookup[key]=row
        assert row['history_sha256']==krows[key[:3]]['history_sha256']
    traces_seen=set()
    comparable={}
    matched_traces=0
    no_buffer_traces=0
    test_actions=0
    for row in jsonl(directory/'test_traces.jsonl.gz'):
        key=(row['seed'],row['collector'],row['budget'],row['variant'],row['scenario'])
        assert key not in traces_seen
        traces_seen.add(key)
        assert row['history_sha256']==krows[key[:3]]['history_sha256']
        trace=row['trace']
        assert trace[0]['state']==World(row['seed']).start
        steps=0
        behavioral=[]
        for i,tick in enumerate(trace):
            assert tick['tick']==i
            behavioral.append({k:v for k,v in tick.items() if k in ('tick','state','body','action','goal','experience')})
            if 'post_completion_decision' in tick:
                p=tick['post_completion_decision']
                behavioral[-1]['post_completion']={k:p[k] for k in ('action','goal')}
            if 'experience' in tick:
                e=tick['experience']
                assert e['state']==tick['state'] and e['action']==tick['action'] and e['before']==tick['body']
                steps+=1
                if i+1<len(trace):
                    assert trace[i+1]['state']==e['next_state'] and trace[i+1]['body']==e['after']
        assert steps==int(episode_lookup[key]['steps'])
        assert steps<=config['max_test_actions']
        test_actions+=steps
        bkey=key[:3]+(key[4],)
        if row['variant']=='learned':
            comparable[bkey]=behavioral
        elif row['variant']=='no_goal_buffer':
            no_buffer_traces+=1
            matched_traces+=comparable[bkey]==behavioral
    assert traces_seen==set(episode_lookup)
    spec=config[manifest['cohort']+'_seeds']
    seed_set=set(range(spec['start'],spec['start']+spec['count']))
    expected_episodes={(s,c,b,v,q) for s in seed_set for c in config['collectors'] for b in config['budgets']
                       for v in config['variants'] for q in config['scenarios']}
    assert traces_seen==expected_episodes
    assert raw_count==manifest['acquisition_interactions']
    assert history_count==manifest['histories']
    assert len(episode_rows)==manifest['episodes']
    assert test_actions==manifest['test_actions']
    result=dict(status='passed',streams=stream_count,raw_acquisition_events=raw_count,
                histories_verified=history_count,initial_learners_verified=len(seen_initial),
                episodes_verified=len(episode_rows),test_actions_verified=test_actions,
                learned_vs_no_goal_buffer_full_behavioral_traces_equal=matched_traces,
                learned_vs_no_goal_buffer_trace_comparisons=no_buffer_traces,
                checks=['Recorded file hashes','Raw history hashes and all knowledge metrics',
                        'Autonomous position continuity and declared body resets',
                        'All initial learner states rebuilt from raw acquisition before test feedback',
                        'Complete declared seeds and test branches; no missing failures',
                        'Trace state/body continuity and action counts'])
    write_json(directory/'data_audit.json',result)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    audit(parser.parse_args().directory)
