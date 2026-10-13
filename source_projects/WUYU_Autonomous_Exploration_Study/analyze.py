"""Predeclared seed-level summaries; no optional stopping or model selection."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

from run_study import ROOT, write_json


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0.0, center-half), min(1.0, center+half)]


def proportion(values):
    n = len(values)
    k = sum(values)
    return dict(successes=k, n=n, rate=k/n, wilson95=wilson(k, n))


def quantile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def paired_bootstrap(pairs, resamples, seed):
    differences = [a-b for a, b in pairs]
    rng = random.Random(seed)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples)]
    return dict(difference=sum(differences)/n, percentile95=[quantile(samples, .025), quantile(samples, .975)],
                n_pairs=n, resamples=resamples, only_directed=sum(a and not b for a,b in pairs),
                only_random=sum(b and not a for a,b in pairs), both=sum(a and b for a,b in pairs),
                neither=sum(not a and not b for a,b in pairs))


def numeric_summary(values):
    if not values:
        return dict(n=0, mean=None, median=None, min=None, max=None)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values))


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'run_manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete run')
    config = manifest['protocol']
    with open(directory/'episodes.csv', newline='') as f:
        episodes = list(csv.DictReader(f))
    with open(directory/'knowledge.csv', newline='') as f:
        knowledge = list(csv.DictReader(f))
    with open(directory/'collector_timing.csv', newline='') as f:
        timings = list(csv.DictReader(f))
    by_scenario, by_individual = defaultdict(list), defaultdict(dict)
    rows_by_key = {}
    for row in episodes:
        seed, budget = int(row['seed']), int(row['budget'])
        c, v, s = row['collector'], row['variant'], row['scenario']
        by_scenario[c,budget,v,s].append(row)
        by_individual[c,budget,v,seed][s] = int(row['success'])
        rows_by_key[c,budget,v,seed,s] = row
    all_four = {}
    composite = []
    for c in config['collectors']:
        for b in config['budgets']:
            for v in config['variants']:
                per_seed = {}
                for (cc,bb,vv,seed), cases in by_individual.items():
                    if (cc,bb,vv) == (c,b,v):
                        if set(cases) != set(config['scenarios']):
                            raise AssertionError('Missing or extra scenario')
                        per_seed[seed] = int(all(cases.values()))
                if len(per_seed) != manifest['world_seeds']:
                    raise AssertionError('Missing seeds')
                all_four[c,b,v] = per_seed
                composite.append(dict(collector=c,budget=b,variant=v,**proportion(list(per_seed.values()))))
    comparisons = []
    for b in config['budgets']:
        directed, random_result = all_four['least_visited',b,'learned'], all_four['random',b,'learned']
        pairs = [(directed[seed], random_result[seed]) for seed in sorted(directed)]
        comparison = dict(budget=b, primary=b==config['primary_budget'],
                          **paired_bootstrap(pairs, config['bootstrap_resamples'], config['bootstrap_seed']))
        comparison['supports_predeclared_direction'] = comparison['percentile95'][0] > 0 if comparison['primary'] else None
        comparisons.append(comparison)
    scenarios = []
    for (c,b,v,s), rows in sorted(by_scenario.items()):
        post = [int(r['post_completion_stop']) for r in rows if r['post_completion_stop'] != '']
        scenarios.append(dict(collector=c,budget=b,variant=v,scenario=s,
             **proportion([int(r['success']) for r in rows]),
             first_choice=proportion([int(r['first_choice_correct']) for r in rows]),
             steps=numeric_summary([int(r['steps']) for r in rows]),
             post_completion_stop=proportion(post) if post else None,
             failure_reasons=dict(sorted(Counter(r['failure_reason'] for r in rows).items()))))
    knowledge_summary = []
    for c in config['collectors']:
        for b in config['budgets']:
            rows = [r for r in knowledge if r['collector']==c and int(r['budget'])==b]
            ks = dict(collector=c,budget=b,n=len(rows),
                      both_discovered=proportion([int(r['both_discovered']) for r in rows]))
            for metric in ('state_action_coverage','visited_states','effect_rmse','self_loop_fraction','repeat_pair_fraction',
                           'samples_a','samples_b','first_discovery_a','first_discovery_b','known_route_a','known_route_b'):
                ks[metric] = numeric_summary([float(r[metric]) for r in rows if r[metric] != ''])
            knowledge_summary.append(ks)
    equivalence = []
    behavioral_fields = ('success','first_choice_correct','steps','first_goal','first_action','outcome',
                         'expected_outcome','post_completion_stop','failure_reason')
    for c in config['collectors']:
        for b in config['budgets']:
            matched, n = 0, 0
            for (cc,bb,v,seed,s), row in rows_by_key.items():
                if (cc,bb,v) == (c,b,'learned'):
                    other = rows_by_key[c,b,'no_goal_buffer',seed,s]
                    if row['history_sha256'] != other['history_sha256']:
                        raise AssertionError('Mechanism comparison has unequal histories')
                    matched += all(row[k]==other[k] for k in behavioral_fields)
                    n += 1
            equivalence.append(dict(collector=c,budget=b,matched_episode_summaries=matched,n_episodes=n))
    timing_summary = []
    for c in config['collectors']:
        for b in sorted({int(r['stream_budget']) for r in timings if r['collector']==c}):
            rows = [float(r['cpu_seconds']) for r in timings if r['collector']==c and int(r['stream_budget'])==b]
            timing_summary.append(dict(collector=c,stream_budget=b,**numeric_summary(rows)))
    result = dict(cohort=manifest['cohort'],world_seeds=manifest['world_seeds'],
                  primary=next(x for x in comparisons if x['primary']),
                  paired_learning_curve=comparisons,all_four=composite,scenarios=scenarios,
                  knowledge=knowledge_summary,goal_buffer_observed_agreement=equivalence)
    # Timings vary between reproductions, so store separately from scientific results.
    write_json(directory/'summary.json', result)
    write_json(directory/'timing_summary.json', timing_summary)
    print(json.dumps(result['primary'], ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    analyze(parser.parse_args().directory)
