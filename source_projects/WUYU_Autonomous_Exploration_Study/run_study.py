"""Run a declared cohort without overwriting evidence. Standard library only."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import time

from study import collect, evaluate, knowledge_metrics

ROOT = Path(__file__).resolve().parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def verify_freeze(path):
    frozen = json.loads(Path(path).read_text())
    for name, digest in frozen['files'].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError('Frozen file changed: ' + name)
    return frozen


def science_sources():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    paths += list((ROOT / 'baseline').rglob('*.py'))
    paths += [ROOT / p for p in ('protocol.json', 'SPECIFICATION.md', 'GIVEN_AND_LEARNED.md', 'PROTOCOL_FREEZE.json')]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def jsonl_gzip(path):
    # Fix gzip header timestamps and filename to allow byte-for-byte replay.
    raw = open(path, 'wb')
    compressed = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0)
    return raw, io.TextIOWrapper(compressed, encoding='utf-8', newline='\n')


def emit(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=True) + '\n')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cohort, output):
    protocol_freeze = verify_freeze(ROOT / 'PROTOCOL_FREEZE.json')
    config = json.loads((ROOT / 'protocol.json').read_text())
    if cohort == 'holdout':
        verify_freeze(ROOT / 'CODE_FREEZE.json')
    output = new_output(output)
    start = time.perf_counter()
    before = science_sources()
    manifest = dict(cohort=cohort, started_utc=utc_now(), status='running',
                    python=platform.python_version(), protocol=config, source_sha256=before,
                    protocol_frozen_utc=protocol_freeze['frozen_utc'])
    write_json(output / 'run_manifest.json', manifest)
    seed_spec = config[cohort + '_seeds']
    seeds = range(seed_spec['start'], seed_spec['start'] + seed_spec['count'])
    raw_handles, streams = [], []
    for name in ('acquisition.jsonl.gz', 'initial_learners.jsonl.gz', 'test_traces.jsonl.gz'):
        raw, stream = jsonl_gzip(output / name)
        raw_handles.append(raw)
        streams.append(stream)
    acquisition, initial_states, traces = streams
    knowledge, episodes, timings = [], [], []
    interaction_count = 0
    try:
        for index, seed in enumerate(seeds):
            for collector in config['collectors']:
                histories = {}
                stream_budgets = config['budgets'] if collector == 'balanced_teacher' else [max(config['budgets'])]
                for stream_budget in stream_budgets:
                    events, cpu_seconds = collect(seed, collector, stream_budget, config)
                    timings.append(dict(seed=seed, collector=collector, stream_budget=stream_budget,
                                        cpu_seconds=cpu_seconds))
                    interaction_count += len(events)
                    for step, event in enumerate(events, 1):
                        emit(acquisition, dict(seed=seed, collector=collector, stream_budget=stream_budget,
                                               step=step, experience=asdict(event)))
                    if collector == 'balanced_teacher':
                        histories[stream_budget] = events
                    else:
                        histories.update({b: events[:b] for b in config['budgets']})
                for budget in config['budgets']:
                    events = histories[budget]
                    key = dict(seed=seed, collector=collector, budget=budget)
                    metrics = knowledge_metrics(events, seed, config)
                    knowledge.append(dict(**key, **metrics))
                    history_digest = metrics['history_sha256']
                    for variant in config['variants']:
                        first_initial = None
                        for scenario in config['scenarios']:
                            row, trace, initial = evaluate(events, seed, variant, scenario, config)
                            if first_initial is None:
                                first_initial = initial
                                emit(initial_states, dict(**key, variant=variant,
                                     history_sha256=history_digest, initial=initial))
                            elif initial != first_initial:
                                raise AssertionError('Test branches received different initial learner states')
                            episode_key = dict(**key, variant=variant, scenario=scenario,
                                               history_sha256=history_digest)
                            episodes.append(dict(**episode_key, **row))
                            emit(traces, dict(**episode_key, trace=trace))
            if (index + 1) % 16 == 0:
                print(f'{cohort}: {index + 1}/{seed_spec["count"]} seeds', flush=True)
    finally:
        for stream in streams:
            stream.close()
        for raw in raw_handles:
            raw.close()
    write_csv(output / 'knowledge.csv', knowledge)
    write_csv(output / 'episodes.csv', episodes)
    write_csv(output / 'collector_timing.csv', timings)
    after = science_sources()
    if before != after:
        raise RuntimeError('Source changed during execution')
    verify_freeze(ROOT / 'PROTOCOL_FREEZE.json')
    if cohort == 'holdout':
        verify_freeze(ROOT / 'CODE_FREEZE.json')
    manifest.update(status='complete', completed_utc=utc_now(), elapsed_seconds=time.perf_counter() - start,
                    world_seeds=seed_spec['count'], histories=len(knowledge), episodes=len(episodes),
                    acquisition_interactions=interaction_count, test_actions=sum(r['steps'] for r in episodes),
                    output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != 'run_manifest.json'})
    write_json(output / 'run_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'cohort', 'histories', 'episodes', 'acquisition_interactions', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['development', 'holdout'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cohort, args.output)
