"""Engineering acceptance of all registered WUYU configurations.

No training policy, frozen experiment, source lock, or scientific sample is
modified. Original tests should additionally be run in isolated processes.
Outputs are create-only and contain per-case evidence, including failures.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from wuyu_unified import codec
from wuyu_unified.audit import verify_sources
from wuyu_unified.paths import config
from wuyu_unified.registry import DESCRIPTIONS, MODES, CASES
from wuyu_unified.runtime import Session, source_locks


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def configurations():
    jobs = []
    for profile in DESCRIPTIONS:
        for mode in MODES.get(profile, ('native_checkpoint',)):
            for case in CASES.get(profile, (None,)):
                if profile in ('history', 'revision'):
                    horizon = config(profile)['life_ticks']
                elif profile == 'exploration':
                    horizon = max(config(profile)['budgets'])
                elif profile in ('integrated', 'workshop_bridge'):
                    horizon = 1536
                elif profile == 'whole':
                    horizon = 1536 if case == 'native' else 2560
                elif profile in ('01', '02', '03'):
                    horizon = 256
                else:
                    horizon = None
                jobs.append(dict(profile=profile, mode=mode, case=case,
                    seed=1770100, horizon=horizon, initialization=0,
                    requested_steps=256 if profile.isdigit() else horizon or 16,
                    kind='registered_configuration'))
    for init in range(1, 5):
        jobs.append(dict(profile='whole', mode='R10', case='r_combined',
            seed=1770100, horizon=2560, initialization=init,
            requested_steps=2560, kind='remaining_model_initialization'))
    for mode in ('R00', 'R10'):
        jobs.append(dict(profile='whole', mode=mode, case='r_combined',
            seed=1770101, horizon=5120, initialization=0,
            requested_steps=5120, kind='long_continuous_life'))
    for i, job in enumerate(jobs):
        job['id'] = f'{i:03}_{job["profile"]}_{job["mode"]}_{job["case"] or "native"}_i{job["initialization"]}_{job["kind"]}'
    return jobs


def run_job(job, directory):
    start = time.monotonic()
    result = dict(job=job, passed=False)
    try:
        kwargs = {k: job[k] for k in ('profile', 'mode', 'case', 'seed', 'horizon', 'initialization')}
        s = Session.create(**kwargs)
        result['initial_state_sha256'] = codec.digest(s.state(False))
        full = job['profile'] == 'whole'
        if full:
            b = s.backend
            original_network = codec.digest(b.native.individual.state()['network'])
            original_actor = None if b.actor is None else codec.digest(b.actor.p)
        previous = None
        steps = 0
        action_hash = hashlib.sha256()
        counters = dict(live_transitions=0, dead_transitions=0, revisions=0,
                        actual_relation_updates=0, products=0, route_denied=0)
        burdens, bodies, actions, alive = [], [], [], []

        def accept(e):
            nonlocal previous, steps
            assert e['event_index'] == steps
            steps += 1
            action_hash.update(codec.canonical(codec.jsonable([e['event_index'], e.get('action')])).encode())
            if full:
                before, after = e['before'], e['after']
                assert e['clock_before'] == steps - 1 and e['clock_after'] == steps
                assert s.backend.native.world.t == steps
                assert s.backend.native.individual.tick == steps
                assert s.backend.native.individual.memory.tick == steps
                assert s.backend.native.individual.pending is None
                assert e['accounting']['primitive_actions'] == 1
                if previous is not None:
                    np.testing.assert_array_equal(before, previous)
                assert np.isfinite(before).all() and np.isfinite(after).all()
                p = e['decision']['final_probability']
                assert p.shape == (7,) and np.isfinite(p).all() and (p >= 0).all()
                assert abs(float(p.sum()) - 1.) < 1e-12
                assert 0 <= e['action'] < 7
                assert (after[14:16] >= 0).all() and (after[14:16] <= 1).all()
                assert bool(e['learning']['updated']) == bool(before[14:16].min() > .001)
                if before[14:16].min() <= .001:
                    assert not e['alive']
                    np.testing.assert_array_equal(before[12:16], after[12:16])
                    counters['dead_transitions'] += 1
                else:
                    counters['live_transitions'] += 1
                assert e['movement']['read_by_policy'] == (job['mode'] in ('R10', 'R11', 'RT'))
                if job['mode'] in ('R00', 'R10'):
                    assert e['movement']['prior_tau'] == 1.
                if job['mode'] == 'RT':
                    assert abs(e['movement']['prior_tau'] - .9) < 1e-12
                counters['revisions'] += int(e['learning']['revision'])
                counters['actual_relation_updates'] += int(e['movement']['update'] is not None)
                counters['products'] += int(e['transition_semantics']['product'])
                counters['route_denied'] += int(e['native_payload']['outcome'].get('route_denied', False))
                burdens.append(e['body_burden']); bodies.append(after[14:16].copy())
                actions.append(e['action']); alive.append(e['alive'])
                previous = after.copy()

        prefix = 1 if job['profile'] == 'goals' else 73
        for _ in range(min(prefix, job['requested_steps'])):
            try:
                e = s.step()
            except StopIteration:
                break
            accept(e)
        with tempfile.TemporaryDirectory(prefix='wuyu-acceptance-') as td:
            checkpoint = Path(td) / 'mid.wuyu'
            s.save(checkpoint)
            other = Session.load(checkpoint)
            assert codec.digest(s.state(False)) == codec.digest(other.state(False))
            result['checkpoint_event'] = s.event_index
            result['pending_actor_samples'] = len(s.backend.actor.buffer) if full and s.backend.actor else None
            future_steps = 0
            for _ in range(min(55, job['requested_steps'] - steps)):
                try:
                    e = s.step()
                except StopIteration:
                    try:
                        other.step()
                    except StopIteration:
                        break
                    raise AssertionError('only one continuation stopped')
                alt = other.step()
                assert codec.digest(e) == codec.digest(alt)
                accept(e); future_steps += 1
            assert codec.digest(s.state(False)) == codec.digest(other.state(False))
            result['exact_resume_future_events'] = future_steps
            result['exact_resume'] = True
            for _ in range(job['requested_steps'] - steps):
                try:
                    e = s.step()
                except StopIteration:
                    break
                accept(e)
            s.save(Path(td) / 'final.wuyu')
            restored = Session.load(Path(td) / 'final.wuyu')
            assert codec.digest(s.state(False)) == codec.digest(restored.state(False))
            if not job['profile'].isdigit():
                final_digest = codec.digest(s.state(False))
                assert s.run(1) == []
                assert final_digest == codec.digest(s.state(False))
                result['declared_end_is_stable'] = True
            if job['kind'] == 'long_continuous_life' or (full and job['case'] == 'r_combined' and job['initialization'] == 0):
                shutil.copyfile(Path(td) / 'final.wuyu', Path(directory) / (job['id'] + '.wuyu'))
        result.update(events=steps, action_stream_sha256=action_hash.hexdigest(),
                      final_state_sha256=codec.digest(s.state(False)))
        if full:
            b = s.backend; ind = b.native.individual
            assert steps == job['requested_steps']
            assert codec.digest(ind.state()['network']) == original_network
            assert ind.memory.updates > 0 and ind.predictor.seen > 0
            assert ind.predictor.replay_updates > 0 and sum(b.visits) > 0
            assert int(sum(b.visits)) == counters['live_transitions']
            if b.actor is not None:
                assert b.actor.updates > 0
                assert original_actor != codec.digest(b.actor.p)
                assert not b.actor.buffer
            result['mechanisms'] = dict(**counters, memory_updates=ind.memory.updates,
                learned_skill_completions=int(ind.memory.skill_count.sum()),
                neural_seen=ind.predictor.seen, neural_replay_updates=ind.predictor.replay_updates,
                evidence_updates=int(sum(b.visits)), actor_updates=0 if b.actor is None else b.actor.updates,
                relations_observed=0 if b.relations is None else int(b.relations.count.sum()),
                old_graph_weights_unchanged=True, relation_read=b.relation_read,
                final_alive=bool(b.native.world.alive[0]), mean_body_burden=float(np.mean(burdens)))
            np.savez_compressed(Path(directory) / (job['id'] + '.npz'),
                body=np.asarray(bodies), action=np.asarray(actions, dtype=np.int8),
                alive=np.asarray(alive), burden=np.asarray(burdens))
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
    result['seconds'] = round(time.monotonic() - start, 3)
    write(Path(directory) / (job['id'] + '.json'), result)
    return result


def syntax_inventory():
    errors = []
    paths = []
    # Includes retained snapshots and all original project trees. Symlinked
    # aliases and generated bytecode are excluded to prevent double counting.
    for directory in ('wuyu_unified', 'tools', 'tests', 'source_projects', 'reference_runtime_v1'):
        for p in sorted((ROOT / directory).rglob('*.py')):
            if p.is_symlink() or any(x.startswith('.') for x in p.relative_to(ROOT).parts):
                continue
            paths.append(p)
            try:
                compile(p.read_bytes(), str(p), 'exec', dont_inherit=True)
            except Exception as exc:
                errors.append(dict(path=p.relative_to(ROOT).as_posix(), error=str(exc)))
    js = []
    for p in (ROOT / 'source_projects').rglob('*.js'):
        r = subprocess.run(['node', '--check', str(p)], capture_output=True, text=True)
        js.append(dict(path=p.relative_to(ROOT).as_posix(), returncode=r.returncode, stderr=r.stderr))
    return dict(passed=not errors and all(x['returncode'] == 0 for x in js),
                python_files=len(paths), errors=errors, javascript=js,
                python_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in paths})


def cli_acceptance(directory):
    directory = Path(directory); directory.mkdir()
    commands = [
        ['run','--profile','whole','--mode','R10','--case','r_combined','--seed','1770100','--horizon','2560','--steps','1173','--out',str(directory/'continuous')],
        ['run','--profile','whole','--mode','R10','--case','r_combined','--seed','1770100','--horizon','2560','--steps','1097','--out',str(directory/'prefix')],
        ['resume',str(directory/'prefix/state.wuyu'),'--steps','76','--out',str(directory/'resumed')],
        ['inspect',str(directory/'resumed/state.wuyu')],
    ]
    for i, args in enumerate(commands):
        r = subprocess.run([sys.executable,'-B','-m','wuyu_unified',*args],cwd=ROOT,capture_output=True,text=True,timeout=300)
        (directory/f'command_{i}.log').write_text(r.stdout+r.stderr)
        if r.returncode:
            raise RuntimeError(r.stderr)
    a=(directory/'continuous/events.jsonl').read_bytes()
    b=(directory/'prefix/events.jsonl').read_bytes()+(directory/'resumed/events.jsonl').read_bytes()
    assert a == b
    assert (directory/'continuous/state.wuyu').read_bytes() == (directory/'resumed/state.wuyu').read_bytes()
    # Retain one complete event stream plus both useful checkpoints; the
    # duplicate prefix/resume event streams are represented by their hashes.
    result=dict(passed=True, events=1173, pause_at=1097, resumed_events=76,
        event_bytes_equal=True, checkpoint_bytes_equal=True,
        events_sha256=hashlib.sha256(a).hexdigest(),
        checkpoint_sha256=file_sha(directory/'resumed/state.wuyu'), commands=commands)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    locks=source_locks('whole'); write(args.out/'SOURCE_LOCK_BEFORE.json',locks)
    source=verify_sources(); write(args.out/'ORIGINAL_SOURCES.json',source)
    syntax=syntax_inventory(); write(args.out/'SYNTAX.json',syntax)
    jobs=configurations(); write(args.out/'CONFIGURATIONS.json',jobs)
    runs=args.out/'runs'; runs.mkdir()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,j,runs) for j in jobs]
        for f in as_completed(futures):
            r=f.result(); rows.append(r)
            print(json.dumps(dict(completed=len(rows),total=len(jobs),job=r['job']['id'],passed=r['passed'],seconds=r['seconds'],error=r.get('error'))),flush=True)
    try:
        cli=cli_acceptance(args.out/'cli')
    except Exception:
        cli=dict(passed=False,error=traceback.format_exc())
    write(args.out/'CLI.json',cli)
    unchanged=locks==source_locks('whole')
    result=dict(passed=source['passed'] and syntax['passed'] and cli['passed'] and unchanged and all(x['passed'] for x in rows),
        completed_utc=datetime.now(timezone.utc).isoformat(), seconds=round(time.monotonic()-start,3),
        profiles=len(DESCRIPTIONS), registered_configurations=sum(x['kind']=='registered_configuration' for x in jobs),
        total_runs=len(rows), whole_runs=sum(x['job']['profile']=='whole' for x in rows),
        main_events=sum(x.get('events',0) for x in rows),
        resume_branch_events=sum(x.get('exact_resume_future_events',0) for x in rows),
        source_locks_unchanged=unchanged, original_sources=source,
        syntax_summary={k:v for k,v in syntax.items() if k!='python_sha256'}, cli=cli,
        failures=[x for x in rows if not x['passed']],
        runs=sorted(rows,key=lambda r:r['job']['id']),
        scope='Engineering acceptance over fixed configurations and seeds; not an additional independent efficacy study, proof of every code path, or indefinite viability.')
    write(args.out/'ACCEPTANCE.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','cli','syntax_summary','original_sources','failures')},ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
