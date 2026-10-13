"""python -m wuyu_core list | run | resume"""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import Session, MODES
from .legacy import LegacySession, DESCRIPTIONS
from .state import read_tree


def compact(r):
    if 'outcome' in r:
        o=r['outcome']
        return dict(event=r['event'],action=r['action'],body=o['body'][0].tolist(),
                    alive=bool(o['alive'][0]),prediction_mse=r['prediction_mse'],bias_applied=r['bias_applied'])
    return {k:np.asarray(r[k]).reshape(-1)[:4].tolist() for k in ('action','reward','net_reward','correct') if k in r}


def main():
    p=argparse.ArgumentParser(description='物欲 01—08 统一运行入口')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    run=sub.add_parser('run');run.add_argument('--profile',default='integrated',choices=['integrated']+[f'{v:02}' for v in range(1,9)])
    run.add_argument('--mode',choices=MODES,default='prior');run.add_argument('--seed',type=int,default=990001)
    run.add_argument('--initialization',type=int,choices=range(5),default=0);run.add_argument('--horizon',type=int,default=1536)
    resume=sub.add_parser('resume');resume.add_argument('checkpoint')
    migrate=sub.add_parser('migrate08');migrate.add_argument('checkpoint');migrate.add_argument('--individual',type=int,default=0)
    migrate.add_argument('--mode',choices=MODES,default='prior');migrate.add_argument('--save',required=True)
    for parser in (run,resume):
        parser.add_argument('--steps',type=int,default=48);parser.add_argument('--save');parser.add_argument('--trace')
    a=p.parse_args()
    if a.command=='list':
        print(json.dumps({'integrated':'统一身体闭环；默认 prior；可选在线学习与回放实验',**{f'{k:02}':v for k,v in DESCRIPTIONS.items()}},ensure_ascii=False,indent=2));return
    if a.command=='migrate08':
        if Path(a.save).exists():p.error('请使用新的存档文件名')
        from .migration import migrate_08
        s=migrate_08(a.checkpoint,a.individual,a.mode);s.save(a.save)
        print(json.dumps(dict(saved=a.save,lineage=s.lineage),ensure_ascii=False,indent=2));return
    if a.steps<1: p.error('steps must be positive')
    if a.save and Path(a.save).exists(): p.error('请使用新的存档文件名，避免覆盖已有个体')
    if a.command=='run':
        if a.horizon<1:p.error('horizon must be positive')
        s=Session.create(a.seed,a.initialization,a.mode,a.horizon) if a.profile=='integrated' else LegacySession.create(int(a.profile),a.seed,a.horizon)
    else:
        state=read_tree(a.checkpoint)
        s=Session.from_state(state) if state.get('profile')=='integrated' else LegacySession.from_state(state)
    rows=[]
    for _ in range(a.steps):
        try: rows.append(compact(s.step()))
        except StopIteration: break
    if a.save: s.save(a.save)
    if a.trace:
        path=Path(a.trace);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    print(json.dumps(dict(executed=len(rows),first=rows[:3],last=rows[-1:] or [],saved=a.save,
                         note='程序结束后不在后台继续；恢复只在既定世界时长内前进。'),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
