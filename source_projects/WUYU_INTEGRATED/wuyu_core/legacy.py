"""One step/save/load contract over eight unchanged historical implementations.

The envelope includes previously external event streams. Version-specific actions
retain their meaning; a macro-action in 02 is never called a primitive action in 08.
"""
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from .state import read_tree, write_tree

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    2: 'results/checkpoint_matched.npz', 3: 'results/main/checkpoint_online_stage5.npz',
    4: 'results/checkpoint_active_mid.npz', 5: 'results/checkpoint_mid.npz',
    6: 'results/checkpoint_full_0_128.npz', 7: 'results/checkpoint_neural_student_0_384.npz',
    8: 'results/checkpoint_graph_varied_0_384.npz'}
MODULES = {1:'wuyu.core',2:'wuyu02.workshop',3:'wuyu03.core',4:'wuyu04.latent',
           5:'wuyu05.core',6:'wuyu06.state',7:'wuyu07.experiment',8:'wuyu08.experiment'}
SERIALIZERS = {1:('save_checkpoint','load_checkpoint'),2:('checkpoint','restore'),
               3:('save_checkpoint','load_checkpoint'),4:('save','load'),5:('save','load'),
               6:('save','load'),7:('checkpoint','restore'),8:('checkpoint','restore')}
DESCRIPTIONS = {
    1:'六位置价值记忆与学习率门控',2:'可消耗工具与 Q 学习宏动作',
    3:'私有循环感知网络、经验学习与工具控制',4:'隐情境信念与主动付费观测（独立世界）',
    5:'学习率对照、独立随机流与重算回放',6:'身体稳态、预测、经验目标与动作',
    7:'经验模型、动作技能与神经学生',8:'关系图神经动作与具身经验记忆'}


def module(version):
    if version not in MODULES: raise ValueError('version must be 1..8')
    directory = str(ROOT / f'legacy/WUYU_{version:02}')
    if directory not in sys.path: sys.path.insert(0,directory)
    return importlib.import_module(MODULES[version])


class LegacySession:
    def __init__(self, version, world, agent, external=None, cursor=0):
        self.version, self.world, self.agent = version, world, agent
        self.external, self.cursor = external, cursor
        self.mod = module(version)

    @classmethod
    def create(cls, version, seed=990001, horizon=2048):
        m = module(version); root = ROOT / f'legacy/WUYU_{version:02}'
        external = None
        if version == 1:
            model = json.loads((root/'results/models.json').read_text())
            scenario = m.make_scenario(('mixed',),1,horizon,seed)
            w, a = m.World(scenario), m.Agent(1,'adaptive',np.asarray(model['selected']['theta']).reshape(1,-1))
        else:
            w, a = getattr(m,SERIALIZERS[version][1])(root/CHECKPOINTS[version])
        if version == 2:
            external = dict(events=dataclasses.asdict(m.events(len(w.charge),horizon,seed)))
        if version == 3:
            external = dict(events=dataclasses.asdict(m.events(np.arange(seed,seed+a.net.n),horizon)),context=0,reverse=False)
        return cls(version,w,a,external)

    def step(self):
        v,m,w,a = self.version,self.mod,self.world,self.agent
        t = self.cursor
        if v == 1:
            if w.t >= w.scenario.horizon: raise StopIteration
            result = m.rollout(w.scenario,a.mode,a.parameter,initial=(w,a),start=w.t,stop=w.t+1,capture=True)
            row = result['trajectory'][0]
            result = dict(reward=np.array([row[0]]),action=np.array([int(row[4])]),alpha=np.array([row[3]]))
        elif v == 2:
            ev = m.Events(**self.external['events'])
            if t >= ev.horizon: raise StopIteration
            result = {k:value[0] for k,value in m.run(ev,w,a,t,t+1).items()}
        elif v == 3:
            ev = m.Exogenous(**self.external['events'])
            if t >= len(ev.target): raise StopIteration
            result = m.step(w,a,ev,t,self.external['context'],self.external['reverse'],frozen=(a.mode=='frozen'))
        elif v in (4,5):
            result = m.step(w,a)
        elif v == 6:
            if w.t >= w.total: raise StopIteration
            result = importlib.import_module('wuyu06.experiment').transition(w,a)
            result = dict(action=result['action'],**result['world'])
        else:
            if w.t >= w.total: raise StopIteration
            result = (m.step if v==7 else m.neural_step)(w,a)
            result = dict(action=result['action'],**result['out'])
        self.cursor += 1
        return result

    def state(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'inner.npz'
            getattr(self.mod,SERIALIZERS[self.version][0])(p,self.world,self.agent)
            payload = p.read_bytes()
        return dict(schema='wuyu.legacy-envelope/1', version=self.version, cursor=self.cursor,
                    external=self.external, inner=np.frombuffer(payload,dtype=np.uint8).copy(),
                    inner_sha256=hashlib.sha256(payload).hexdigest())

    def save(self,path): write_tree(path,self.state())

    @classmethod
    def from_state(cls,s):
        if s.get('schema') != 'wuyu.legacy-envelope/1': raise ValueError('unsupported legacy schema')
        v=s['version'];m=module(v);data=s['inner'].tobytes()
        if hashlib.sha256(data).hexdigest() != s['inner_sha256']: raise ValueError('checkpoint checksum mismatch')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'inner.npz';p.write_bytes(data)
            w,a=getattr(m,SERIALIZERS[v][1])(p)
        return cls(v,w,a,s['external'],s['cursor'])

    @classmethod
    def load(cls,path): return cls.from_state(read_tree(path))
