"""One whole embodied mainline: corrected predictors + action learning + state.

A/B/C/D are internal configurations of this same lifecycle, not separate toy
agents. The original WorkshopBridge remains a frozen regression anchor.
"""
from copy import deepcopy
import importlib
import numpy as np
from .paths import source_module
from .workshop_bridge import WorkshopBridge,features,FEATURE_SCHEMA
from .action_learning import ActionLearner,softmax
from .whole_world import WholeWorld,HORIZON
from .relations import MovementRelations

RELATION_MODES = ('R00','R10','R01','R11','RT')

TARGET_SCHEMA='body-delta2-inventory-delta2-same-device-stage-change-product-present/2'
ACTION_FEATURE_SCHEMA='visible-observation-old-memory-predictions-pre-feedback-errors/1'
ACTION_INPUTS=378

def target(before,after):
    same=int(before[:6].argmax())==int(after[:6].argmax())
    progress=(after[6:9].argmax()-before[6:9].argmax())/2. if same else 0.
    return np.concatenate([after[14:16]-before[14:16],after[12:14]-before[12:14],
                           [progress,float(after[16:20].sum()>.5)]])

def corrected_feedback(ind,next_obs):
    """Same native learning lifecycle, corrected target for neural AND replay."""
    if ind.pending is None:raise RuntimeError('decision required')
    q=ind.pending;before=q['obs'][0];after=np.asarray(next_obs)[0]
    y=target(before,after);y[:2]/=.28
    alive=bool(before[14:16].min()>.001)
    if alive and ind.mode!='prior':ind.predictor.feedback(q['x'],y,replay=ind.mode in ('replay','shadow'))
    ind.memory.feedback(dict(action=q['action'],obs=before),after)
    ind.prev=np.concatenate([np.eye(7)[[q['action']]],((after[14:16]-before[14:16])/.28)[None]],axis=1)
    q.update(target=y,prediction_mse=float(np.mean((q['prediction']-y)**2)),alive=alive)
    ind.pending=None;ind.tick+=1
    return q

class WholeAgent(WorkshopBridge):
    @classmethod
    def create(cls,seed,mode='A',horizon=None,initialization=0,case='native',
               split='development',action_configuration=None,relation_configuration=None,**_):
        if mode not in ('A','B','C','D') + RELATION_MODES:raise ValueError(mode)
        total=horizon or (1536 if case in ('native',None) else HORIZON)
        obj=cls();base=WorkshopBridge.create(seed,'combined',total,initialization)
        obj.__dict__.update(base.__dict__);obj.condition=mode
        if case and case.startswith('r_'):
            from .relation_world import RelationWorld
            obj.native.world=RelationWorld(seed,total,case,split)
        elif case not in ('native',None):obj.native.world=WholeWorld(seed,total,case,split)
        obj.target_schema=TARGET_SCHEMA;obj.action_feature_schema=ACTION_FEATURE_SCHEMA
        configuration = dict(action_configuration or {})
        if mode in RELATION_MODES:
            configuration.update(allocate_gate=True, prior_gate='learned' if mode in ('R01','R11') else 'fixed' if mode=='RT' else 'off')
        obj.actor=None if mode=='A' else ActionLearner(ACTION_INPUTS,seed*59+initialization*100003,'C' if mode in RELATION_MODES else mode,configuration)
        obj.relation_read = mode in ('R10','R11','RT')
        obj.relations = MovementRelations(**(relation_configuration or {})) if mode in RELATION_MODES or (case and case.startswith('r_')) else None
        obj.errors=np.zeros((7,2));obj.error_counts=np.zeros(7,int)
        obj.suppress_until=-1;obj.last_action_learning=None;obj.migration=None
        return obj

    def encode_observation(self, obs):
        enc = super().encode_observation(obs)
        if getattr(self, 'relation_read', False):
            enc['node'][0,:,28:30] = self.relations.distances(int(obs[:6].argmax()), self.native.individual.tick)
        return enc

    def decide_with_relations(self, obs, forced):
        """Original private lifecycle with the SAME encoding in preview/commit.

        The inherited source remains immutable. Predictor labels and world
        interfaces do not change; only the two declared distance inputs do.
        """
        ind = self.native.individual
        if ind.pending is not None: raise RuntimeError('feedback required')
        ind.ingest(obs[None]); x = self.encode_observation(obs)
        r = ind.net.forward(x, ind.h)
        context = np.concatenate([obs,ind.prev[0],x['node'][0,obs[:6].argmax()]])
        predictions,candidates = ind.predictor.predict(context)
        need = (1-obs[14:16])**2+.02
        utility = predictions[:,:2]@need+.05*predictions[:,2:4].sum(1)+.05*predictions[:,4]
        bias = np.clip(utility*ind.bias_strength,-1.,1.)
        use = ind.mode in ('online','replay') and ind.predictor.seen>=ind.warmup
        probability = softmax(np.log(np.maximum(r['ap'][0],1e-12))+(bias if use else 0.))
        ind.rng.random()
        action = int(forced)
        if action not in range(7): raise ValueError('invalid action')
        ind.h = r['h'].copy()
        ind.pending = dict(action=action,obs=obs[None].copy(),x=candidates[action].copy(),
                           prediction=predictions[action].copy(),probability=probability.copy(),
                           bias=bias.copy(),bias_applied=bool(use))
        return action

    def action_features(self,obs,d,evidence):
        ind=self.native.individual;encode=importlib.import_module('wuyu_core.features').encode
        enc=self.encode_observation(obs)
        ev=np.array([[min(e['age']/240,1),e['n']/24,
                      1. if e['coverage'] is None else min(e['coverage']/4,1)] for e in evidence])
        pred=d['predictions'].copy();pred[:,:2]/=.28
        x=np.concatenate([obs,ind.prev[0],enc['node'].ravel(),enc['edge'].ravel(),
                          d['probability'],np.clip(pred,-2,2).ravel(),
                          d['neural_predictions'].ravel(),ev.ravel(),
                          np.clip(self.errors,0,2).ravel()])
        if len(x)!=ACTION_INPUTS:raise AssertionError(len(x))
        return x

    def decision(self,obs):
        d=self.candidate_decision(obs);tick=self.native.individual.tick
        evidence=[m.evidence(d['features'],tick) for m in self.models]
        cache=None;probability=d['probability'].copy()
        if self.actor is not None:
            x=self.action_features(obs,d,evidence)
            cache=self.actor.forward(x,probability,suppress=tick<self.suppress_until,value_offset=self.value_prior(obs))
            probability=cache['probability']
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(self.native.individual.rng.bit_generator.state)
        draw=rng.random();chosen=int(min(np.searchsorted(probability.cumsum(),draw),6))
        return d,evidence,cache,probability,chosen

    def value_prior(self,obs):
        # Initial value guess: the visible deficit would persist. This does not
        # change the physical reward, add subgoals, or read future outcomes.
        if self.actor is None or not self.actor.c.get('body_value_prior',False):return 0.
        return -float(np.sum((1-np.asarray(obs)[14:16])**2))/(1-self.actor.c['gamma'])

    def step(self,forced=None):
        world=self.native.world;ind=self.native.individual
        if world.t>=world.total:raise StopIteration('declared horizon reached')
        before=world.observation()[0].copy();tick=world.t;was=bool(world.alive[0])
        ind.ingest(before[None]);d,evidence,cache,probability,chosen=self.decision(before)
        errors_before=self.errors.copy()
        relation_before = None if self.relations is None else self.relations.success(tick)
        selected = chosen if forced is None else forced
        action=self.decide_with_relations(before,selected) if self.relation_read else ind.decide(before[None],forced=selected)
        if cache is not None and was:self.actor.commit(cache)
        outcome=world.step(np.array([action]));after=world.observation()[0].copy()
        record=corrected_feedback(ind,after[None]);record.update(outcome=outcome,event=world.t)
        relation_event = None if self.relations is None else self.relations.update(before, action, after, tick)
        y=target(before,after);eid=f'whole:{ind.seed}:{tick}'
        revision=False;residual=None
        if was:
            ridge_error=float(np.mean(((y-d['predictions'][action])/np.array([.28,.28,1,1,1,1]))**2))
            neural_y=y.copy();neural_y[:2]/=.28
            neural_error=float(np.mean((neural_y-d['neural_predictions'][action])**2))
            self.errors[action]=.95*self.errors[action]+.05*np.array([ridge_error,neural_error])
            self.error_counts[action]+=1
            residual,revision=self.models[action].update(d['features'],y,tick,eid,d['predictions'][action],True)
            self.visits[action]+=1;self.last_use[action]=tick
        alive=bool(world.alive[0]);burden=float(np.sum((1-after[14:16])**2)) if alive else 2.
        update=None
        if self.actor is not None and was:
            # No action-dependent evaluation label enters the reward. Death is
            # a fixed absorbing physical deficit, handled by the value target.
            nxt=0.
            if alive and world.t<world.total and len(self.actor.buffer)+1>=self.actor.c['block']:
                nd=self.candidate_decision(after)
                ne=[m.evidence(nd['features'],world.t) for m in self.models]
                nx=self.action_features(after,nd,ne)
                nxt=self.actor.forward(nx,nd['probability'],value_offset=self.value_prior(after))['value']
            actual=deepcopy(cache)
            update=self.actor.feedback(actual,action,-burden,alive,nxt,end=world.t>=world.total)
        self.last_action_learning=update
        return dict(clock_before=tick,clock_after=world.t,event_id=eid,before=before,
                    action=action,after=after,alive=alive,body_burden=burden,
                    decision={**{k:v for k,v in d.items() if k!='features'},
                              'base_choice':d['chosen'],'chosen':action,'final_probability':probability,
                              'residual':np.zeros(7) if cache is None else cache['residual'],
                              'error_history_before':errors_before,
                              'value_prediction':None if cache is None else cache['value'],
                              'work_state':None if cache is None else cache['h'],
                              'write_gate':None if cache is None else cache['u'],
                              'erase_or_reset_gate':None if cache is None else cache['e'],
                              'state_suppressed':False if cache is None else cache['suppress']},
                    movement=dict(success_before=relation_before,read_by_policy=self.relation_read,
                                  update=relation_event,prior_tau=1. if cache is None else cache['prior_tau']),
                    evidence_before=evidence,learning=dict(updated=was,residual=residual,
                        revision=revision,target=y,action_update=update),
                    native_payload=record,condition=self.condition,feature_schema=FEATURE_SCHEMA,
                    target_schema=TARGET_SCHEMA,action_feature_schema=ACTION_FEATURE_SCHEMA,
                    accounting=dict(primitive_actions=1,actor_parameters=0 if self.actor is None else self.actor.count,
                                    actor_updates=0 if self.actor is None else self.actor.updates),
                    transition_semantics=dict(moved=bool(before[:6].argmax()!=after[:6].argmax()),
                                              same_device_progress=float(y[4]),product=bool(y[5])))

    def state(self):
        s=super().state();s.update(condition=self.condition,action_feature_schema=self.action_feature_schema,
              actor=None if self.actor is None else self.actor.state(),errors=self.errors.copy(),
              error_counts=self.error_counts.copy(),suppress_until=self.suppress_until,
              last_action_learning=deepcopy(self.last_action_learning),migration=deepcopy(self.migration))
        s.update(relation_read=self.relation_read, relations=None if self.relations is None else self.relations.state())
        return s

    @classmethod
    def restore(cls,s):
        if s.get('target_schema')!=TARGET_SCHEMA or s.get('action_feature_schema')!=ACTION_FEATURE_SCHEMA:
            raise ValueError('incompatible whole-agent target/features; use explicit migration')
        obj=cls();obj.native=source_module('integrated').Session.from_state(s['native'])
        if 'relation_world_schema' in s['native']['world']:
            from .relation_world import RelationWorld
            obj.native.world=RelationWorld.from_state(s['native']['world'])
        elif 'whole_schema' in s['native']['world']:
            obj.native.world=WholeWorld.from_state(s['native']['world'])
        from .evidence_model import WindowedEvidenceModel
        obj.models=[WindowedEvidenceModel.restore(m) for m in s['models']]
        for k in ('mode','configuration','visits','last_use','feature_schema','target_schema','condition',
                  'action_feature_schema','errors','error_counts','suppress_until','last_action_learning','migration'):
            setattr(obj,k,deepcopy(s[k]))
        obj.actor=None if s['actor'] is None else ActionLearner.restore(s['actor'])
        obj.relation_read=s.get('relation_read',False)
        obj.relations=None if s.get('relations') is None else MovementRelations.restore(s['relations'])
        return obj
