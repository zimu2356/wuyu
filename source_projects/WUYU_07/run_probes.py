"""Observer-only counterfactual branches. No branch state returns to main life."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,time
from pathlib import Path
import numpy as np
from wuyu07.world import World
from wuyu07.agent import Agent,softmax
from wuyu07.student import Student
from wuyu07.experiment import restore,step
from wuyu07.state import write_tree
ROOT=Path(__file__).resolve().parent


def matched_world(w,site,phase,budget,body=(.75,.75)):
    c=copy.deepcopy(w);c.regime_override=phase;c.total=max(c.total,c.t+budget+1)
    c.pos[:]=site;c.stage[:]=0;c.inventory[:]=0;c.last[:]=4
    if c.alive[0]:c.body[:]=body
    return c


def trace_header(w,label):
    return dict(label=label,seed=int(w.seeds[0]),start_t=w.t,phase=w.phase(),total=w.total,
        position=int(w.pos[0]),body=w.body[0].tolist(),inventory=w.inventory[0].tolist(),alive=bool(w.alive[0]),actions=[],products=[],bodies=[])


def operate(w,a,site,phase,budget=24,label='',learning=False,supply_inputs=False):
    w=matched_world(w,site,phase,budget);a=copy.deepcopy(a);a.learn_enabled=learning
    if supply_inputs:w.inventory[:]=1
    trace=trace_header(w,label);trace['learning']=learning
    success=0;cost=budget
    for t in range(budget):
        if not w.alive[0]:break
        r=a.act(w.observation()[0],forced_goal=site,explore=False);out=w.step([r['action']]);a.feedback(r,w.observation()[0],out['reward'][0])
        trace['actions'].append(r['action']);trace['products'].append(int(w.last[0]));trace['bodies'].append(w.body[0].tolist())
        if w.last[0]<4 and w.pos[0]==site:success=1;cost=t+1;break
    trace.update(success=success,cost=cost,target_site=site,budget=budget)
    return success,cost,w,a,trace


def student_one(a,index):
    s=copy.deepcopy(a.state())
    for k in ['seeds','h','prev','rng']:s[k]=s[k][index:index+1].copy()
    s['memories']=[s['memories'][index]];return Student.from_state(s)


def autonomous(w,a,need,label,budget=40):
    body=[.9,.9];body[need]=.25;w=matched_world(w,int(w.seeds[0]%6),1,budget,body);a=copy.deepcopy(a)
    trace=trace_header(w,label);success=0;cost=budget;first=-1
    for t in range(budget):
        if not w.alive[0]:break
        r=step(w,a);trace['actions'].append(int(r['action'][0]));trace['products'].append(int(w.last[0]));trace['bodies'].append(w.body[0].tolist())
        if w.last[0]<2:
            first=int(w.last[0]);success=int(first==need);cost=t+1 if success else budget;break
    trace.update(success=success,cost=cost,target_need=need,first_body_product=first,budget=budget)
    return success,cost,trace


if __name__=='__main__':
    out=ROOT/'results';cfg=json.loads((ROOT/'CONFIG.json').read_text());traces=[];data={};start=time.perf_counter()
    def push(key,i,row):data.setdefault(key,[[],[],[]])[i].append(row)
    for init,initialization in enumerate(cfg['initializations']):
        w384,agents384=restore(out/f'checkpoint_full_{init}_384.npz')
        w1152,agents1152=restore(out/f'checkpoint_full_{init}_1152.npz')
        w1536,agents1536=restore(out/f'checkpoint_full_{init}_1536.npz')
        sw,students=restore(out/f'checkpoint_neural_student_{init}_384.npz')
        for j,seed in enumerate(w384.seeds):
            w=w384.one(j);mature=agents384[j];birth=Agent(seed,initialization)
            independent={key:[] for key in ['mature_success','birth_success','mature_cost','birth_cost','no_skill_success','no_skill_cost']}
            for site in range(6):
                for label,a in [('mature',mature),('birth',birth),('no_skill',copy.deepcopy(mature))]:
                    if label=='no_skill':a.mode='no_skill'
                    s,c,_,_,trace=operate(w,a,site,0,label=f'g1_{label}_{init}_{j}_{site}');traces.append(trace)
                    independent[label+'_success'].append(s);independent[label+'_cost'].append(c)
            for k,v in independent.items():push('independent_'+k,init,v)
            composite={key:[] for key in ['mature_success','birth_success','mature_cost','birth_cost']}
            for site in np.flatnonzero(w.products[0]<2):
                for label,a in [('mature',mature),('birth',birth)]:
                    s,c,_,_,trace=operate(w,a,int(site),1,40,f'g2_{label}_{init}_{j}_{site}');traces.append(trace)
                    composite[label+'_success'].append(s);composite[label+'_cost'].append(c)
            for k,v in composite.items():push('composite_'+k,init,v)
            correct=[];fixed=[];goals=[]
            for need in [0,1]:
                for label,target in [('adaptive',correct),('fixed',fixed)]:
                    a=copy.deepcopy(mature)
                    if label=='fixed':a.mode='fixed_value'
                    b=[.95,.95];b[need]=.2;c=matched_world(w,int(seed%6),0,1,b)
                    # Observer asks for the ultimate goal before any new action feedback.
                    obs=c.observation()[0];pos,_,req,_,_=a.decode(obs);a.requirement[pos]=req
                    goal,subgoal=a.plan(obs,allow_exploration=False)
                    ok=int(c.products[0,goal]==need and c.alive[0]);target.append(ok)
                    goals.append(dict(need=need,label=label,goal=goal,goal_product=int(c.products[0,goal])))
            push('revalue_adaptive',init,correct);push('revalue_fixed',init,fixed)
            traces.append(dict(label=f'g3_{init}_{j}',seed=int(seed),decisions=goals))
            # Same pre-change individual; expose only the changed device through actions.
            prew=w1152.one(j);prea=agents1152[j];changed=int(prew.changed[0])
            bs,bc,_,_,trace=operate(prew,prea,changed,3,24,f'g4_before_{init}_{j}',supply_inputs=True);traces.append(trace)
            _,_,pw,pa,trace=operate(prew,prea,changed,3,16,f'g4_practice_{init}_{j}',learning=True,supply_inputs=True);traces.append(trace)
            as_,ac,_,_,trace=operate(pw,pa,changed,3,24,f'g4_after_{init}_{j}',supply_inputs=True);traces.append(trace)
            push('change_before_success',init,[bs]);push('change_after_success',init,[as_]);push('change_before_cost',init,[bc]);push('change_after_cost',init,[ac])
            before=[];after=[]
            for site in range(6):
                if site==changed:continue
                s,_,_,_,trace=operate(prew,prea,site,2,24,f'g4_retention_before_{init}_{j}_{site}',supply_inputs=True);traces.append(trace);before.append(s)
                s,_,_,_,trace=operate(w1536.one(j),agents1536[j],site,3,24,f'g4_retention_after_{init}_{j}_{site}',supply_inputs=True);traces.append(trace);after.append(s)
            push('retention_before',init,before);push('retention_after',init,after)
            # Semantic predictions: all buttons, both states, observer-only truth.
            pred_error=[];initial_error=[]
            for site in range(6):
                for state in range(2):
                    for button in range(3):
                        target=state+1 if w.codes[0,site,state]==button else 0;onehot=np.eye(3)[target]
                        pred_error.append(float(((softmax(mature.transition[site,state,button])-onehot)**2).mean()))
                        initial_error.append(float(((softmax(birth.transition[site,state,button])-onehot)**2).mean()))
            push('semantic_mature_brier',init,pred_error);push('semantic_birth_brier',init,initial_error)
            sm=student_one(students,j);sb=Student(copy.deepcopy(sm.net),[seed],initialization);sr=copy.deepcopy(sm);sr.memories=[Agent(seed,initialization)]
            for label,a in [('mature',sm),('birth',sb),('memory_reset',sr)]:
                ss=[];cc=[]
                for need in [0,1]:
                    s,c,trace=autonomous(sw.one(j),a,need,f'student_{label}_{init}_{j}_{need}');traces.append(trace);ss.append(s);cc.append(c)
                push('student_'+label+'_success',init,ss);push('student_'+label+'_cost',init,cc)
        print(json.dumps(dict(probes_init=init,seconds=time.perf_counter()-start)),flush=True)
    arrays={k:np.asarray(v) for k,v in data.items()}
    write_tree(out/'probes_raw.npz',dict(arrays=arrays,traces=traces))
    summary={k:dict(mean=float(v.mean()),by_initialization=v.mean(axis=tuple(range(1,v.ndim))).tolist()) for k,v in arrays.items()}
    (out/'probes_means.json').write_text(json.dumps(summary,indent=2));print('PROBES_COMPLETE',flush=True)
