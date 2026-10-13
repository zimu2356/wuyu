"""Run fixed selection and main experiments, refusing to overwrite evidence."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
import numpy as np
from wuyu03 import core,legacy
from wuyu03.stats import interval,source_hashes

ROOT=Path(__file__).resolve().parent
PHASES=[('A',0,False),('B',1,False),('A_return',0,False),
        ('A_reversed',0,True),('A_restored',0,False),('B_return',1,False)]
MODES=['online','repeat','replay','frozen','no_memory']


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def pack_outputs(items):return {k:np.stack([v[k] for v in items]) for k in items[0]}


def metrics(data,tail):
    valid=data['valid'][-tail:]
    return {'accuracy':data['correct'][-tail:].sum(axis=0)/np.maximum(valid.sum(axis=0),1),
            'reward':data['reward'][-tail:].mean(axis=0),
            'maintenance':(data['action'][-tail:]>=3).mean(axis=0),
            'tool_use':((data['action'][-tail:]>=1)&(data['action'][-tail:]<=2)&valid).mean(axis=0),
            'valid_answers':valid.sum(axis=0)}


def select_rate(out):
    seeds=np.arange(720000,720008);ev=core.events(seeds,2048)
    candidates={};raw={};start=time.perf_counter()
    for rate in [.005,.02,.05]:
        a=core.Agent(seeds,'online',rate);w=core.World(seeds);scores=[]
        for stage,context in enumerate([0,1]):
            items=[]
            for t in range(stage*1024,(stage+1)*1024):
                nxt=1 if t==1023 else context
                items.append(core.step(w,a,ev,t,context,next_context=nxt))
            data=pack_outputs(items)
            m=metrics(data,256);scores.append(m['accuracy'])
            for k,v in data.items():raw[f'{rate}__{stage}__{k}']=v
        candidates[str(rate)]={'score':float(np.mean(scores)),
                               'stage_tail_accuracy':np.asarray(scores).tolist()}
        print('validation rate',rate,'score',candidates[str(rate)]['score'],flush=True)
    chosen=max([.005,.02,.05],key=lambda rate:(candidates[str(rate)]['score'],-rate))
    result={'selected_rate':chosen,'candidates':candidates,'seconds':time.perf_counter()-start,
            'selection':'Mean of A/B last-256 valid-answer accuracy; 8 independent validation seeds'}
    np.savez_compressed(out/'selection_raw.npz',**raw);dump(out/'selection.json',result)
    return chosen


def run_core(out,rate):
    start=time.perf_counter();seeds=np.arange(730000,730032);ev=core.events(seeds,12288)
    raw={};summary={};counts={};evaluation={}
    for mode in MODES:
        a=core.Agent(seeds,mode,rate);w=core.World(seeds)
        mode_start=time.perf_counter()
        for stage,(name,context,reverse) in enumerate(PHASES):
            items=[]
            for t in range(stage*2048,(stage+1)*2048):
                nxt=PHASES[stage+1][1] if t==(stage+1)*2048-1 and stage<5 else context
                items.append(core.step(w,a,ev,t,context,reverse,nxt,
                                       frozen=(mode=='frozen' and stage>0)))
            data=pack_outputs(items);tag=f'{mode}__{stage}'
            for k,v in data.items():raw[tag+'__'+k]=v
            m=metrics(data,512);early=metrics({k:v[:128] for k,v in data.items()},128)
            summary[tag]={'name':name,'tail':{k:interval(v) for k,v in m.items()},
                          'early':{k:interval(v) for k,v in early.items()}}
            for ctxt in [0,1]:
                result=core.evaluate(w,a,seeds,ctxt,reverse=(stage==3 and ctxt==0),
                                     offset=1000000+stage*10000+ctxt*1000)
                etag=tag+f'__eval{ctxt}'
                evaluation[etag]={k:interval(v) for k,v in result.items()}
                for k,v in result.items():raw[etag+'__'+k]=v
            if stage in [1,3,5]:
                core.save_checkpoint(out/f'checkpoint_{mode}_stage{stage}.npz',w,a)
            print('B completed',mode,name,'accuracy',round(m['accuracy'].mean(),4),
                  'reward',round(m['reward'].mean(),4),flush=True)
        counts[mode]={'seconds':time.perf_counter()-mode_start,'events_per_individual':12288,
                      'network_updates':interval(a.net.updates),'controller_real_updates':a.ctrl.real_updates,
                      'controller_extra_backups':a.ctrl.extra_updates,
                      'network_parameters_per_individual':a.net.parameter_count,
                      'q_values_per_individual':a.ctrl.q[0].size,
                      'hidden_state_values':a.net.k,'max_replay_events':a.memory.capacity}
    def e(mode,stage,ctx):return raw[f'{mode}__{stage}__eval{ctx}__accuracy']
    comparisons={
        'memory_contribution_A':interval(e('online',0,0)-e('no_memory',0,0)),
        'reversal_online_minus_frozen':interval(e('online',3,0)-e('frozen',3,0)),
        'replay_minus_repeat_retained_A_after_B':interval(e('replay',1,0)-e('repeat',1,0)),
        'online_A_retention_change_after_B':interval(e('online',1,0)-e('online',0,0)),
        'repeat_A_retention_change_after_B':interval(e('repeat',1,0)-e('repeat',0,0)),
        'replay_A_retention_change_after_B':interval(e('replay',1,0)-e('replay',0,0)),
        'replay_minus_repeat_new_B_after_B':interval(e('replay',1,1)-e('repeat',1,1))}
    gates={'B1_learned_decoding':summary['online__0']['tail']['accuracy']['low']>.55,
           'B2_memory_used':comparisons['memory_contribution_A']['low']>.05,
           'B3_reversal_adaptation':comparisons['reversal_online_minus_frozen']['low']>.05,
           'B4_replay_retention_advantage':comparisons['replay_minus_repeat_retained_A_after_B']['low']>.03,
           'B5_exact_checkpoint_restore':'verified by mandatory pre-run test'}
    np.savez_compressed(out/'core_raw.npz',**raw)
    return {'stages':summary,'evaluations':evaluation,'comparisons':comparisons,'gates':gates,
            'costs':counts,'seconds':time.perf_counter()-start,'test_individuals_per_method':32}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/main')
    args=parser.parse_args();out=Path(args.output).resolve()
    if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
    out.mkdir(parents=True)
    validation=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                              cwd=ROOT,capture_output=True,text=True)
    (out/'validation.log').write_text(validation.stdout+validation.stderr)
    if validation.returncode:raise SystemExit('Validation failed; see validation.log')
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
              'numpy':np.__version__,'platform':platform.platform(),'device':'CPU',
              'source_sha256':source_hashes(ROOT),'protocol':'PROTOCOL.md version 1.0'}
    dump(out/'manifest.json',manifest);start=time.perf_counter()
    result_a=legacy.run(ROOT,out);dump(out/'legacy.json',result_a)
    rate=select_rate(out)
    result_b=run_core(out,rate);dump(out/'core.json',result_b)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                    selected_rate=rate,source_unchanged=source_hashes(ROOT)==manifest['source_sha256'])
    dump(out/'manifest.json',manifest)
    print(json.dumps({'A':result_a['comparisons'],'B':result_b['comparisons'],'gates':result_b['gates']},indent=2),flush=True)


if __name__=='__main__':main()
