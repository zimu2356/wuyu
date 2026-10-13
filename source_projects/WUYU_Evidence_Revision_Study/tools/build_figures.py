"""Post-freeze descriptive scientific figures; never changes endpoints or trajectories."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from storage import read_events,read_gz_json,write_json

ROOT=Path(__file__).resolve().parents[1]
COLORS={'B0':'#9ba4b0','BE':'#3178b5','BU':'#c4862f','BEU':'#128c80','BR':'#8455a3'}
NAMES={'stable':'Stable','local_swap':'Local tool swap','opportunity':'Silent opportunity','noise_pulse':'Temporary noise','visible_shock':'Visible state shock','recurrence':'A → B → A'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','savefig.facecolor':'white','figure.facecolor':'white','axes.labelcolor':'#34425a',
                     'text.color':'#17263a','xtick.color':'#485971','ytick.color':'#485971'})

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    (ROOT/'figures').mkdir(exist_ok=True);c=json.loads((ROOT/'formal_protocol.json').read_text())
    analysis=json.loads((ROOT/'holdout/ANALYSIS.json').read_text());rows=json.loads((ROOT/'holdout/summaries.json').read_text())
    look={(r['seed'],r['case'],r['method']):r for r in rows};seeds=sorted({r['seed'] for r in rows});n=len(seeds)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),layout='constrained')
    for ax,case in zip(axes.flat,c['cases']):
        values=[np.mean([look[s,case,m]['post_loss'] for s in seeds]) for m in c['methods']]
        bars=ax.bar(c['methods'],values,color=[COLORS[m] for m in c['methods']],width=.66)
        ax.set_title(NAMES[case]);ax.set_ylabel('Mean body loss');ax.set_ylim(0,max(values)*1.3)
        for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.04,f'{v:.4f}',ha='center',fontsize=9)
        ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'All six conditions · {n} paired seeds · 512 post-boundary actions\nMeans only; each panel uses its own vertical scale',fontsize=15)
    save(fig,'six_conditions')
    x=np.array([look[s,'local_swap','BE']['post_loss'] for s in seeds]);y=np.array([look[s,'local_swap','BEU']['post_loss'] for s in seeds]);br=np.array([look[s,'local_swap','BR']['post_loss'] for s in seeds])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,xx,yy,xlabel,ylabel in [(axes[0],x,y,'BE loss','BEU loss'),(axes[1],br,y,'BR loss','BEU loss')]:
        lim=max(xx.max(),yy.max())*1.04;ax.plot([0,lim],[0,lim],ls='--',c='#a6aebc',lw=1);ax.scatter(xx,yy,s=21,color=COLORS['BEU'],alpha=.65)
        ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(0,lim),ylim=(0,lim));ax.set_aspect('equal')
    axes[0].set_title('Main comparison: one dot per seed');axes[1].set_title('Random retesting reference')
    axes[2].hist(x-y,bins=18,color=COLORS['BEU'],alpha=.8,edgecolor='white');axes[2].axvline(.02,ls='--',color='#bd672d',label='Useful threshold: 0.02')
    axes[2].axvline(0,color='#617085',lw=1);axes[2].set(xlabel='BE − BEU (positive favors BEU)',ylabel='Seeds',title='Paired differences');axes[2].legend(fontsize=8)
    p=analysis['primary'];fig.suptitle(f"Main mean difference {p['mean']:.4f}, seed bootstrap 95% CI [{p['low']:.4f}, {p['high']:.4f}]",fontsize=14)
    save(fig,'paired_results')
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained');curves={};selected_errors={}
    relative=np.arange(-256,1112)
    for ax,case in zip(axes.flat,c['cases']):
        for method in c['methods']:
            series=[];errors=[]
            for s in seeds:
                r=look[s,case,method];e=read_events(ROOT/'holdout'/f'{s}_{case}_{method}.npy.gz');loss=np.sum(e['after'][:,:2]**2,axis=1)
                series.append(loss[r['tau']-256:r['tau']+1112]);errors.append(float(np.mean(e['residual'][r['tau']:r['tau']+512]**2/2)))
            mean=np.mean(series,axis=0);smooth=np.convolve(mean,np.ones(32)/32,mode='valid')
            ax.plot(relative[31:],smooth,label=method,color=COLORS[method],lw=1.5);curves[case+':'+method]=mean.tolist();selected_errors[case+':'+method]=errors
        ax.axvline(0,c='#637083',ls='--',lw=1)
        if case=='recurrence': ax.axvline(600,c='#637083',ls=':',lw=1)
        ax.set_title(NAMES[case]);ax.set_xlabel('Action index relative to boundary');ax.set_ylabel('Mean body loss');ax.set_ylim(bottom=0);ax.yaxis.grid(True,alpha=.15)
    axes[0,0].legend(ncol=5,fontsize=8,loc='upper right');fig.suptitle('Every seed retained · mean trajectories with a trailing 32-action display average',fontsize=14)
    save(fig,'aligned_trajectories');write_json(ROOT/'figures/DESCRIPTIVE_CURVES.json',dict(relative_ticks=relative.tolist(),curves=curves,actual_selected_prequential_mse=selected_errors))
    d=json.loads((ROOT/'development_diagnostics_v2/DIAGNOSTICS_SUMMARY.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    panels=[(axes[0,0],['Policy-ranked','Round-robin','Oldest'],list(d['sampling'].values()),'Equal 16 tool probes + 48 WAIT steps','Following 128-action body loss'),
            (axes[0,1],['FIFO','Calendar decay','Local revision'],[d['same_data_mse'][k] for k in ('FIFO','calendar_decay','local_revision')],'Exactly the same real transition stream','Pre-update body prediction MSE'),
            (axes[1,0],['Learned prediction','Noiseless reference'],[d['coordination'][k] for k in ('learned','truth')],'Prediction replacement only','128-action body loss')]
    # JSON key order is alphabetical; sampling order is explicit here.
    panels[0]=(axes[0,0],['Policy-ranked','Round-robin','Oldest'],[d['sampling'][k] for k in ('policy_ranked_tools','round_robin','oldest')],panels[0][3],panels[0][4])
    for ax,names,values,title,ylabel in panels:
        ax.bar(names,values,color=['#748ba1','#128c80','#c4862f'][:len(names)],width=.6);ax.set(title=title,ylabel=ylabel,ylim=(0,max(values)*1.25))
        for i,v in enumerate(values): ax.text(i,v+max(values)*.04,f'{v:.4f}',ha='center',fontsize=9)
    ax=axes[1,1];groups=['recorded_support','material_interpolation','body_fatigue_extrapolation','boundary'];xx=np.arange(len(groups));width=.35
    for i,(kind,color) in enumerate([('conditional_mse','#3178b5'),('mean_mse','#c4862f')]):
        ax.bar(xx+(i-.5)*width,[d['predictor_before'][g][kind] for g in groups],width,label=kind.replace('_mse',''),color=color)
    ax.set_xticks(xx,['Recorded\nsupport','Material\ninterpolation','Body/fatigue\nextrapolation','Boundary']);ax.set_yscale('log');ax.set_ylabel('Body prediction MSE (log scale)');ax.set_title('Same samples; different prediction rules');ax.legend(fontsize=8)
    fig.suptitle('Development diagnostics only · 32 seeds · distinct questions, distinct denominators',fontsize=14);save(fig,'mechanism_diagnostics')
    diag=read_gz_json(ROOT/'development_diagnostics_v2/development_diagnostics.json.gz');supp=[]
    for r in diag:
        for mode,s in r['sampling'].items():
            losses=[sum(b*b for b in e['after']['body']) for e in s['sampling_events']]+[e['loss'] for e in s['evaluation']['events']]
            supp.append(dict(seed=r['seed'],mode=mode,all_192_step_loss=float(np.mean(losses)),sampling_steps=s['primitive_probe_steps'],active_probes=s['active_probes']))
    write_json(ROOT/'figures/EQUAL_SAMPLING_TOTAL_COST.json',supp)
    print(json.dumps(dict(figures=4,seeds=n,diagnostic_budget_checks=all(r['sampling_steps']==64 and r['active_probes']==16 for r in supp))))

if __name__=='__main__': main()
