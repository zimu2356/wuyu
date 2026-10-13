"""Scientific figures from complete, audited world-level evidence."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MODES=('A','R00','R10','R01','R11','RT')
COLORS={'A':'#9aabb2','R00':'#526979','R10':'#147d73','R01':'#c5a36b','R11':'#9161a8','RT':'#4f9dc5'}

def get(a,case,name,key):
    return next(v for v in a['contrasts'] if (v['case'],v['contrast'],v['metric'])==(case,name,key))

def save(fig,name):
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=180,facecolor='white',bbox_inches='tight')
    fig.savefig(directory/(name+'.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)

def main():
    p=ROOT/'experiments/relation_validation'
    evidence=json.loads((p/'evidence_readout.json').read_text())
    assert evidence['formal_lives']==3904
    a=json.loads((ROOT/'experiments/relation_holdout_main/analysis.json').read_text())
    s=json.loads((ROOT/'experiments/relation_holdout_scenarios/analysis.json').read_text())
    long=json.loads((ROOT/'experiments/relation_holdout_long/analysis.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#273e49','text.color':'#273e49','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(12.5,9.3),gridspec_kw={'hspace':.43,'wspace':.28})
    fig.subplots_adjust(bottom=.1,top=.88)
    x=np.arange(len(MODES))
    for ax,case,key,title in ((axes[0,0],'r_combined','challenge_burden','Combined challenge: body loss'),
                              (axes[1,0],'r_stable','total_burden','Stable world: whole-life body loss')):
        vals=[a['averages'][case][m][key] for m in MODES]
        ax.bar(x,vals,color=[COLORS[m] for m in MODES],width=.65)
        ax.set_xticks(x,MODES);ax.set_ylim(0,max(vals)*1.22);ax.set_title(title,loc='left')
        ax.set_ylabel('Mean absorbing body deficit (lower is better)')
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.3f}' if max(vals)>.1 else f'{v:.4f}',ha='center',fontsize=9)
    ax=axes[0,1]
    surv=[a['averages']['r_combined'][m]['survivors'] for m in MODES]
    ax.bar(x,np.array(surv)/256,color=[COLORS[m] for m in MODES],width=.65)
    ax.set_xticks(x,MODES);ax.set_ylim(0,1.1);ax.set_ylabel('Fraction alive at tick 2,560')
    ax.set_title('Combined challenge: all assigned lives',loc='left')
    for i,v in enumerate(surv):ax.text(i,v/256+.025,f'{v}/256',ha='center',fontsize=9)
    ax=axes[1,1]
    names=('R10-R00','R01-R00','R11-R10','R11-RT','R10-A')
    for i,name in enumerate(names):
        d=get(a,'r_combined',name,'challenge_burden');lo,hi=d['ci95'];color='#147d73' if i==0 else '#526979'
        ax.errorbar(d['delta'],i,xerr=[[d['delta']-lo],[hi-d['delta']]],fmt='o',color=color,capsize=4,lw=2)
    ax.axvline(0,color='#a7b4b9',lw=1);ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
    ax.set_title('World-paired effects, 95% bootstrap intervals',loc='left')
    ax.set_xlabel('Difference in challenge body loss (left is better)')
    fig.suptitle('WUYU 1.2 | Main holdout: 128 worlds × 2 fixed initializations',fontsize=15,y=.98)
    fig.text(.06,.015,'R10−R00 is the predeclared primary comparison. All other intervals are descriptive and unadjusted. Death remains loss 2.',fontsize=9)
    save(fig,'relation_results')

    fig,axes=plt.subplots(1,2,figsize=(13.4,5.2),gridspec_kw={'wspace':.4})
    fig.subplots_adjust(bottom=.2,top=.87,left=.18,right=.97)
    cases=('r_block','r_noise','r_reopen','r_relocate')
    for i,case in enumerate(cases):
        d=get(s,case,'R10-R00','challenge_burden');lo,hi=d['ci95']
        axes[0].errorbar(d['delta'],i,xerr=[[d['delta']-lo],[hi-d['delta']]],fmt='o',color='#ad7045' if case=='r_noise' else '#147d73',capsize=4)
        for mode,dy in (('R00',-.12),('R10',.12)):
            v=s['averages'][case][mode]
            axes[1].scatter(v['survivors']/v['lives'],i+dy,color=COLORS[mode],label=mode if i==0 else None,s=45)
    axes[0].axvline(0,color='#a7b4b9',lw=1);axes[0].set_yticks(range(4),['Blocked route','12% movement noise','Route reopens','Obstacle relocates'])
    axes[0].set_xlabel('R10−R00 challenge loss; 95% paired CI');axes[0].set_title('Scenario checks: 16 worlds each',loc='left')
    axes[1].set_xlim(-.04,1.12);axes[1].set_yticks(range(4),cases);axes[1].set_xlabel('Fraction alive at tick 2,560')
    axes[1].set_title('32 assigned lives per condition',loc='left');axes[1].legend(frameon=False)
    for ax in axes:ax.invert_yaxis()
    noise=get(s,'r_noise','R10-R00','challenge_burden')
    axes[0].annotate(f"Noise: {noise['delta']:+.6f}\n95% CI [{noise['ci95'][0]:.6f}, {noise['ci95'][1]:.6f}]",
        xy=(noise['delta'],1),xytext=(-.20,1.50),fontsize=8,color='#8a592e',
        arrowprops=dict(arrowstyle='-',color='#bba28d'),bbox=dict(facecolor='white',edgecolor='none',alpha=.95))
    fig.text(.05,.01,'Secondary checks; no multiple-comparison correction. A change in blocked attempts alone is not a body-benefit result.',fontsize=9)
    save(fig,'relation_scenarios')

    with np.load(p/'mean_timelines.npz') as t:
        fig,axes=plt.subplots(2,2,figsize=(12.5,8),gridspec_kw={'hspace':.34,'wspace':.27})
        fig.subplots_adjust(bottom=.13,top=.88)
        for col,(cohort,modes,title) in enumerate((('relation_holdout_main',('A','R00','R10','R11'),'Main holdout: 256 lives / condition'),
                                                  ('relation_holdout_long',('R00','R10'),'Long lives: 32 lives / condition'))):
            for mode in modes:
                for row,metric in enumerate(('burden','alive')):
                    vals=t[cohort+'__r_combined__'+mode+'__'+metric]
                    # Nonoverlapping block means; death stays in every block.
                    y=vals.reshape(-1,32).mean(1);xx=np.arange(len(y))*32+15.5
                    axes[row,col].plot(xx,y,color=COLORS[mode],label=mode,lw=2)
            for row in range(2):
                ax=axes[row,col];ax.axvline(1024,color='#9badb6',ls='--',lw=1)
                if col==1:ax.axvline(2560,color='#c1b7a4',ls=':',lw=1)
                ax.set_xlabel('Continuous physical tick');ax.set_ylim(0,2.05 if row==0 else 1.03)
            axes[0,col].set_title(title,loc='left');axes[0,col].legend(frameon=True,facecolor='white',framealpha=.95,ncol=2,loc='upper left')
        axes[0,0].set_ylabel('Mean body deficit, including death')
        axes[1,0].set_ylabel('Fraction alive, all assigned lives')
        fig.text(.05,.01,'32-tick block means. Later challenge events vary in time and order across worlds; long lives have no added events after tick 2,560.',fontsize=9)
        save(fig,'relation_timeline')
    print('Saved three PNG/SVG figure pairs from audited evidence.')

if __name__=='__main__':main()
