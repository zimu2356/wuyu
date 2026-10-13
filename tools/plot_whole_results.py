"""Standalone scientific figures from complete frozen holdout results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(directory,destination):
    root=Path(directory);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    assert (root/'RUN_COMPLETE.json').exists()
    p=json.loads((root/'protocol.json').read_text());a=json.loads((root/'analysis.json').read_text())
    rows=[json.loads(q.read_text()) for q in sorted((root/'summaries').glob('*.json'))]
    colors=['#52677c','#bc8244','#8876a5','#267f7c'];modes=list('ABCD');rng=np.random.default_rng(81233)
    draw=rng.integers(0,len(p['seeds']),(10000,len(p['seeds'])))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#283747','xtick.color':'#283747','ytick.color':'#283747'})
    fig,axs=plt.subplots(2,2,figsize=(12,8));fig.patch.set_facecolor('white')
    def points(ax,case,metric,title):
        means=[];lo=[];hi=[]
        for mode in modes:
            v=np.array([np.mean([r[metric] for r in rows if r['seed']==seed and r['case']==case and r['mode']==mode]) for seed in p['seeds']])
            ci=np.quantile(v[draw].mean(1),[.025,.975]);means.append(v.mean());lo.append(v.mean()-ci[0]);hi.append(ci[1]-v.mean())
        for i,m in enumerate(means):
            ax.errorbar(i,m,yerr=np.array([[lo[i]],[hi[i]]]),fmt='o',color=colors[i],capsize=5,markersize=8)
            ax.annotate(f'{m:.4f}',(i,m),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(range(4),modes);ax.set_xlim(-.6,3.6);ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylabel('Mean bodily deficit (lower is better)');ax.grid(axis='y',alpha=.18);ax.set_ylim(bottom=0)
    points(axs[0,0],'challenge','challenge_burden','a. Challenge loss, fixed windows')
    points(axs[1,1],'stable','total_burden','d. Stable-world retention')
    ax=axs[0,1];cs=['D-A','B-A','C-A','D-C']
    for i,c in enumerate(cs):
        r=next(r for r in a['contrasts'] if r['case']=='challenge' and r['metric']=='challenge_burden' and r['contrast']==c)
        d=r['delta'];lo,hi=r['ci95'];ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color='#267f7c' if c=='D-A' else '#52677c',capsize=5)
    ax.axvline(0,color='#888',lw=1);ax.axvline(-.01,color='#aaa',lw=.8,ls=':')
    ax.set_yticks(range(len(cs)),cs);ax.invert_yaxis();ax.set_xlabel('Paired change in deficit (negative is better)')
    ax.set_title('b. World-paired effects, 95% intervals',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.18)
    ax=axs[1,0]
    for i,mode in enumerate(modes):
        n=a['averages']['challenge'][mode]['lives'];surv=a['averages']['challenge'][mode]['survivors']
        ax.bar(i,surv/n,color=colors[i],width=.55);ax.text(i,surv/n+.03,f'{surv}/{n}',ha='center')
    ax.set_xticks(range(4),modes);ax.set_ylim(0,1.14);ax.set_ylabel('Fraction alive at tick 2560')
    ax.set_title('c. Challenge survival',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.18)
    fig.suptitle('WUYU whole-system experiment | 64 paired worlds × 2 fixed initializations',fontweight='bold',fontsize=14)
    fig.text(.5,.015,'A: corrected baseline   B: action learning   C: GRU state   D: write / retain / erase state\nBootstrap unit: world; other than D-A, intervals are descriptive and unadjusted.',ha='center',fontsize=9,color='#52677c')
    fig.tight_layout(rect=[0,.065,1,.95]);fig.savefig(dest/'whole_results.png',dpi=180);fig.savefig(dest/'whole_results.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4.5));means={}
    for mode,color in zip(modes,colors):
        tracks=[]
        for r in rows:
            if r['mode']==mode and r['case']=='challenge':
                path=root/'traces'/f"{r['seed']}_i{r['initialization']}_challenge_{mode}.npz"
                tracks.append(np.load(path)['trace']['burden'])
        curve=np.mean(tracks,axis=0);means[mode]=curve.tolist()
        ax.plot(np.arange(0,len(curve),16)+7.5,curve.reshape(-1,16).mean(1),label=mode,color=color,lw=2)
    labels=['Prerequisite','Blocked route / recipe','Need interruption','Benefit removed']
    for x,label in zip((1024,1408,1792,2176),labels):
        ax.axvline(x,lw=.8,color='#a4a9af',ls='--');ax.text(x+10,1.9,label,rotation=90,va='top',fontsize=9,color='#65727e')
    ax.set(xlabel='Continuous lifetime tick',ylabel='Mean bodily deficit; death = 2',ylim=(0,2.05),xlim=(0,2560))
    ax.set_title('Continuous body history | fixed 16-step bins, all lives included',loc='left',fontweight='bold');ax.legend(ncol=4,loc='upper left');ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(dest/'whole_timeline.png',dpi=180);fig.savefig(dest/'whole_timeline.svg');plt.close(fig)
    (dest/'whole_timeline_values.json').write_text(json.dumps(means)+'\n')

if __name__=='__main__':render(sys.argv[1],sys.argv[2])
