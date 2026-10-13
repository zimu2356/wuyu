"""Copy only the old evidence required for the declared diagnostic."""
from pathlib import Path
import json
import shutil
import numpy as np
from wuyu03.stats import digest

ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'WUYU_02'
dest=ROOT/'inputs'
out=dest/'legacy.npz'
if out.exists():raise SystemExit('Refusing to overwrite legacy inputs')
idx=np.r_[np.arange(64),np.arange(128,192)]
with np.load(old/'results/checkpoint_matched.npz') as d:
    q=d['q'][idx].copy()
with np.load(old/'results/development_raw.npz') as d:
    data={k:d['common__'+k][:,idx] for k in ['action','reward','cue']}
    # checkpoint_matched is BEFORE the common sequence; replay its real updates.
    from wuyu03.control import Control,encode
    c=Control(len(idx));c.q=q.reshape(len(idx),25,5).copy()
    charge=np.zeros((len(idx),2),dtype=np.int16)
    states=[];afters=[]
    for t in range(len(data['action'])):
        s=encode(charge).copy();a=data['action'][t]
        for j in range(2):
            charge[(a==j+1)&(charge[:,j]>0),j]-=1
            charge[a==j+3,j]=4
        sn=encode(charge).copy()
        c.backup(s,a,data['reward'][t],sn)
        states.append(s);afters.append(sn)
    data.update(q=c.q,state=np.asarray(states),after=np.asarray(afters),old_indices=idx)
np.savez_compressed(out,**data)
shutil.copy2(old/'wuyu02/workshop.py',dest/'legacy_world_source.py')
(dest/'provenance.json').write_text(json.dumps({
 'source_checkpoint_sha256':digest(old/'results/checkpoint_matched.npz'),
 'source_trajectories_sha256':digest(old/'results/development_raw.npz'),
 'source_world_sha256':digest(old/'wuyu02/workshop.py'),
 'note':'Common-stage Q reconstructed from pre-common checkpoint and actual recorded updates; first 64 historical pairs.'
},indent=2)+'\n')
print('Prepared legacy cohort',data['q'].shape)
