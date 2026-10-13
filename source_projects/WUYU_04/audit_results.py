"""Read-only recomputation of material reported results from delivered raw arrays."""
import json
from pathlib import Path
import numpy as np
from wuyu04.common import hashes,write_json
from wuyu04.latent import load

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
raw=np.load(R/'hidden_raw.npz',allow_pickle=False)
summary=json.loads((R/'hidden.json').read_text())
checks={}; reference=None
for mode in ('single','oracle','passive','active'):
    correct=raw[mode+'__correct'];answer=raw[mode+'__answer'];cue=raw[mode+'__cue']
    recovered=np.where(correct,answer,1-answer)
    if reference is None:reference=(cue,recovered)
    else:
        np.testing.assert_array_equal(reference[0],cue)
        np.testing.assert_array_equal(reference[1],recovered)
    np.testing.assert_array_equal(raw[mode+'__reward'],2*correct.astype(float)-1)
    np.testing.assert_allclose(raw[mode+'__net_reward'],2*correct.astype(float)-1-.1*raw[mode+'__query'])
    b=raw[mode+'__belief_after']
    np.testing.assert_allclose(b.sum(axis=2),1,atol=1e-14)
    assert np.isfinite(b).all() and (b>=0).all()
    value=correct.reshape(9,512,32)[[2,4,5],:32].mean()
    assert abs(value-summary['methods'][mode]['old_return_first32']['mean'])<1e-14
    w,a=load(R/f'checkpoint_{mode}_final.npz')
    if mode!='single':np.testing.assert_array_equal(a.w[:,:2],a.apprentice_bank)
    checks[mode]=dict(paired_cues_and_targets=True,feedback_and_cost_exact=True,belief_normalized=True,
                      ordinary_hidden_events=correct.shape[0],individuals=correct.shape[1])
old=json.loads((R/'frozen_sources.json').read_text());current=hashes(ROOT)
checks['frozen_sources_unchanged']=all(current.get(k)==v for k,v in old.items())
assert checks['frozen_sources_unchanged']
d=np.load(R/'diagnostic_raw.npz',allow_pickle=False)
ds=json.loads((R/'diagnostic.json').read_text())
for mode in ('joint','sequential','dense','gated','repeat','stale','burnin','count'):
    ev=d[mode+'__evaluations'][-1].mean(axis=1)
    np.testing.assert_allclose(ev,[x['mean'] for x in ds['methods'][mode]['final']],atol=1e-14)
checks['diagnostic_means_recomputed']=True
checks['console_capture_note']='console.log retains only the first two stdout lines; complete results are in NPZ/JSON, not reconstructed stdout.'
write_json(R/'audit.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
