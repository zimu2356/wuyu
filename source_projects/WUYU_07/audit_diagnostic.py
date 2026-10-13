import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,re,zipfile
from pathlib import Path
import numpy as np
from wuyu07.world import World
from wuyu07.state import read_tree
from analyze import stats
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    out=ROOT/'results';s=read_tree(out/'student_diagnostic_raw.npz');recomputed={k:np.zeros_like(v) for k,v in s['arrays'].items()};steps=0
    for tr in s['traces']:
        w=World([tr['seed']],tr['total']);w.t=tr['start_t'];w.regime_override=tr['phase'];w.pos[:]=tr['position'];w.body[:]=tr['body'];w.inventory[:]=tr['inventory'];w.alive[:]=tr['alive']
        first=-1;cost=tr['budget']
        for t,a in enumerate(tr['actions']):
            w.step([a]);assert int(w.last[0])==tr['products'][t];np.testing.assert_array_equal(w.body[0],tr['bodies'][t]);steps+=1
            if first<0 and w.last[0]<2:first=int(w.last[0]);cost=t+1
        success=int(first==tr['target_need']);cost=cost if success else tr['budget'];assert success==tr['success'] and cost==tr['cost']
        m=re.fullmatch(r'(.+)_(\d+)_(\d+)_(\d+)',tr['label']);key,i,j,need=m.groups();recomputed[key][int(i),int(j),int(need)]=success
    summary=json.loads((out/'student_diagnostic.json').read_text())
    for k,v in recomputed.items():
        np.testing.assert_array_equal(v,s['arrays'][k]);assert stats(v)==summary[k]
    with zipfile.ZipFile(out/'student_diagnostic_raw.npz') as z:assert z.testzip() is None
    audit=dict(traces=len(s['traces']),steps=steps,metrics_recomputed=True,physics_replayed=True,crc_passed=True)
    (out/'diagnostic_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
