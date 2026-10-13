"""Recover the four initially truncated archives without retraining or retuning."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,shutil,hashlib,zipfile,tempfile
import numpy as np
from wuyu06.network import Network
from wuyu06.state import read_tree
from wuyu06.experiment import rollout

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
TARGETS=['raw_full_1.npz','raw_full_2.npz','raw_plain_rnn_1.npz','final_no_replay_0.npz']

if __name__=='__main__':
    records=[];original=json.loads((OUT/'formal_metrics.json').read_text());archive=OUT/'initial_incomplete';archive.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wuyu06_recovery_') as td:
        for filename in TARGETS:
            path=OUT/filename
            if zipfile.is_zipfile(path):continue
            name=filename.split('_',1)[1][:-4];mode=name.rsplit('_',1)[0];initialization=int(name[-1]);family='plain_rnn' if mode=='plain_rnn' else 'full'
            selected=json.loads((OUT/f'training_{family}_{initialization}.json').read_text())['selected']['checkpoint']
            before={'file':filename,'incomplete_bytes':path.stat().st_size,'incomplete_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            shutil.copy2(path,archive/(filename+'.partial'))
            net=Network.from_state(read_tree(OUT/selected));m,_,_,_,_=rollout(net,np.arange(980000,980032),2048,mode,td,name)
            for k,v in m.items():np.testing.assert_allclose(v,original[name][k],atol=1e-12,rtol=0)
            replacement=Path(td)/filename
            with zipfile.ZipFile(replacement) as z:assert z.testzip() is None
            shutil.copy2(replacement,path)
            with zipfile.ZipFile(path) as z:assert z.testzip() is None
            records.append(before|{'restored_bytes':path.stat().st_size,'restored_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_original_metrics_match':True})
            print(json.dumps(records[-1]),flush=True)
    (OUT/'recovery.json').write_text(json.dumps({'reason':'Four initial NPZ files had no ZIP end directory. Underlying truncation cause undetermined. Re-executed exactly the selected model/seed/config; did not retrain or tune. Initial partial bytes retained.','files':records},indent=2))
