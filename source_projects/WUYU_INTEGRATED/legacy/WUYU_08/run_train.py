import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json
from pathlib import Path
from wuyu08.experiment import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--init',type=int,default=0);p.add_argument('--config',default='CONFIG.json');p.add_argument('--output',default='results');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());train(cfg,a.cell,a.init,a.output,a.pilot)
