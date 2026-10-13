"""Continue a real saved individual, including its learned models and RNG."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from pathlib import Path
import numpy as np
from wuyu07.world import ACTIONS
from wuyu07.student import Student
from wuyu07.experiment import restore,step,checkpoint
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default=str(ROOT/'results/checkpoint_neural_student_0_384.npz'))
    parser.add_argument('--steps',type=int,default=48);parser.add_argument('--save');args=parser.parse_args()
    if args.steps<0 or args.steps>100000:parser.error('--steps must be between 0 and 100000')
    w,a=restore(args.checkpoint);start=w.t
    print(f'已恢复 {w.n} 个个体，当前事件 {w.t}；下方展示其中第一个。')
    for i in range(args.steps):
        if w.t>=w.total:
            w.regime_override=3;w.total=w.t+args.steps-i
            print('超出原正式测试时长；沿用最后阶段规律继续，这部分属于新的探索运行。')
        r=step(w,a)
        if i%8==0 or i==args.steps-1:
            skills=a.memories[0].valid.sum() if isinstance(a,Student) else a[0].valid.sum()
            print(f"t={w.t} action={ACTIONS[r['action'][0]]} energy={w.body[0,0]:.3f} integrity={w.body[0,1]:.3f} skills={skills} alive={bool(w.alive[0])}")
    if args.save:checkpoint(args.save,w,a);print(f'已保存到 {args.save}')
    print(f'本次执行了 {w.t-start} 个事件；程序结束后个体不会继续后台运行。')
