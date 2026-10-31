import itertools, numpy as np, test_walk as tw
from wbc import GaitParams
res=[]
for T,k,w,ds in itertools.product([0.35,0.45],[2.0,4.0],[0.14,0.18],[0.1,0.25]):
    gp=GaitParams(T_step=T,k_dcm=k,step_width=w,ds_frac=ds)
    rob,wbc,L=tw.run(gp=gp,stride=0.10,nsteps=14,verbose=False)
    print(T,k,w,ds,'fell' if rob.fallen() else 'ok', round(rob.t,2), np.round(rob.d.qpos[:2],2), flush=True)
