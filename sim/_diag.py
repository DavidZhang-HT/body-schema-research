import numpy as np, test_walk as tw
from wbc import WBC
orig = WBC.control
lastk = [-1]
def ctl(self, o, t):
    tau, info = orig(self, o, t)
    pl = self.planner
    if pl.mode == "walk" and pl.k != lastk[0]:
        lastk[0] = pl.k
        print(f"t={t:.3f} k={pl.k} S_k-1={pl.S[max(pl.k-1,0)][0]}{np.round(pl.S[max(pl.k-1,0)][1],3)} xi={np.round(info['xi'],3)} xiref={np.round(info['xi_ref'],3)} com={np.round(info['com'],3)} feetL={np.round(info['foot_pos']['l'],3)} R={np.round(info['foot_pos']['r'],3)} grf={ {k:round(v,1) for k,v in info['grf'].items()} } adj={info.get('step_adjust')}")
    return tau, info
WBC.control = ctl
import sys; tw.run(stride=float(sys.argv[1]), nsteps=14)
