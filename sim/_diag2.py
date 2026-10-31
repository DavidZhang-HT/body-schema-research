import numpy as np, test_walk as tw
from wbc import WBC
orig = WBC.control
cnt=[0]
def ctl(self, o, t):
    tau, info = orig(self, o, t)
    cnt[0]+=1
    if 0.98 < t < 2.1 and cnt[0] % 25 == 0:
        bm=self.bm; com, Jc, dJ = bm.com_jac()
        ach = Jc @ self.qdd_last + dJ
        w=self.planner.omega()
        print(f"t={t:.3f} k={getattr(self.planner,'k',-1)} xi={np.round(info['xi'],3)} ref={np.round(info['xi_ref'],3)} pcmd={np.round(info['p_cmd'],3)} accdes_y={w**2*(info['com'][1]-info['p_cmd'][1]):.2f} ach={np.round(ach,2)} st={info['stance']} sw={info['swing']}")
    return tau, info
WBC.control = ctl
tw.run(stride=0.0, nsteps=6)
