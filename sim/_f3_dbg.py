import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from getup_schema import Schema, REGIONS
from getup_attention import SchemaAttention
from getup_mpc import QSMPC, MPCParams, make_phases, certify
S = Schema(); A = SchemaAttention(S); p = MPCParams()
d = np.load('../results/f3/run_a.npz')
i = int(sys.argv[1]) if len(sys.argv) > 1 else 35
q = d['q'][i]
ph = make_phases(S)[0]
k = S.kin(q, with_G=False, with_pairs=True, pair_dist=0.01)
print('zmin per region', {r: round(float(k.z[S.region_points[r]].min())*1e3, 2) for r in REGIONS if k.z[S.region_points[r]].min() < 0.02})
print('self min', min([dd for dd, _, _ in k.pairs], default=1))
qj = q[S.r.qadr]; m = np.minimum(qj - S.lo, S.hi - qj)
from model import JOINTS
print('jl margins <0.03', {JOINTS[j]: round(float(m[j]), 4) for j in range(20) if m[j] < 0.03})
for eps in (0.004, 0.005):
    lp = S.statics_lp(k, active=k.z <= eps, mu=p.mu)
    print('eps', eps, lp['feasible'], lp['rho'], {r: round(v, 3) for r, v in lp.get('loads', {}).items() if v > 0.005})
mpc = QSMPC(S, A, p)
log = {}
dq, info = mpc.step(q, ph, log)
print('cold step:', info['status'], info.get('infeasible_diag'), (info.get('elastic') or [])[:4])
