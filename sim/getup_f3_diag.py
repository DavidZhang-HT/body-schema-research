"""Stall diagnosis: at a stalled configuration, which constraint family blocks task progress?
For each inequality family, drop it and measure the predicted task-cost decrease of one MPC step
(linearised) vs. the full problem.  Families whose removal unlocks progress are binding."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from getup_schema import Schema, REGIONS
from getup_attention import SchemaAttention
from getup_mpc import QSMPC, MPCParams, make_phases, certify
import getup_mpc as gm

S = Schema(); A = SchemaAttention(S); p = MPCParams()
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
d = np.load(f'../results/f3/run_{tag}.npz')
qs = d['q']; i = int(np.where(np.isfinite(qs[:, 0]))[0][-1])
q = qs[i]
phs = make_phases(S); ph = [x for x in phs if x.name == str(d['phase'][i])][0]
k = S.kin(q, with_G=False, with_pairs=True, pair_dist=0.01)
tilt = np.degrees(np.arccos(np.clip(k.R[2, 2], -1, 1)))
print('tick', i, 'phase', ph.name, 'tilt deg', round(tilt, 1), 'z', round(q[2], 3))
for t in ph.tasks(S, k):
    print('  task', t.name, 'cost', round(float(t.w * np.sum(t.r ** 2)), 3), 'r', np.round(t.r, 3))
print('  region z<2cm (mm)', {r: round(float(k.z[S.region_points[r]].min()) * 1e3, 1) for r in REGIONS if k.z[S.region_points[r]].min() < 0.02})
c = certify(S, q, p.mu); print('  cert rho', round(c['rho'], 3), c['binding'], {r: round(v, 3) for r, v in c['loads'].items() if v > 0.005})
print('  selfcol pairs <1cm', [(round(dd * 1e3, 1), S.m.geom(g[0]).name, S.m.geom(g[1]).name) for dd, _, g in k.pairs if dd < 0.01])
from model import JOINTS
qj = q[S.r.qadr]; m = np.minimum(qj - S.lo, S.hi - qj)
print('  joints within 0.05 rad of limit', {JOINTS[j]: round(float(m[j]), 3) for j in range(20) if m[j] < 0.05})

# family-drop sensitivity: monkeypatch _solve to record, then re-run step with families removed
orig = QSMPC._solve
def run(drop):
    def patched(P, qv, rows, lo, up, D=None, _fam=[None]):
        return orig(P, qv, rows, lo, up, D)
    mpc = QSMPC(S, A, p)
    # build via step but intercept rows: easiest is to relax families through params
    return mpc
# relax families by parameter where available
variants = {
    'full': {},
    'no_selfcol': dict(use_selfcol=False),
    'no_jlim': dict(use_jlim=False),
    'big_trust': dict(tr_joint=0.1, tr_base_lin=0.03, tr_base_ang=0.1),
    'rho_cap_1e3': dict(rho_cap=1e3),
    'mu_2': dict(mu=2.0),
}
for name, kw in variants.items():
    pp = MPCParams(**{**p.__dict__, **kw})
    mpc = QSMPC(S, A, pp)
    qq = q.copy(); costs = []
    for it in range(15):
        dq, info = mpc.step(qq, ph, {})
        if dq is None:
            costs.append('INF'); break
        qq = mpc.project(S.integrate(qq, dq))
        kk = S.kin(qq, with_G=False, with_pairs=False)
        costs.append(round(float(sum(t.w * np.sum(t.r ** 2) for t in ph.tasks(S, kk))), 3))
    kk = S.kin(qq, with_G=False, with_pairs=False)
    print(f'  {name:12s} cost after 15 ticks: {costs[-1]}  tilt {np.degrees(np.arccos(np.clip(kk.R[2,2],-1,1))):.1f}')
