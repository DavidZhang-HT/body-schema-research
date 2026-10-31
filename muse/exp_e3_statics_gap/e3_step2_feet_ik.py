"""E3 step 2: IK-nearby search for a FEET-ONLY certified static equilibrium from B's plateau.

Bounded least squares over x = (base translation 3, base rotation vector 3, joints 24), applied to the
plateau qpos with mj_integratePos (the plateau is the reference).  Residuals (all human-readable):
  sole_flat   the 4 sole-polygon corners of each foot at z = 0           (w 30)  [flat feet]
  pin         hinge(|sole centre xy - plateau contact xy| - R_pin)        (w 30)
  clear       hinge(0.01 - z) for the lowest point of every non-foot geom (w 30)   [A's 'forbid']
  pen         hinge(-z) for the lowest foot point of every foot geom      (w 30)
  selfcol     hinge(0.003 - dist) for every plant self-collision pair within 2 cm  (w 30)
  com         CoM xy - target (support centroid or shrink candidates)      (w 10)
  posture     w_post * (q_joints - q_plateau)                              (w_post swept)
  keep        (lift-off mode) kept non-foot region's lowest point z = 0 and xy pinned (w 30)
Statics is not in the least squares (as in A's KeyframeImaginer): each solution is certified afterwards.
"""
import json, sys, time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from pathlib import Path
from e3_plant import GroundWorld, load_plateau
from e3_cert import BSchema, MU_PLAN, cert_ok

OUT = Path(__file__).resolve().parent / 'results'; OUT.mkdir(exist_ok=True)


class FeetIK:
    def __init__(self, S: BSchema, q_ref):
        self.S = S; self.m = S.m; self.q_ref = q_ref.copy()
        import experiment
        self.d2 = mj.MjData(self.m)
        # sole polygon from B's standing pose: foot hull points within 1 mm of the floor
        qs = experiment.standing(self.m); d = S.forward(qs); W = S.points_world(d)
        self.sole = {}
        for f in ('L_foot', 'R_foot'):
            idx = np.where((S.p_region == f) & (W[:, 2] < W[S.p_region == f, 2].min() + 0.001))[0]
            loc = S.p_local[idx]; h = ConvexHull(loc[:, :2]).vertices
            # 4 extreme corners (min/max along foot x, y in the foot frame)
            L = loc[h]; c = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    c.append(L[np.argmax(sx * L[:, 0] + sy * L[:, 1])])
            self.sole[f] = (int(S.p_body[idx[0]]), np.array(c), loc.mean(0))
        self.geoms = np.unique(S.p_geom)
        assert np.all(np.diff(S.p_geom) >= 0)          # points are stored grouped by geom
        self.g_start = np.searchsorted(S.p_geom, self.geoms)
        self.g_region = S.p_region[self.g_start]
        self.g_foot = np.isin(self.g_region, ['L_foot', 'R_foot'])
        self.pa = np.array([a for a, b in S.pairs]); self.pb = np.array([b for a, b in S.pairs])
        self.prad = self.m.geom_rbound[self.pa] + self.m.geom_rbound[self.pb] + 0.02
        self.lo = self.m.jnt_range[1:, 0] + .02; self.hi = self.m.jnt_range[1:, 1] - .02
        d = S.forward(q_ref); W = S.points_world(d)
        self.pin = {}
        for f in ('L_foot', 'R_foot'):
            i = np.where(S.p_region == f)[0]; self.pin[f] = W[i[np.argmin(W[i, 2])], :2].copy()

    def qpos(self, x):
        q = self.q_ref.copy(); v = np.r_[x[:6], np.zeros(24)]
        # translation in world frame: integratePos treats free-joint linear part in world frame
        mj.mj_integratePos(self.m, q, v, 1.0)
        q[7:] = self.q_ref[7:] + x[6:]
        return q

    def sole_world(self, d, f):
        b, c, cen = self.sole[f]; R = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + c @ R.T, d.xpos[b] + R @ cen

    def resid(self, x, p):
        S = self.S; q = self.qpos(x); d = S.forward(q); W = S.points_world(d); z = W[:, 2]
        r = []
        poly = []
        for f in ('L_foot', 'R_foot'):
            cw, cen = self.sole_world(d, f); poly.append(cw[:, :2])
            r += list(30 * cw[:, 2])
            r.append(30 * max(0.0, np.linalg.norm(cen[:2] - self.pin[f]) - p['R_pin']))
        zg = np.minimum.reduceat(z, self.g_start)
        floor_ok = self.g_foot.copy()
        if p.get('keep') is not None:
            floor_ok |= self.g_region == p['keep']
        r += list(30 * np.where(floor_ok, np.maximum(0, -zg), np.maximum(0, 0.01 - zg)))
        if p.get('keep') is not None:
            km = S.p_region == p['keep']; i = np.where(km)[0][np.argmin(z[km])]
            r.append(30 * z[i]); r.append(30 * max(0.0, np.linalg.norm(W[i, :2] - p['keep_xy']) - 0.03))
        ft = np.zeros(6); sc = np.zeros(len(self.pa))
        near = np.where(np.linalg.norm(d.geom_xpos[self.pa] - d.geom_xpos[self.pb], axis=1) < self.prad)[0]
        for k in near:
            dist = mj.mj_geomDistance(self.m, d, int(self.pa[k]), int(self.pb[k]), 0.02, ft)
            sc[k] = 30 * max(0.0, 0.003 - dist)
        r += list(sc)
        P = np.vstack(poly); com = d.subtree_com[1][:2]
        tgt = self.com_target(P, p['shrink'])
        r += list(10 * (com - tgt))
        r += list(p['w_post'] * x[6:]) + list(0.3 * p['w_post'] * x[:6])
        return np.array(r)

    @staticmethod
    def com_target(P, shrink):
        h = P[ConvexHull(P).vertices]; c = h.mean(0)
        return c  # centroid of the 8-corner support polygon (shrink kept for API symmetry)

    def solve(self, p, x0=None, max_nfev=60):
        x0 = np.zeros(30) if x0 is None else x0
        lb = np.r_[np.full(6, -np.inf), self.lo - self.q_ref[7:]]; ub = np.r_[np.full(6, np.inf), self.hi - self.q_ref[7:]]
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        res = least_squares(lambda x: self.resid(x, p), x0, bounds=(lb, ub), diff_step=1e-4, max_nfev=max_nfev, x_scale=1.0)
        return res

    def support_info(self, q):
        d = self.S.forward(q); out = {}
        P = []
        for f in ('L_foot', 'R_foot'):
            cw, cen = self.sole_world(d, f); P.append(cw[:, :2])
            out[f] = dict(sole_centre=cen.tolist(), pin=self.pin[f].tolist(), pin_dist=float(np.linalg.norm(cen[:2] - self.pin[f])),
                          corner_z_mm=(1e3 * cw[:, 2]).round(2).tolist())
        P = np.vstack(P); h = P[ConvexHull(P).vertices]
        com = d.subtree_com[1][:2]
        # signed distance of CoM to support polygon edges (positive = inside)
        hull = ConvexHull(P); sd = min(-(eq[:2] @ com + eq[2]) for eq in hull.equations)
        out['com_xy'] = com.tolist(); out['com_margin_m'] = float(sd); out['base'] = d.qpos[:3].tolist()
        out['up_z'] = float(d.xmat[self.m.body('base_link').id].reshape(3, 3)[2, 2])
        return out


def dist(q, q0):
    return dict(dq_max=float(np.max(np.abs(q[7:] - q0[7:]))), dq_l2=float(np.linalg.norm(q[7:] - q0[7:])),
                dbase_m=float(np.linalg.norm(q[:3] - q0[:3])))


def run_feet_only(tag, R_pin, w_posts, seeds_x0, S, ik, q0, log):
    best = None
    for w_post in w_posts:
        for name, x0 in seeds_x0:
            p = dict(R_pin=R_pin, w_post=w_post, shrink=0.0)
            t = time.time(); res = ik.solve(p, x0)
            # polish (A's keyframes accept a solution only if the hard residuals vanish): restart from
            # the solution with a 20x weaker posture regulariser so contact/clearance terms dominate
            res2 = ik.solve(dict(p, w_post=0.05 * w_post), res.x, max_nfev=150)
            q = ik.qpos(res2.x)
            c6 = S.certify(q, mu=MU_PLAN, feet_only=True); c8 = S.certify(q, mu=0.8, feet_only=True)
            call = S.certify(q, mu=MU_PLAN)
            rec = dict(tier=tag, R_pin=R_pin, w_post=w_post, x0=name, cost=float(res2.cost), nfev=int(res.nfev + res2.nfev),
                       sec=round(time.time() - t, 1), q=q.tolist(), cert_feet_mu06=c6, cert_feet_mu08=c8,
                       cert_all_mu06=call, support=ik.support_info(q), **dist(q, q0))
            # all-contacts check must not touch non-foot (clearance) -- record touching set
            rec['nonfoot_touching'] = [r for r in call['touching'] if r not in ('L_foot', 'R_foot')]
            rec['certified'] = bool(c6['ok'] and not rec['nonfoot_touching'])
            log.append(rec)
            print(f"[{tag}] w_post {w_post} x0 {name}: cost {res2.cost:.4f} nfev {res.nfev + res2.nfev} cert={rec['certified']} "
                  f"rho {c6['rho']:.3f} why '{c6['why']}' pen {1e3*c6['pen']:.1f} selfd {1e3*c6['selfd']:.1f} "
                  f"margin {1e3*rec['support']['com_margin_m']:.1f}mm dq_max {rec['dq_max']:.2f} pin "
                  f"{rec['support']['L_foot']['pin_dist']:.3f}/{rec['support']['R_foot']['pin_dist']:.3f} "
                  f"base_z {q[2]:.3f} nonfoot_touch {rec['nonfoot_touching']} ({rec['sec']}s)", flush=True)
            if rec['certified'] and (best is None or rec['dq_l2'] < best['dq_l2']):
                best = rec
    return best


def main():
    w = GroundWorld(); S = BSchema(w.m)
    q0 = np.load(OUT / 'plateau_snapshot.npy')[:31]
    ik = FeetIK(S, q0)
    print('sole corners (foot frame):', {f: v[1].round(3).tolist() for f, v in ik.sole.items()})
    print('pins', ik.pin)
    # extra initial guesses: B's standing joint pose and a deep squat, keeping the plateau base pose
    import experiment
    qs = experiment.standing(w.m)
    x_stand = np.r_[np.zeros(6), qs[7:] - q0[7:]]
    squat = qs[7:].copy()
    for s in (0, 6):
        squat[s + 0] = -1.3; squat[s + 3] = 2.0; squat[s + 4] = -0.75
    x_squat = np.r_[np.zeros(6), np.clip(squat, ik.lo, ik.hi) - q0[7:]]
    seeds = [('plateau', np.zeros(30)), ('stand_joints', x_stand), ('squat_joints', x_squat)]
    log = []
    bestA = run_feet_only('A_pin6cm', 0.06, [1.0, 0.3, 0.1], seeds, S, ik, q0, log)
    (OUT / 'step2_feet_only_log.json').write_text(json.dumps(log, indent=1))
    bestB = run_feet_only('B_pin20cm', 0.20, [0.3, 0.1], seeds, S, ik, q0, log)
    (OUT / 'step2_feet_only_log.json').write_text(json.dumps(log, indent=1))
    summ = dict(tierA_best=bestA, tierB_best=bestB, n_solves=len(log),
                n_certified_A=sum(r['certified'] for r in log if r['tier'] == 'A_pin6cm'),
                n_certified_B=sum(r['certified'] for r in log if r['tier'] == 'B_pin20cm'))
    (OUT / 'step2_feet_only_summary.json').write_text(json.dumps(summ, indent=1))
    if bestA: np.save(OUT / 'feet_only_eq_tierA.npy', np.array(bestA['q']))
    if bestB: np.save(OUT / 'feet_only_eq_tierB.npy', np.array(bestB['q']))
    print('certified A', summ['n_certified_A'], 'certified B', summ['n_certified_B'])


if __name__ == '__main__':
    main()
