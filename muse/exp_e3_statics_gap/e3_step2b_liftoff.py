"""E3 step 2b: quasi-static CONNECTIVITY from the plateau to feet-only support (supporting evidence).

Existence of a feet-only equilibrium near the plateau (step 2) does not by itself say the load can be
moved onto the feet without momentum.  Here we ask the A-style keyframe question:

  LIFT-OFF  is there a configuration in which a non-foot support is still TOUCHING the floor
            and the feet-only certificate already holds (so the non-foot support can be released
            with zero load)?  Variants for which support stays down:
              L1 keep R_hand at its plateau xy (+-3 cm)        [the plateau's actual support]
              L2 keep R_hand anywhere (re-placed hand)
              L3 keep trunk (pelvis/butt) touching anywhere   [sit-squat lift-off]
  CHAIN     20 keyframes from the plateau to the L1 lift-off: base/joint linear interpolation,
            repaired by IK (feet lowest point on the floor and pinned, R_hand kept down and pinned,
            no penetration), each certified with all contacts within 5 mm (A's certify).
  MAX-MARGIN  for L1, the largest achievable signed CoM margin to the feet polygon (positive = inside)
            with the hand pinned, from a CoM-dominant solve: a continuous measure of the gap.
"""
import json, time
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from pathlib import Path
from e3_plant import GroundWorld
from e3_cert import BSchema, MU_PLAN
from e3_step2_feet_ik import FeetIK, dist, OUT


class KeepIK(FeetIK):
    def resid(self, x, p):
        S = self.S; q = self.qpos(x); d = S.forward(q); W = S.points_world(d); z = W[:, 2]
        r = []; poly = []
        for f in ('L_foot', 'R_foot'):
            cw, cen = self.sole_world(d, f); poly.append(cw[:, :2])
            if p.get('flat', True):
                r += list(30 * cw[:, 2])
            else:  # chain repair: lowest foot point on the floor (foot may be on an edge)
                fm = S.p_region == f; r.append(30 * z[fm].min())
            r.append(30 * max(0.0, np.linalg.norm(cen[:2] - self.pin[f]) - p['R_pin']))
        zg = np.minimum.reduceat(z, self.g_start)
        ok = self.g_foot | (self.g_region == p['keep'])
        clear = p.get('clear', 0.01)
        r += list(30 * np.where(ok, np.maximum(0, -zg), np.maximum(0, clear - zg)))
        km = S.p_region == p['keep']; i = np.where(km)[0][np.argmin(z[km])]
        r.append(30 * z[i])
        if p.get('keep_xy') is not None:
            r.append(30 * max(0.0, np.linalg.norm(W[i, :2] - p['keep_xy']) - p.get('keep_R', 0.03)))
        ft = np.zeros(6); sc = np.zeros(len(self.pa))
        near = np.where(np.linalg.norm(d.geom_xpos[self.pa] - d.geom_xpos[self.pb], axis=1) < self.prad)[0]
        for k in near:
            dd = mj.mj_geomDistance(self.m, d, int(self.pa[k]), int(self.pb[k]), 0.02, ft)
            sc[k] = 30 * max(0.0, 0.003 - dd)
        r += list(sc)
        P = np.vstack(poly); com = d.subtree_com[1][:2]
        if p.get('w_com', 10) > 0:
            r += list(p.get('w_com', 10) * (com - self.com_target(P, 0)))
        if p.get('x_ref') is not None:
            r += list(p['w_post'] * (x - p['x_ref']))
        else:
            r += list(p['w_post'] * x[6:]) + list(0.3 * p['w_post'] * x[:6])
        return np.array(r)


def liftoff(S, ik, q0, name, keep, keep_xy, keep_R, log, w_posts=(1.0, 0.3, 0.1), seeds=None):
    best = None
    for w_post in w_posts:
        for sname, x0 in seeds:
            p = dict(R_pin=0.06, w_post=w_post, keep=keep, keep_xy=keep_xy, keep_R=keep_R)
            t = time.time(); res = ik.solve(p, x0)
            res2 = ik.solve(dict(p, w_post=0.05 * w_post), res.x, max_nfev=150)
            q = ik.qpos(res2.x); d = S.forward(q); W = S.points_world(d)
            km = S.p_region == keep; kz = float(W[km, 2].min()); kxy = W[km][np.argmin(W[km, 2]), :2]
            cf = S.certify(q, mu=MU_PLAN, feet_only=True); ca = S.certify(q, mu=MU_PLAN)
            sup = ik.support_info(q)
            other = [r for r in ca['touching'] if r not in ('L_foot', 'R_foot', keep)]
            okk = bool(cf['ok'] and kz <= 0.005 and not other and (keep_xy is None or np.linalg.norm(kxy - keep_xy) <= keep_R + 0.005))
            rec = dict(variant=name, keep=keep, w_post=w_post, x0=sname, cost=float(res2.cost), certified_liftoff=okk,
                       keep_z_mm=1e3 * kz, keep_xy=kxy.tolist(), other_touching=other, cert_feet=cf, cert_all=ca,
                       support=sup, q=q.tolist(), **dist(q, q0))
            log.append(rec)
            print(f"[{name}] w_post {w_post} x0 {sname}: cost {res2.cost:.4f} liftoff_cert={okk} feet-only rho {cf['rho']:.3f} "
                  f"'{cf['why']}' keep_z {1e3*kz:.1f}mm margin {1e3*sup['com_margin_m']:.1f}mm dq_max {rec['dq_max']:.2f} "
                  f"base_z {q[2]:.3f} other {other} ({time.time()-t:.0f}s)", flush=True)
            if okk and (best is None or rec['dq_l2'] < best['dq_l2']):
                best = rec
    return best


def max_margin(S, ik, q0, keep_xy, log):
    """CoM-dominant solve with the hand pinned: how far can the CoM get into the feet polygon?"""
    out = []
    for w_com in (10, 30, 100):
        p = dict(R_pin=0.06, w_post=0.02, keep='R_hand', keep_xy=keep_xy, keep_R=0.03, w_com=w_com)
        res = ik.solve(p, np.zeros(30), max_nfev=200); q = ik.qpos(res.x)
        d = S.forward(q); W = S.points_world(d); km = S.p_region == 'R_hand'
        sup = ik.support_info(q); cf = S.certify(q, mu=MU_PLAN, feet_only=True)
        rec = dict(w_com=w_com, cost=float(res.cost), margin_m=sup['com_margin_m'], keep_z_mm=1e3 * float(W[km, 2].min()),
                   pen=cf['pen'], selfd=cf['selfd'], feet_cert=cf['ok'], rho=cf['rho'], q=q.tolist())
        out.append(rec)
        print(f"[max-margin] w_com {w_com}: margin {1e3*rec['margin_m']:.1f}mm hand_z {rec['keep_z_mm']:.1f}mm feet_cert {cf['ok']} '{cf['why']}'", flush=True)
    log.append(dict(variant='max_margin_L1', runs=out))
    return out


def chain(S, ik, q0, q1, n=20, keep='R_hand', keep_xy=None):
    """Interpolate base pose and joints q0->q1, repair each keyframe by IK, certify all contacts."""
    x1 = np.zeros(30)
    # x1 = integration displacement from q0 to q1
    v = np.zeros(30); mj.mj_differentiatePos(ik.m, v, 1.0, q0, q1); x1[:6] = v[:6]; x1[6:] = q1[7:] - q0[7:]
    rows = []; xprev = np.zeros(30)
    for k in range(n + 1):
        s = k / n; xr = s * x1
        p = dict(R_pin=0.06, w_post=1.0, keep=keep, keep_xy=keep_xy, keep_R=0.03, flat=False, clear=0.0, w_com=0, x_ref=xr)
        res = ik.solve(p, xprev, max_nfev=80); xprev = res.x; q = ik.qpos(res.x)
        c = S.certify(q, mu=MU_PLAN); sup = ik.support_info(q)
        rows.append(dict(s=s, cost=float(res.cost), ok=c['ok'], feasible=c['feasible'], rho=c['rho'], why=c['why'],
                         loaded=c['loaded'], touching=c['touching'], pen=c['pen'], selfd=c['selfd'],
                         com_margin_feet_m=sup['com_margin_m'], dev_from_interp=float(np.max(np.abs(res.x - xr))), q=q.tolist()))
        print(f"  chain s={s:.2f} ok={c['ok']} rho {c['rho']:.3f} '{c['why']}' loaded {c['loaded']} margin {1e3*sup['com_margin_m']:.0f}mm "
              f"dev {rows[-1]['dev_from_interp']:.3f} cost {res.cost:.4f}", flush=True)
    return rows


def main():
    w = GroundWorld(); S = BSchema(w.m)
    q0 = np.load(OUT / 'plateau_snapshot.npy')[:31]
    ik = KeepIK(S, q0)
    d = S.forward(q0); W = S.points_world(d); km = S.p_region == 'R_hand'
    hand_xy = W[km][np.argmin(W[km, 2]), :2].copy(); print('plateau R_hand contact xy', hand_xy)
    import experiment
    qs = experiment.standing(w.m)
    seeds = [('plateau', np.zeros(30)), ('stand_joints', np.r_[np.zeros(6), qs[7:] - q0[7:]])]
    log = []; summ = {}
    summ['L1'] = liftoff(S, ik, q0, 'L1_hand_pinned', 'R_hand', hand_xy, 0.03, log, seeds=seeds)
    (OUT / 'step2b_log.json').write_text(json.dumps(log, indent=1))
    summ['max_margin_L1'] = [{k: v for k, v in r.items() if k != 'q'} for r in max_margin(S, ik, q0, hand_xy, log)]
    (OUT / 'step2b_log.json').write_text(json.dumps(log, indent=1))
    summ['L2'] = liftoff(S, ik, q0, 'L2_hand_anywhere', 'R_hand', None, 1.0, log, w_posts=(0.3, 0.1), seeds=seeds)
    summ['L3'] = liftoff(S, ik, q0, 'L3_trunk_touching', 'trunk', None, 1.0, log, w_posts=(0.3, 0.1), seeds=seeds)
    (OUT / 'step2b_log.json').write_text(json.dumps(log, indent=1))
    if summ['L1'] is not None:
        print('CHAIN plateau -> L1 lift-off'); q1 = np.array(summ['L1']['q'])
        summ['chain_L1'] = chain(S, ik, q0, q1, keep_xy=hand_xy)
        summ['chain_L1_all_ok'] = all(r['ok'] for r in summ['chain_L1'])
    (OUT / 'step2b_summary.json').write_text(json.dumps(summ, indent=1))
    print('L1', summ['L1'] is not None, 'L2', summ['L2'] is not None, 'L3', summ['L3'] is not None,
          'chain_all_ok', summ.get('chain_L1_all_ok'))


if __name__ == '__main__':
    main()
