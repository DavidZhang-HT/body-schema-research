"""Existence probe for quasi-static ROLLED equilibria (supine -> right side), independent of the
search/oracle path-finding.  For each roll angle theta the trunk is rotated about the world x axis
(body +y rises, right flank goes down), then base position + joints are
solved by bounded least squares (base xyz + all 20 joints; arms may move out of the way) so that: the lowest point of the SIDE support group {trunk, R_upper_arm} touches the floor, both
soles are flat on the floor (4 lowest corners at z = 0), nothing penetrates; regularised toward
the supine-settled posture.  The resulting configuration is then certified with the exact statics
LP (getup_mpc.certify: friction mu 0.6, peak |tau|/tau_max) and the self-collision distance.

Reading: if certified equilibria exist along theta = 0..45 deg, a quasi-static roll is NOT ruled
out by statics; an oracle failure on SUPINE->SIDE is then a path-finding (local-descent) failure,
not a physical impossibility.  Deterministic (no randomness).

Usage: python getup_roll_probe.py   -> results/getup_search/roll_probe.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
from scipy.optimize import least_squares

from getup_schema import Schema, REGIONS
from getup_mpc import certify
from getup_modes import classify, mode_check

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "getup_search")


def quat_mul(a, b):
    r = np.zeros(4); mujoco.mju_mulQuat(r, a, b); return r


def main():
    S = Schema()
    q0 = np.load(os.path.join(HERE, "..", "results", "f3", "q_supine_settled.npy"))
    qa = S.r.qadr
    legs = list(range(20))          # all 20 joints (legs + arms)
    trunk = np.concatenate([S.region_points[r] for r in ("upper_trunk", "lower_back", "pelvis", "R_upper_arm")])   # SIDE req group
    feet = [S.region_points["L_foot"], S.region_points["R_foot"]]
    res_all = []
    x_prev = None
    for th_deg in [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90]:
        th = np.radians(th_deg)
        qr = np.array([np.cos(th / 2), np.sin(th / 2), 0, 0])
        quat = quat_mul(qr, q0[3:7])

        def build(x):
            q = q0.copy()
            q[0:3] = x[0:3]
            q[3:7] = quat
            q[qa[legs]] = x[3:]
            return q

        def resid(x):
            q = build(x)
            k = S.kin(q, with_G=False, with_pairs=False)
            r = [10 * k.z[trunk].min()]
            for idx in feet:
                zs = np.sort(k.z[idx])[:4]
                r += list(10 * zs)
            pen = np.minimum(k.z, 0.0)
            r += list(30 * pen)
            ks = S.kin(q, with_G=False, with_pairs=True, pair_dist=0.01)
            r.append(30 * np.sqrt(sum(min(0.0, dd - 0.003) ** 2 for dd, _, _ in ks.pairs)))   # self-collision (fixed length)
            r += list(0.05 * (x[3:] - q0[qa[legs]]))
            return np.array(r)

        x0 = x_prev if x_prev is not None else np.concatenate([q0[0:3], q0[qa[legs]]])
        lb = np.concatenate([[-1, -1, 0.0], S.lo[legs]])
        ub = np.concatenate([[1, 1, 0.5], S.hi[legs]])
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        sol = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, max_nfev=400)
        x_prev = sol.x
        q = build(sol.x)
        c = certify(S, q, S.mu)
        k = S.kin(q, with_G=False, with_pairs=False)
        d = S.region_height(k)
        rec = dict(theta_deg=th_deg, lsq_cost=float(sol.cost), feasible=bool(c["feasible"]),
                   rho=float(c["rho"]) if np.isfinite(c["rho"]) else None, binding=c["binding"],
                   loads={r: round(float(v), 3) for r, v in c["loads"].items() if v > 0.005},
                   pen_mm=round(1e3 * c["pen"], 2), selfd_mm=round(1e3 * c["selfd"], 2),
                   left_z=round(float(k.R[2, 1]), 3),
                   touching=[r for r, v in zip(REGIONS, d) if v <= 0.005],
                   modes=classify(S, k, c))
        print(rec, flush=True)
        res_all.append(rec)
        np.save(os.path.join(OUT, f"roll_probe_q_{th_deg}.npy"), q)
    with open(os.path.join(OUT, "roll_probe.json"), "w") as f:
        json.dump(res_all, f, indent=1)


if __name__ == "__main__":
    main()
