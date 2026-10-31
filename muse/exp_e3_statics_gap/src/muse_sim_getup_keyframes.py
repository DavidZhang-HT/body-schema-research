"""IMAGINED KEYFRAME CHAIN for one contact-mode transition (part of the oracle / interface layer).

"Imagine first, then move": before the MPC is asked to move, the oracle imagines a chain of
static keyframes from the source configuration to a configuration satisfying the destination
mode B, each one certified with the exact statics LP.  This separates two questions that a single
local MPC run conflates (lesson of the first SUPINE->SIDE run, results/getup_search/run1_restart4):

    EXISTENCE     does a certified quasi-static equilibrium exist at every intermediate stage?
                  (a gap here is a PHYSICAL diagnosis: "no equilibrium of this kind at stage s")
    CONNECTIVITY  can the MPC (getup_mpc, unchanged) move between consecutive keyframes while
                  staying in static equilibrium?  (a failure here is a local path-finding failure)

Keyframe = bounded least squares over (base position 3, base rotation increment 3, 20 joints),
all residuals human-readable:
    axis       trunk axis `col` of B onto the stage target (angle-interpolated from the start)
    height     base z toward the stage bound of B's extra predicate (STAND / SQUAT), interpolated
    req        lowest point of every required any-of group of B on the floor
    keep       every limb effector touching at the start stays down ("make before break");
               at the final stage only those that B allows (others must be released);
               a kept contact is PINNED at its start xy (held contacts stick, no sliding)
    base_xy    regulariser of the base xy toward the previous keyframe (no free drift)
    rot_min    regulariser of the base rotation toward the MINIMAL rotation that takes the axis to
               its stage target (no gratuitous pitch / yaw)
    forbid     (final stage only) regions B forbids at least 1 cm above the floor
    soles      (if feet are required) the 4 lowest sole corners on the floor (flat feet)
    penetrate  no shell point below the floor;  selfcol  every plant self-collision pair >= 3 mm
    com        (candidate-dependent) CoM ground projection toward a support point (see below)
    posture    regulariser toward the previous keyframe joints
Statics is not inside the least squares; instead a deterministic list of CoM candidates is tried
(none / dynamic centroid of the supports at the solution itself / centroid of the previous
keyframe's support set / midpoints toward each previous support point) and the first
candidate whose solution passes certify() (statics LP feasible, rho <= 0.8, penetration >= -1.5 mm,
self distance >= -1 mm) is kept.  If none passes, the stage is halved once; if still none, the
chain reports a GAP at that stage with every candidate's failure reason (at most MAX_HALVINGS
halvings per edge).  The chain stops at the first certified keyframe that already satisfies the
full predicate of B (getup_modes.mode_check) -- B's orientation target is only a direction.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import least_squares

from getup_schema import REGIONS, EFFECTORS
from getup_mpc import certify
from getup_modes import MODE, TOUCH_EPS, RHO_CAP, mode_check

STEP_DEG = 15.0              # max trunk-axis rotation per keyframe
STEP_Z = 0.04                # max base-height change per keyframe [m]
EXTRA_Z = {"STAND": (0.335, None), "SQUAT": (None, 0.25)}   # base-z bounds inside B's predicate
PEN_OK, SELF_OK = -0.0015, -0.001
MAX_HALVINGS = 3             # per edge
HARD_TOL = 0.005             # hard-residual cost accepted (e.g. ~1 deg axis error, ~3 mm contact gaps)
W_BASE_XY, W_ROT_MIN, W_POSTURE = 1.0, 1.0, 0.15   # regularisers (smallest change between keyframes)


def _elev(v):
    return np.degrees(np.arcsin(np.clip(v[2], -1, 1)))


def _axis_vec(h, elev_deg):
    e = np.radians(elev_deg)
    return np.cos(e) * h + np.sin(e) * np.array([0, 0, 1.0])


def _heading(a):
    h = np.array([a[0], a[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else np.array([1.0, 0, 0])


def cert_ok(c):
    return bool(c["feasible"]) and c["rho"] <= RHO_CAP and c["pen"] >= PEN_OK and c["selfd"] >= SELF_OK


def cert_why(c):
    w = []
    if not c["feasible"]:
        w.append("statics LP infeasible (no friction-cone forces balance gravity)")
    elif c["rho"] > RHO_CAP:
        w.append(f"rho {c['rho']:.2f} > {RHO_CAP}")
    if c["pen"] < PEN_OK:
        w.append(f"penetration {1e3 * c['pen']:.1f} mm")
    if c["selfd"] < SELF_OK:
        w.append(f"self-collision {1e3 * c['selfd']:.1f} mm")
    return "; ".join(w) or "ok"


class KeyframeImaginer:
    def __init__(self, S, mu):
        self.S, self.mu = S, mu
        self.qa = S.r.qadr

    def solve(self, q_ref, dst, axis_tgt, col, zb, keep, final, com_c, anchors=None):
        S, m = self.S, MODE[dst]
        need = set().union(*m.req)
        forbid = [r for r in REGIONS if r not in m.allowed and r not in need] if final else []
        feet_req = any(g & {"L_foot", "R_foot"} for g in m.req)
        quat0 = q_ref[3:7].copy()
        qj0 = q_ref[self.qa].copy()
        a_ref = S.kin(q_ref, with_G=False, with_pairs=False).R[:, col]
        cx = np.cross(a_ref, axis_tgt)
        w_min = cx / max(np.linalg.norm(cx), 1e-12) * np.arctan2(np.linalg.norm(cx), a_ref @ axis_tgt)

        def build(x):
            q = q_ref.copy()
            q[0:3] = x[0:3]
            qq = np.zeros(4); mujoco.mju_axisAngle2Quat(qq, x[3:6] / max(np.linalg.norm(x[3:6]), 1e-12),
                                                       np.linalg.norm(x[3:6]))
            r = np.zeros(4); mujoco.mju_mulQuat(r, qq, quat0)
            q[3:7] = r
            q[self.qa] = x[6:]
            return q

        def zmin(k, reg):
            return k.z[S.region_points[reg]].min()

        def resid(x, soft=True):
            q = build(x)
            k = S.kin(q, with_G=False, with_pairs=True, pair_dist=0.01)
            r = list(5.0 * (k.R[:, col] - axis_tgt))
            lo, hi = zb
            r.append(10 * min(0.0, q[2] - lo) if lo is not None else 0.0)
            r.append(10 * max(0.0, q[2] - hi) if hi is not None else 0.0)
            for g in m.req:
                r.append(10 * min(zmin(k, reg) for reg in g))
            for reg in keep:
                r.append(10 * zmin(k, reg))
                if anchors and reg in anchors:     # held contact: no sliding (stick)
                    j = S.region_points[reg][np.argmin(k.z[S.region_points[reg]])]
                    r += list(10 * (k.xy[j] - anchors[reg]))
            for reg in forbid:
                r.append(10 * min(0.0, zmin(k, reg) - 0.01))
            if feet_req:
                for f in ("L_foot", "R_foot"):
                    r += list(10 * np.sort(k.z[S.region_points[f]])[:4])
            r += list(30 * np.minimum(k.z, 0.0))
            r.append(30 * np.sqrt(sum(min(0.0, dd - 0.003) ** 2 for dd, _, _ in k.pairs)))
            if not soft:
                return np.array(r)
            if isinstance(com_c, str):          # "dynamic": centroid of the supports AT x
                pts = [k.xy[S.region_points[reg][np.argmin(k.z[S.region_points[reg]])]]
                       for reg in [min(g, key=lambda rr: zmin(k, rr)) for g in m.req] + list(keep)]
                r += list(3.0 * (k.com[:2] - np.mean(pts, 0)))
            elif com_c is not None:
                r += list(3.0 * (k.com[:2] - com_c))
            r += list(W_POSTURE * (x[6:] - qj0))
            r += list(W_BASE_XY * (x[0:2] - q_ref[0:2]))
            r += list(W_ROT_MIN * (x[3:6] - w_min))     # prefer the minimal trunk rotation
            return np.array(r)

        x0 = np.concatenate([q_ref[0:3], np.zeros(3), qj0])
        lb = np.concatenate([[-np.inf, -np.inf, 0.0], [-np.pi] * 3, S.lo])
        ub = np.concatenate([[np.inf, np.inf, 1.0], [np.pi] * 3, S.hi])
        x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
        sol = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-9, ftol=1e-9, max_nfev=300)
        q = build(sol.x)
        hard = resid(sol.x, soft=False)
        return q, float(0.5 * hard @ hard)          # HARD residual cost only (regularisers excluded)

    def support_points(self, q, dst, keep):
        S, m = self.S, MODE[dst]
        k = S.kin(q, with_G=False, with_pairs=False)
        xy = S.region_xy(k)
        d = S.region_height(k)
        pts = []
        for g in m.req:
            r = min(g, key=lambda rr: d[REGIONS.index(rr)])
            pts.append(xy[REGIONS.index(r)])
        for r in keep:
            pts.append(xy[REGIONS.index(r)])
        return pts

    def stage(self, q_prev, dst, axis_tgt, col, zb, keep, final, anchors=None):
        """Try the CoM candidates in order; return (q, cert, cand_name, attempts)."""
        attempts = []
        cands = [("none", None), ("dynamic_support_centroid", "dynamic")]
        pts = self.support_points(q_prev, dst, keep)
        if pts:
            cands.append(("support_centroid", np.mean(pts, 0)))
            for i, p in enumerate(pts):
                cands.append((f"support_pt{i}", 0.5 * (p + np.mean(pts, 0))))
        for name, c in cands:
            q, cost = self.solve(q_prev, dst, axis_tgt, col, zb, keep, final, c, anchors)
            ce = certify(self.S, q, self.mu)
            attempts.append(dict(cand=name, lsq=round(cost, 5), why=cert_why(ce),
                                 rho=round(float(ce["rho"]), 3) if np.isfinite(ce["rho"]) else None))
            if cert_ok(ce) and cost < HARD_TOL:
                return q, ce, name, attempts
        return None, None, None, attempts

    def chain(self, q_src, dst, col, elev_final, verbose=True):
        """Returns dict(ok, keyframes=[(stage, q, cert summary, cand)], gap=None|dict)."""
        S, m = self.S, MODE[dst]
        k0 = S.kin(q_src, with_G=False, with_pairs=False)
        a0 = k0.R[:, col].copy()
        h = _heading(a0)
        e0 = _elev(a0)
        d0 = S.region_height(k0)
        keep = [r for r, v in zip(REGIONS, d0) if v <= TOUCH_EPS and r in EFFECTORS]
        xy0 = S.region_xy(k0)
        anchors = {r: xy0[REGIONS.index(r)].copy() for r in keep}
        z0 = q_src[2]
        lo, hi = EXTRA_Z.get(dst, (None, None))
        z_fin = lo if (lo is not None and z0 < lo) else (hi if (hi is not None and z0 > hi) else None)
        n = max(1, int(np.ceil(abs(elev_final - e0) / STEP_DEG)),
                int(np.ceil(abs(z_fin - z0) / STEP_Z)) if z_fin is not None else 1)
        fr = list(np.linspace(0, 1, n + 1)[1:])
        out, q, f_prev, halved, n_half = [], q_src.copy(), 0.0, False, 0
        while fr:
            f = fr[0]
            final = abs(f - 1.0) < 1e-9
            tgt = _axis_vec(h, e0 + f * (elev_final - e0))
            zb = (None, None)
            if z_fin is not None:
                zt = z0 + f * (z_fin - z0)
                zb = (zt, None) if lo is not None else (None, zt)
            keep_f = [r for r in keep if r in m.allowed or r in set().union(*m.req)] if final else keep
            qk, ce, cand, att = self.stage(q, dst, tgt, col, zb, keep_f, final, anchors)
            if qk is None and not halved and f - f_prev > 0.02 and n_half < MAX_HALVINGS:
                fr.insert(0, 0.5 * (f_prev + f)); halved = True; n_half += 1
                if verbose:
                    print(f"     keyframe stage f={f:.2f} failed -> halving", flush=True)
                continue
            stage_name = f"elev {e0 + f * (elev_final - e0):+.0f}deg" + (f" z>={zb[0]:.3f}" if zb[0] else "") + \
                         (f" z<={zb[1]:.3f}" if zb[1] else "")
            if qk is None:
                return dict(ok=False, keyframes=out, keep=keep,
                            gap=dict(frac=round(f, 3), stage=stage_name, attempts=att))
            kk = S.kin(qk, with_G=False, with_pairs=False)
            dd = S.region_height(kk)
            out.append(dict(frac=round(f, 3), stage=stage_name, q=qk, cand=cand,
                            rho=round(float(ce["rho"]), 3), binding=ce["binding"],
                            loads={r: round(float(v), 3) for r, v in ce["loads"].items() if v > 0.005},
                            touching=[r for r, v in zip(REGIONS, dd) if v <= TOUCH_EPS]))
            if verbose:
                print(f"     keyframe {stage_name}: rho {ce['rho']:.3f} cand {cand} loads {out[-1]['loads']}", flush=True)
            q, f_prev, halved = qk, f, False
            fr.pop(0)
            # early stop: the chain ends at the first certified keyframe that already IS mode B
            if mode_check(S, kk, ce, dst)[0]:
                out[-1]["in_dst_mode"] = True
                break
        return dict(ok=True, keyframes=out, keep=keep, gap=None)
