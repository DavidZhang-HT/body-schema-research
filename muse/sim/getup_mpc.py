"""Explicit quasi-static model-predictive control ("imagine") for get-up -- F3 test.

Formulation (receding-horizon sequential QP, real-time-iteration style; dense exact QP via quadprog):

  decision per horizon node k = 1..N :  dq_k (26, tangent increment of the full floating-base
  configuration), tau_k (20 joint torques), rho_k (peak torque ratio), f_k (3-D contact force
  for every shell point that is allowed to carry load at node k).
  q_k = q_0 (+) dq_1 (+) ... (+) dq_k ;  every node is linearised about the warm-start node qbar_k
  (shifted previous solution), e_k = q_k (-) qbar_k.

  min  sum_k  || W_T (r_T,k + J_T,k e_k) ||^2                      imagined sub-goal (task)
            + sum_j  w_m / (a_j + a0)  dq_kj^2                      MOTION attention price
            + w_e sum_j (1 + alpha_kj) (tau_kj / tau_max_j)^2 + w_rho rho_k   EFFORT attention
            + w_f sum_p (f_z,kp / mg)^2 / (a_r(p) + a0) + w_t |f_t|^2       SUPPORT attention price
            + smoothness
  s.t. (all linearised at qbar_k)
       sum_p J_p^T f_kp + S^T tau_k = g(qbar_k) + G(qbar_k) e_k     quasi-static equilibrium
                                                                    (full floating-base statics:
                                                                     sum F = mg, sum M = 0)
       f_z >= 0, |f_x|,|f_y| <= mu/sqrt(2) f_z                      unilateral + friction pyramid
       only points within eps (4 mm) of the floor get force variables
       contact state machine from the warm start: a point loaded > F_s is a HELD contact
       (|dp_kp| <= 0.5 mm in 3-D: stick, cannot lift off while loaded); an unloaded point may
       move freely but picks up at most F_s per tick -> loads ramp continuously, a support is
       always unloaded before it lifts off
       z_kp >= 0  for every shell point                            non-penetration
       |tau_kj| <= rho_k tau_max_j,  rho_k <= rho_cap (0.8)         torque limits / F3 criterion
       q_lo + m <= q_k <= q_hi - m                                  joint limits
       dist_ab + n^T (J_b - J_a) e_k >= d_min                       self-collision (all plant pairs)
       |dq_k| <= trust region                                       quasi-static step size

  Only the first increment is executed (receding horizon); every executed configuration is
  re-certified with the exact statics LP (Schema.statics_lp).  No velocities, no momentum: the
  path is a sequence of static equilibria (this is exactly what F3 tests).

Phases = imagined sub-goals, executed in sequence (hand-specified mode graph per analysis G5;
which contacts / joints participate is decided by the attention heads, not by the phase):
  P1 head_up    rotate the trunk from lying to upright                    (sit up)
  P2 tuck       bring both soles flat to the floor in front of the pelvis
  P3 waist_up   move the CoM over the feet and lift the pelvis off the floor
  P4 stand      raise the CoM to standing height with an upright trunk
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import quadprog

from getup_schema import Schema, Kin, REGIONS, EFFECTORS, EPS_CONTACT, G
from getup_attention import SchemaAttention, AttnParams
from model import JOINTS, FEET, Z_STAND_NOM


def rot_err_axis(a, b):
    """Rotation vector (world) that turns unit vector a into unit vector b."""
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    ang = np.arctan2(s, float(a @ b))
    return c / s * ang if s > 1e-9 else np.zeros(3)


# ============================================================================ tasks
@dataclass
class Task:
    name: str
    r: np.ndarray          # residual (want r + J e -> 0)
    J: np.ndarray          # m x nv
    w: float


def t_trunk_upright(k: Kin, w, lean=0.0):
    zb = k.R[:, 2]
    target = np.array([np.sin(lean), 0.0, np.cos(lean)])
    e = rot_err_axis(zb, target)
    return Task("trunk_upright", -e, k.Jrot, w)


def t_com(k: Kin, w, target, axes=(0, 1, 2), scale=10.0):
    ax = list(axes)
    return Task("com", scale * (k.com[ax] - np.asarray(target)[ax]), scale * k.Jcom[ax], w)


def t_region_up(S: Schema, k: Kin, region, h, w, scale=10.0):
    idx = S.region_points[region]
    j = idx[np.argmin(k.z[idx])]
    r = min(0.0, k.z[j] - h)
    return Task(f"{region}_up", np.array([scale * r]), scale * k.Jp[j][2:3], w)


def t_region_down(S: Schema, k: Kin, region, w, scale=10.0):
    idx = S.region_points[region]
    j = idx[np.argmin(k.z[idx])]
    return Task(f"{region}_land", np.array([scale * k.z[j]]), scale * k.Jp[j][2:3], w)


def t_foot_flat(k: Kin, w):
    rs, Js = [], []
    for f in FEET:
        pc, Rb, jp, jr = k.body_pos[f + "/sole"]
        e = rot_err_axis(Rb[:, 2], np.array([0, 0, 1.0]))
        rs.append(-e); Js.append(jr)
    return Task("foot_flat", np.concatenate(rs), np.vstack(Js), w)


def t_foot_place(k: Kin, w, x_rel, y_abs=0.085, scale=10.0):
    """Sole centres at x = pelvis_x + x_rel (in front of the pelvis) and |y| = y_abs."""
    rs, Js = [], []
    pel = k.body_pos["base_link"]
    for f, sy in zip(FEET, (1, -1)):
        pc, Rb, jp, jr = k.body_pos[f + "/sole"]
        rs += [scale * (pc[0] - (pel[0] + x_rel)), scale * (pc[1] - sy * y_abs)]
        Js += [scale * jp[0], scale * jp[1]]
    return Task("foot_place", np.array(rs), np.vstack(Js), w)


def t_posture(S: Schema, k: Kin, w, qref, joints):
    jidx = [JOINTS.index(j) for j in joints]
    q = k.qpos[S.r.qadr]
    J = np.zeros((len(jidx), S.nv))
    for i, j in enumerate(jidx):
        J[i, S.jv[j]] = 1.0
    return Task("posture", q[jidx] - qref[jidx], J, w)


def feet_center(k: Kin):
    return 0.5 * (k.body_pos[FEET[0] + "/sole"][0] + k.body_pos[FEET[1] + "/sole"][0])


# ============================================================================ phases
@dataclass
class Phase:
    name: str
    tasks: callable            # (S, k) -> list[Task]
    done: callable             # (S, k, cert) -> bool
    c_star: callable           # (S, k) -> target CoM ground projection (support utility query)
    max_ticks: int = 400


def stand_ref(S):
    q = S.r.default_q(knee=0.35)
    return q


def make_phases(S: Schema):
    qstand = stand_ref(S)
    legs = JOINTS[:12]

    def tilt(k):
        return np.degrees(np.arccos(np.clip(k.R[2, 2], -1, 1)))

    def soles_flat(k, tol=0.006):
        ok = True
        for r in ("L_foot", "R_foot"):
            idx = S.region_points[r]
            zb = np.sort(k.z[idx])[:4]               # 4 lowest corners = sole bottom
            ok &= bool(zb.max() < tol)
        return ok

    def sit_com_target(k, ahead=0.04):
        """Imagined sitting CoM: ground projection just in front of the pelvis, toward the feet
        (inside the pelvis-edge / legs support region that remains once the back is released)."""
        pel = k.body_pos["base_link"]
        fc = feet_center(k)
        u = (fc - pel)[:2]; u = u / max(np.linalg.norm(u), 1e-9)
        return np.array([pel[0] + ahead * u[0], pel[1] + ahead * u[1], 0.0])

    def p1_tasks(S, k):
        return [t_trunk_upright(k, 30.0), t_com(k, 30.0, sit_com_target(k), axes=(0, 1))]

    def p1_done(S, k, cert):
        return tilt(k) < 15.0

    def p2_tasks(S, k):
        return [t_foot_flat(k, 10.0), t_foot_place(k, 10.0, x_rel=0.16),
                t_region_down(S, k, "L_foot", 10.0), t_region_down(S, k, "R_foot", 10.0),
                t_trunk_upright(k, 5.0, lean=0.25)]

    def p2_done(S, k, cert):
        fc = feet_center(k)
        return soles_flat(k) and abs(fc[0] - (k.body_pos["base_link"][0] + 0.16)) < 0.03

    def p3_tasks(S, k):
        fc = feet_center(k)
        return [t_com(k, 30.0, fc, axes=(0, 1)), t_region_up(S, k, "pelvis", 0.04, 10.0),
                t_foot_flat(k, 5.0), t_trunk_upright(k, 2.0, lean=0.5)]

    def p3_done(S, k, cert):
        d = S.region_height(k)
        return d[REGIONS.index("pelvis")] > 0.03 and soles_flat(k) and cert.get("feet_only", False)

    def p4_tasks(S, k):
        fc = feet_center(k)
        return [t_com(k, 30.0, fc, axes=(0, 1)), t_com(k, 10.0, [0, 0, 0.30], axes=(2,)),
                t_trunk_upright(k, 10.0), t_foot_flat(k, 5.0),
                t_posture(S, k, 1.0, qstand, legs)]

    def p4_done(S, k, cert):
        return (k.qpos[2] >= 0.85 * Z_STAND_NOM and tilt(k) <= 20.0 and cert.get("feet_only", False))

    return [Phase("P1_head_up", p1_tasks, p1_done, lambda S, k: sit_com_target(k)),
            Phase("P2_tuck", p2_tasks, p2_done, lambda S, k: k.com),
            Phase("P3_waist_up", p3_tasks, p3_done, lambda S, k: feet_center(k)),
            Phase("P4_stand", p4_tasks, p4_done, lambda S, k: feet_center(k))]


# ============================================================================ MPC
@dataclass
class MPCParams:
    N: int = 4
    rho_cap: float = 0.8
    mu: float = 0.6
    eps: float = EPS_CONTACT
    win: float = 0.012          # points below this height get force variables
    pen_win: float = 0.08       # points below this height get non-penetration rows
    tr_joint: float = 0.03
    tr_base_lin: float = 0.008
    tr_base_ang: float = 0.03
    jl_margin: float = 0.02
    d_min: float = 0.003
    F_s: float = 3.0            # max new load an unloaded point may pick up per tick [N]
    F_hold: float = 2.4         # a point carrying more than this is a held (sticking) contact
    a_gate: float = 0.05        # support attention needed to pick up new load
    delta_s: float = 0.004
    w_m: float = 0.05
    a0: float = 0.02
    w_e: float = 0.5
    w_rho: float = 1.0
    w_f: float = 2.0
    w_t: float = 0.02
    w_smooth: float = 0.5
    w_base: float = 0.01
    w_land: float = 20.0
    a_land: float = 0.08        # a^des threshold for an effector landing sub-task
    use_selfcol: bool = True
    use_jlim: bool = True


class QSMPC:
    def __init__(self, S: Schema, attn: SchemaAttention, p: MPCParams = MPCParams()):
        self.S, self.A, self.p = S, attn, p
        self.nv = S.nv
        self.reset_warm()

    def held_points(self, k: Kin):
        return {pi for pi, f in self.fz_bar[0].items() if f > self.p.F_hold and k.z[pi] <= self.p.eps}

    def project(self, q, iters=3):
        """Constraint-manifold projection after executing a step: held contacts back exactly onto
        the floor (z = 0) at their stick anchors (xy).  Removes linearisation drift so that held
        supports cannot creep off the floor over many ticks.  Damped least-norm Gauss-Newton."""
        S = self.S
        k = S.kin(q, with_G=False, with_pairs=False)
        held = sorted(self.held_points(k))
        self.anchors = {pi: self.anchors.get(pi, k.xy[pi].copy()) for pi in held} if hasattr(self, "anchors") else \
            {pi: k.xy[pi].copy() for pi in held}
        if not held and k.z.min() >= -0.0005:
            return q
        q0j = q[S.r.qadr].copy()
        for _ in range(iters):
            pen = [int(i) for i in np.where(k.z < -0.0005)[0] if int(i) not in held]
            r = np.array([v for pi in held for v in (k.z[pi], *(k.xy[pi] - self.anchors[pi]))]
                         + [k.z[i] for i in pen])
            if len(r) == 0:
                break
            Jm = np.vstack([k.Jp[pi][[2, 0, 1]] for pi in held] + [k.Jp[pi][2:3] for pi in pen])
            if np.abs(r).max() < 1e-5:
                break
            dq = -Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(len(r)), r)
            # bounded correction: a drift correction must stay small (the residual is mm-scale; a
            # near-singular Jm otherwise produces rad-scale joint jumps) and respect joint limits
            dq *= min(1.0, 0.005 / max(np.abs(dq).max(), 1e-12))
            q = S.integrate(q, dq)
            qa = S.r.qadr
            q[qa] = np.clip(q[qa], np.minimum(S.lo, q0j), np.maximum(S.hi, q0j))
            k = S.kin(q, with_G=False, with_pairs=False)
        return q

    def reset_warm(self):
        N, nv = self.p.N, self.nv
        self.dq_bar = np.zeros((N, nv))
        self.tau_bar = None
        self.fz_bar = [dict() for _ in range(N)]

    @staticmethod
    def _solve(P, qv, rows, lo, up, D=None):
        """Dense exact active-set QP (Goldfarb-Idnani, quadprog): reliable infeasibility verdicts.
        Variables are scaled x = D x~ (forces in units of mg, torques in units of tau_max)."""
        n = len(qv)
        D = np.ones(n) if D is None else D
        Ps = P * D[:, None] * D[None, :]
        Ps = 0.5 * (Ps + Ps.T) + 1e-7 * np.eye(n)
        qs = qv * D
        R = rows * D[None, :]
        nr = np.linalg.norm(R, axis=1); nr[nr < 1e-12] = 1.0
        R = R / nr[:, None]; l = lo / nr; u = up / nr
        eq = np.isclose(l, u)
        Ceq = R[eq]; beq = l[eq]
        lf = (~eq) & np.isfinite(l); uf = (~eq) & np.isfinite(u)
        C = np.vstack([Ceq, R[lf], -R[uf]])
        b = np.concatenate([beq, l[lf], -u[uf]])
        class Res: pass
        res = Res(); res.info = Res()
        try:
            sol = quadprog.solve_qp(Ps, -qs, C.T, b, meq=int(eq.sum()))
            res.x = sol[0] * D
            res.info.status = "solved"
        except ValueError as e:
            res.x = None
            res.info.status = "infeasible" if ("infeasible" in str(e) or "inconsistent" in str(e)) else f"error: {e}"
        return res

    @staticmethod
    def elastic(rows, lo, up, fam, tag, D):
        """Elastic feasibility LP: min sum of slacks s.t. lo - s <= A x <= up + s.
        Rows needing slack form an (approximate) irreducible infeasible set -> names the
        binding constraints (family, horizon node, point/pair)."""
        from scipy.optimize import linprog
        n = rows.shape[1]
        R = rows * D[None, :]
        nr = np.linalg.norm(R, axis=1); nr[nr < 1e-12] = 1
        R = R / nr[:, None]; l = lo / nr; u = up / nr
        m = len(l)
        eq = np.isclose(l, u)
        # variables x (n), s (m) >= 0 ; statics rows kept hard (they define the physics)
        hard = (np.array(fam) == "statics")
        A_ub, b_ub = [], []
        Iu = np.eye(m)
        lf = np.isfinite(l) & ~hard; uf = np.isfinite(u) & ~hard
        A_ub = np.vstack([np.hstack([-R[lf], -Iu[lf]]), np.hstack([R[uf], -Iu[uf]])])
        b_ub = np.concatenate([-l[lf], u[uf]])
        A_eq = np.hstack([R[hard], np.zeros((hard.sum(), m))])
        b_eq = l[hard]
        c = np.concatenate([np.zeros(n), np.ones(m)])
        bounds = [(None, None)] * n + [(0, None)] * m
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status != 0:
            return [("elastic_failed", res.status)]
        sl = res.x[n:]
        idx = np.argsort(-sl)
        return [(fam[i], tag[i], float(sl[i])) for i in idx if sl[i] > 1e-7][:12]

    # ------------------------------------------------------------------ one MPC tick
    def step(self, q0, phase: Phase, log: dict):
        S, p, N, nv = self.S, self.p, self.p.N, self.nv
        t0 = time.time()
        # linearisation trajectory
        qbar, kins = [], []
        q = q0.copy()
        for kk in range(N):
            q = S.integrate(q, self.dq_bar[kk])
            qbar.append(q.copy())
            kins.append(S.kin(q, with_G=True, with_pairs=p.use_selfcol))
        k0 = S.kin(q0, with_G=False, with_pairs=False)
        if all(len(f) == 0 for f in self.fz_bar):
            # cold start: current load distribution from the exact statics LP (which points are
            # already loaded, hence sticking)
            lp = S.statics_lp(k0, active=k0.z <= p.eps, mu=p.mu)
            if lp["feasible"]:
                f0 = {int(pi): float(lp["f"][c, 2]) for c, pi in enumerate(lp["idx"])}
                self.fz_bar = [dict(f0) for _ in range(N)]
        # ---- attention (at the current configuration, w.r.t. the phase sub-goal)
        tasks0 = phase.tasks(S, k0)
        JT = np.vstack([t.J * np.sqrt(t.w) for t in tasks0])
        eT = -np.concatenate([t.r * np.sqrt(t.w) for t in tasks0])
        sup = self.A.support(k0, JT, eT, phase.c_star(S, k0))
        a_cur = sup["a_cur"]
        support_regions = [i for i in range(len(REGIONS)) if sup["touching"][i] and a_cur[i] >= 0.02]
        mot = self.A.motion(k0, JT, eT, support_regions)
        a_j = mot["a_joint"]
        alpha = self.A.effort(self.tau_bar)
        if alpha is None:
            alpha = np.ones((N, 20))
        landing = [REGIONS[i] for i in range(len(REGIONS))
                   if REGIONS[i] in EFFECTORS and not sup["touching"][i] and sup["a_des"][i] >= p.a_land]
        # ---- variable layout
        nc = []
        cidx = []
        # held contacts are decided once per tick from the CURRENT loads (warm-start node 0) and
        # apply to every node of the horizon (consistent contact mode over the horizon)
        held = {pi for pi, f in self.fz_bar[0].items() if f > p.F_hold and k0.z[pi] <= p.eps}
        # SUPPORT ATTENTION SELECTS THE CONTACT SET: an unloaded point may pick up new load only
        # if its region is attended (a_cur >= a_gate); held points stay until unloaded (their
        # release is driven by the attention force price).
        # Fallback (only after an infeasible QP): open the gate -- every near point may carry load,
        # attention then only PRICES the forces.  Keeps the verdict about physics, not the gate.
        gate = (a_cur >= p.a_gate) | bool(getattr(self, "_open_gate", False))
        for kk in range(N):
            near = [int(i) for i in np.where(kins[kk].z <= p.eps)[0] if gate[S.p_region[i]]]
            idx = np.array(sorted(held | set(near)), int)
            cidx.append(idx); nc.append(len(idx))
        off = []
        o = 0
        for kk in range(N):
            off.append(dict(dq=o, tau=o + nv, rho=o + nv + 20, f=o + nv + 21))
            o += nv + 21 + 3 * nc[kk]
        nx = o
        P = np.zeros((nx, nx)); qv = np.zeros(nx)
        rows, lo, up, fam, tag = [], [], [], [], []
        cur = ["statics"]
        ctag = [None]

        def add(row, l, u):
            rows.append(row); lo.append(l); up.append(u); fam.append(cur[0]); tag.append(ctag[0])

        def E(kk):
            """e_k = L x - c :  returns (L (nv x nx), c (nv))."""
            L = np.zeros((nv, nx)); c = np.zeros(nv)
            for i in range(kk + 1):
                L[:, off[i]["dq"]:off[i]["dq"] + nv] += np.eye(nv)
                c += self.dq_bar[i]
            return L, c

        mg = S.mg
        jv = S.jv
        for kk in range(N):
            k = kins[kk]
            L, c = E(kk)
            od, ot, orho, of = off[kk]["dq"], off[kk]["tau"], off[kk]["rho"], off[kk]["f"]
            # ---------------- statics: sum J^T f + S^T tau - G e = g
            cur[0] = "statics"
            ctag[0] = (kk,)
            Aeq = np.zeros((nv, nx))
            for ci, pi in enumerate(cidx[kk]):
                Aeq[:, of + 3 * ci:of + 3 * ci + 3] = k.Jp[pi].T
            Aeq[jv, ot + np.arange(20)] += 1.0
            Aeq -= k.Gq @ L
            beq = k.g - k.Gq @ c
            for r_ in range(nv):
                add(Aeq[r_], beq[r_], beq[r_])
            # ---------------- contact forces
            s = p.mu / np.sqrt(2)
            for ci, pi in enumerate(cidx[kk]):
                fx, fy, fz = of + 3 * ci, of + 3 * ci + 1, of + 3 * ci + 2
                ctag[0] = (kk, int(pi), S.points[pi].region)
                cur[0] = "friction"
                for ax in (fx, fy):
                    r1 = np.zeros(nx); r1[ax] = 1; r1[fz] = -s; add(r1, -np.inf, 0)
                    r2 = np.zeros(nx); r2[ax] = -1; r2[fz] = -s; add(r2, -np.inf, 0)
                # f_z >= 0
                cur[0] = "unilateral"
                r_ = np.zeros(nx); r_[fz] = 1; add(r_, 0, np.inf)
                # contact state machine (from the warm start): a LOADED point (f > F_s) is a held
                # contact -- its 3-D position is fixed (stick, no lift-off while loaded); an
                # UNLOADED point within eps of the floor may move freely but may pick up at most
                # F_s this step (loads ramp up / down continuously across ticks).
                cur[0] = "stick_hold"
                loaded = int(pi) in held
                if loaded:
                    for ax in range(2):                      # stick (per-step slip <= 0.3 mm)
                        r_ = np.zeros(nx); r_[od:od + nv] = k.Jp[pi][ax]
                        add(r_, -0.0003, 0.0003)
                    Jz = k.Jp[pi][2]                         # held on the floor: z <= 1 mm, or no
                    zcap = max(0.001, k0.z[pi])             # higher than now (projection -> 0)
                    add(Jz @ L, -np.inf, zcap - k.z[pi] + Jz @ c)
                else:
                    r_ = np.zeros(nx); r_[fz] = 1; add(r_, -np.inf, p.F_s)
                # support attention force price
                ar = a_cur[S.p_region[pi]]
                P[fz, fz] += 2 * p.w_f / (ar + p.a0) / mg ** 2
                P[fx, fx] += 2 * p.w_t / mg ** 2
                P[fy, fy] += 2 * p.w_t / mg ** 2
            # ---------------- non-penetration
            cur[0] = "non_penetration"
            for pi in np.where(k.z <= p.pen_win)[0]:
                ctag[0] = (kk, int(pi), S.points[pi].region)
                Jz = k.Jp[pi][2]
                r_ = Jz @ L
                zmin = min(-0.0005, k0.z[pi])                    # never worse than now
                add(r_, zmin - k.z[pi] + Jz @ c, np.inf)
            # ---------------- torque limits / rho
            cur[0] = "torque_limit"
            ctag[0] = (kk,)
            for j in range(20):
                r1 = np.zeros(nx); r1[ot + j] = 1 / S.tau_max[j]; r1[orho] = -1; add(r1, -np.inf, 0)
                r2 = np.zeros(nx); r2[ot + j] = -1 / S.tau_max[j]; r2[orho] = -1; add(r2, -np.inf, 0)
            r_ = np.zeros(nx); r_[orho] = 1; add(r_, 0, p.rho_cap)
            # ---------------- joint limits
            cur[0] = "joint_limit"
            if p.use_jlim:
                qj = k.qpos[S.r.qadr]
                for j in range(20):
                    r_ = L[jv[j]]
                    # "never worse than now" (like non-penetration): the post-step projection can
                    # leave a joint inside the margin; then it may not go further, but need not jump
                    q0j = k0.qpos[S.r.qadr][j]
                    lo_j = min(S.lo[j] + p.jl_margin, q0j) - qj[j]
                    hi_j = max(S.hi[j] - p.jl_margin, q0j) - qj[j]
                    add(r_, lo_j + c[jv[j]], hi_j + c[jv[j]])
            # ---------------- self-collision
            cur[0] = "self_collision"
            for dist, Jd, gp in k.pairs:
                ctag[0] = (kk, S.m.geom(gp[0]).name or "trunk", S.m.geom(gp[1]).name or "trunk")
                r_ = Jd @ L
                lb = (p.d_min - dist) if dist >= p.d_min else 0.0
                add(r_, lb + Jd @ c, np.inf)
            # ---------------- trust region (box on dq_k)
            cur[0] = "trust_region"
            ctag[0] = (kk,)
            for i in range(nv):
                tr = p.tr_base_lin if i < 3 else (p.tr_base_ang if i < 6 else p.tr_joint)
                r_ = np.zeros(nx); r_[od + i] = 1; add(r_, -tr, tr)
            # ---------------- costs
            wk = 1.0 + kk / max(N - 1, 1)                     # later nodes weigh more
            tasks = phase.tasks(S, k)
            for lname in landing:
                tasks.append(t_region_down(S, k, lname, p.w_land * sup["a_des"][REGIONS.index(lname)]))
            for t in tasks:
                A_ = t.J @ L                                  # residual r + J (Lx - c)
                b_ = t.r - t.J @ c
                P += 2 * wk * t.w * A_.T @ A_
                qv += 2 * wk * t.w * A_.T @ b_
            for j in range(20):
                P[od + jv[j], od + jv[j]] += 2 * p.w_m / (a_j[j] + p.a0)
            for i in range(6):
                P[od + i, od + i] += 2 * p.w_base
            for j in range(20):
                P[ot + j, ot + j] += 2 * p.w_e * (1 + alpha[kk, j]) / S.tau_max[j] ** 2 / 20
            qv[orho] += p.w_rho
            if kk > 0:
                odp = off[kk - 1]["dq"]
                for i in range(nv):
                    P[od + i, od + i] += 2 * p.w_smooth
                    P[odp + i, odp + i] += 2 * p.w_smooth
                    P[od + i, odp + i] -= 2 * p.w_smooth
                    P[odp + i, od + i] -= 2 * p.w_smooth
        rows = np.array(rows); lo = np.array(lo); up = np.array(up); fam = np.array(fam)
        tag = list(tag)
        D = np.ones(nx)
        for kk in range(N):
            D[off[kk]["tau"]:off[kk]["tau"] + 20] = S.tau_max
            D[off[kk]["f"]:off[kk]["f"] + 3 * nc[kk]] = mg
        Pm = P
        self._D = D
        res = self._solve(Pm, qv, rows, lo, up, D)
        status = res.info.status
        ok = status == "solved"
        info = dict(status=status, nx=nx, ncons=len(rows), t=time.time() - t0)
        if not ok and not getattr(self, "_retry", False) and np.any(self.dq_bar != 0):
            # linearisation-trajectory artefact check: retry once from a cold warm start
            self.dq_bar = np.zeros((N, nv)); self._retry = True
            try:
                return self.step(q0, phase, log)
            finally:
                self._retry = False
        if not ok and not getattr(self, "_open_gate", False):
            # contact-state re-synchronisation (only after an infeasible QP): the held set / load
            # ramp limits come from the PREDICTED loads of the previous plan, which can drift from
            # the true load distribution after execution + projection.  Re-seed the contact state
            # from the exact statics LP at the current configuration (= cold start) and open the
            # attention gate (every near point may carry load; attention still prices forces).
            self.reset_warm(); self._open_gate = True
            try:
                dq_, info_ = self.step(q0, phase, log)
            finally:
                self._open_gate = False
            if dq_ is not None:
                self.n_gate_open = getattr(self, "n_gate_open", 0) + 1
                info_["gate_opened"] = True
                log["gate_opened"] = True
            return dq_, info_
        if not ok:
            # binding-constraint diagnosis: which constraint family, when dropped, restores
            # feasibility of this imagined step?
            diag = {}
            for f_ in sorted(set(fam)):
                if f_ == "statics":
                    continue
                keep = fam != f_
                r2 = self._solve(Pm, qv, rows[keep], lo[keep], up[keep], D)
                diag[f_] = r2.info.status
            info["infeasible_diag"] = diag
            info["elastic"] = self.elastic(rows, lo, up, fam, tag, D)
            self.reset_warm()
            info.update(sup=sup, mot=mot, alpha=alpha, landing=landing, dq=None)
            log.update(info)
            return None, info
        x = res.x
        dq = np.array([x[off[kk]["dq"]:off[kk]["dq"] + nv] for kk in range(N)])
        tau = np.array([x[off[kk]["tau"]:off[kk]["tau"] + 20] for kk in range(N)])
        rho = np.array([x[off[kk]["rho"]] for kk in range(N)])
        fz = []
        for kk in range(N):
            fz.append({int(pi): float(x[off[kk]["f"] + 3 * ci + 2]) for ci, pi in enumerate(cidx[kk])})
        # shift warm start (receding horizon)
        self.dq_bar = np.vstack([dq[1:], np.zeros((1, nv))])
        self.tau_bar = np.vstack([tau[1:], tau[-1:]])
        self.fz_bar = fz[1:] + [fz[-1]]
        info.update(sup=sup, mot=mot, alpha=alpha, landing=landing, rho_plan=rho, dq=dq[0], tau=tau[0],
                    fz=fz[0], task_cost=float(sum(t.w * np.sum(t.r ** 2) for t in tasks0)))
        log.update(info)
        return dq[0], info


# ============================================================================ certification
def certify(S: Schema, q, mu, eps_cert=0.005):
    k = S.kin(q, with_G=False, with_pairs=True, pair_dist=0.01)
    lp = S.statics_lp(k, active=k.z <= eps_cert, mu=mu)
    d = S.region_height(k)
    loaded = {r for r, v in lp.get("loads", {}).items() if v > 0.01}
    pen = float(k.z.min())
    selfd = min([dd for dd, _, _ in k.pairs], default=1.0)
    qj = q[S.r.qadr]
    jl = float(np.min(np.minimum(qj - S.lo, S.hi - qj)))
    return dict(rho=lp["rho"], binding=lp.get("binding"), loads=lp.get("loads", {}), feasible=lp["feasible"],
                feet_only=bool(loaded) and loaded <= {"L_foot", "R_foot"}, pen=pen, selfd=selfd, jl=jl,
                ratio=lp.get("ratio"), region_h=d)


# ============================================================================ phase runner
def run_phase(S, mpc: QSMPC, q, phase: Phase, mu, logs: list, stall_ticks=60, stall_tol=0.01,
              verbose=True):
    """Run one imagined sub-goal to completion, stagnation or failure.  Returns (q, outcome)."""
    hist = []
    for it in range(phase.max_ticks):
        log = dict(phase=phase.name, tick=it)
        dq, info = mpc.step(q, phase, log)
        if dq is None:
            log["cert"] = certify(S, q, mu)
            logs.append(log)
            return q, "QP_INFEASIBLE"
        q_try = None
        for sc in (1.0, 0.5, 0.25):
            # certification line search: an executed configuration must be an exact static
            # equilibrium (statics LP feasible, rho <= cap) -- otherwise shorten the step
            qc = mpc.project(S.integrate(q, dq * sc))
            cert = certify(S, qc, mu)
            if cert["feasible"] and cert["rho"] <= mpc.p.rho_cap:
                q_try = qc; log["step_scale"] = sc
                break
        if q_try is None:
            log["cert_violation"] = True          # flagged, counted against the verdict
            q_try = qc; log["step_scale"] = sc
        q = q_try
        log["cert"] = cert
        log["q"] = q.copy()
        logs.append(log)
        k = S.kin(q, with_G=False, with_pairs=False)
        cost = float(sum(t.w * np.sum(t.r ** 2) for t in phase.tasks(S, k)))
        log["task_cost_after"] = cost
        hist.append(cost)
        if verbose and it % 20 == 0:
            print(f"  [{phase.name} {it:3d}] cost {cost:9.4f} rho_cert {cert['rho']:.3f} ({cert['binding']}) "
                  f"z {q[2]:.3f} pen {cert['pen']*1e3:+.1f}mm self {cert['selfd']*1e3:+.1f}mm "
                  f"land {info['landing']} qp {info['t']*1e3:.0f}ms", flush=True)
        if phase.done(S, k, cert):
            return q, "DONE"
        if len(hist) > stall_ticks and hist[-stall_ticks] - cost < stall_tol * max(hist[-stall_ticks], 1e-6):
            return q, "STALLED"
    return q, "MAX_TICKS"
