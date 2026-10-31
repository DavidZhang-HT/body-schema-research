"""Whole-body controller for pi+ : full rigid-body-dynamics QP + DCM/footstep planner.

This is the per-instant *dynamic feasibility projection* (Phase 3 inner layer) and the
fast stabilisation layer (Phase 4, layer 1):

  min_x   sum_i w_i || A_i x - b_i ||^2                     (reference + control cost)
  s.t.    M(q;th) qdd + h(q,qd;th) = S^T tau + Jc^T lambda   (full floating-base dynamics)
          J_st qdd + Jdot_st qd = -Kv v_st                   (stance contact kinematics)
          lambda_corner in linearised friction cone (mu_ctrl), lambda_z >= 0 (unilateral)
          |tau| <= tau_max ;   joint limits via qdd bounds   (joint / torque limits)
  x = [qdd (26), lambda (3 x 4 corners x n_stance)]

All quantities (M, h, CoM, Jacobians) come from the controller's *internal body model*
with parameters theta_hat (nominal URDF for baseline (a), identified for (b)/(c)).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import numpy as np
import mujoco
import quadprog

from model import PiPlus, BodyParams, JOINTS, TAU_MAX, FOOT_CORNERS, SOLE_CENTER, SOLE_Z, ROOT

G = 9.81
SOLE_PT = np.array([SOLE_CENTER[0], 0.0, SOLE_Z])   # sole-centre point in ankle_roll_link frame


@dataclass
class GaitParams:
    """Decision variables of the outer (Phase 3) optimisation. Nothing here is hard-coded
    in the controller: trunk pitch, step width, cadence (T_step), stride all come from z."""
    T_step: float = 0.45        # one step duration (cadence = 1/T_step steps/s)
    ds_frac: float = 0.2        # double-support fraction of a step
    step_width: float = 0.16    # lateral sole-centre separation [m]
    stride: float = 0.10        # forward step length [m]
    trunk_pitch: float = 0.0    # base pitch target [rad]
    com_height: float = 0.30    # CoM height above ground [m]
    swing_height: float = 0.035
    k_dcm: float = 2.5          # DCM feedback gain
    zmp_margin: float = 0.015   # CoP margin inside the sole [m]

    def as_dict(self):
        return dict(self.__dict__)


def quat_to_mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)


def rot_err(R, Rd):
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def minjerk(u):
    u = np.clip(u, 0, 1)
    return 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4), (60 * u - 180 * u**2 + 120 * u**3)


class SwingProfile:
    """Normalised swing-height profile taken from the retargeted human reference."""

    def __init__(self):
        path = os.path.join(ROOT, "data", "ref_walk.npz")
        prof = None
        if os.path.exists(path):
            z = np.load(path)
            c, feet = z["contact"][:, 0], z["feet"][:, 0, 2]
            lo = [i for i in range(1, len(c)) if c[i - 1] and not c[i]]
            td = [i for i in range(1, len(c)) if not c[i - 1] and c[i]]
            if lo and td:
                a = lo[0]; b = min([t for t in td if t > a], default=None)
                if b is not None and b - a > 4:
                    seg = feet[a:b + 1] - np.linspace(feet[a], feet[b], b - a + 1)
                    seg = np.maximum(seg, 0)
                    if seg.max() > 1e-3:
                        prof = seg / seg.max()
        # the raw human ankle-height curve is jerky (segmentation noise); keep only its
        # timing (normalised peak time u*) and use a smooth beta-shaped profile u^a (1-u)^b
        u_star = 0.5
        if prof is not None:
            u_star = float(np.argmax(prof)) / (len(prof) - 1)
        self.u_star_ref = u_star
        self.u_star = float(np.clip(u_star, 0.4, 0.6))
        a = 5 * self.u_star; b = 5 * (1 - self.u_star)
        self.u = np.linspace(0, 1, 201)
        p = self.u ** a * (1 - self.u) ** b
        self.p = p / p.max()
        self.dp = np.gradient(self.p, self.u); self.ddp = np.gradient(self.dp, self.u)

    def __call__(self, u):
        u = float(np.clip(u, 0, 1))
        return np.interp(u, self.u, self.p), np.interp(u, self.u, self.dp), np.interp(u, self.u, self.ddp)


class BodyModel:
    """Internal model used by the controller (theta_hat) + its posterior covariance."""

    def __init__(self, params: BodyParams):
        self.r = PiPlus(params)
        self.m, self.d = self.r.m, self.r.d
        self.p = copy.deepcopy(params)
        self.nv = self.m.nv
        self.jacp = np.zeros((3, self.nv)); self.jacr = np.zeros((3, self.nv))
        self.M = np.zeros((self.nv, self.nv))

    def set_params(self, p: BodyParams):
        self.r.set_params(p); self.p = copy.deepcopy(p)

    def load_state(self, o, extrapolate=0.0, qdd_last=None):
        d, r = self.d, self.r
        d.qpos[0:3] = o.base_pos; d.qpos[3:7] = o.base_quat
        d.qpos[r.qadr] = o.q - np.asarray(self.p.q0, float)   # zero-offset correction (q0_hat)
        R = quat_to_mat(o.base_quat)
        d.qvel[0:3] = o.base_linvel; d.qvel[3:6] = o.gyro
        d.qvel[r.vadr] = o.qd
        if extrapolate > 0:     # delay compensation: predict state at t + delay
            if qdd_last is not None:
                d.qvel[:] += qdd_last * extrapolate
            mujoco.mj_integratePos(self.m, d.qpos, d.qvel, extrapolate)
        mujoco.mj_forward(self.m, d)

    # kinematic helpers on current state
    def point(self, body, local):
        return self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ local

    def jac6(self, body, local):
        p = self.point(body, local)
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, p, body)
        J = np.vstack([self.jacp, self.jacr]).copy()
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacDot(self.m, self.d, jp, jr, p, body)
        dJqd = np.concatenate([jp @ self.d.qvel, jr @ self.d.qvel])
        return p, J, dJqd

    def jacp_point(self, body, p):
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, p, body)
        return self.jacp.copy()

    def com_jac(self):
        m, d = self.m, self.d
        Jc = np.zeros((3, self.nv)); dJ = np.zeros(3)
        mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        Mtot = m.body_subtreemass[0]
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        for b in range(1, m.nbody):
            if m.body_mass[b] < 1e-6:
                continue
            mujoco.mj_jacDot(m, d, jp, jr, d.xipos[b], b)
            dJ += m.body_mass[b] * (jp @ d.qvel)
        return d.subtree_com[0].copy(), Jc, dJ / Mtot

    def dyn(self):
        mujoco.mj_fullM(self.m, self.d, self.M)
        h = self.d.qfrc_bias.copy() - self.d.qfrc_passive   # passive (damping) treated as known
        return self.M, h


# =============================================================================
class Planner:
    """DCM planner + capture-point footstep adjustment (fast layer of Phase 4)."""

    def __init__(self, gp: GaitParams, w_min=0.102):
        self.gp = gp
        self.w_min = w_min
        self.mode = "stand"
        self.stand = {}          # stand-mode targets (Phase 2 excitation): xy,vxy,axy,dz,vz,az,pitch
        self.prof = SwingProfile()

    def omega(self):
        return np.sqrt(G / self.gp.com_height)

    def start_walk(self, t0, feet_xy, com_xy, n_steps, stride, first_swing="l", init_T=0.6):
        gp = self.gp
        self.t0 = t0; self.init_T = init_T
        self.n = n_steps
        L, R = feet_xy
        self.stance_seq = []           # (side, xy) for each step k=1..n
        self.S = []
        side = "r" if first_swing == "l" else "l"
        cur = {"l": L.copy(), "r": R.copy()}
        mid = 0.5 * (L + R)
        y_l = mid[1] + gp.step_width / 2; y_r = mid[1] - gp.step_width / 2
        x = 0.5 * (L[0] + R[0])
        # stance sequence: first stance = current foot of 'side'
        S = [(side, cur[side].copy())]
        for k in range(1, n_steps + 1):
            sw = "l" if S[-1][0] == "r" else "r"
            ramp = min(1.0, k / 2.0) if k < n_steps else 0.0
            x = (S[-1][1][0] + stride * ramp) if k < n_steps else S[-1][1][0]
            y = y_l if sw == "l" else y_r
            S.append((sw, np.array([x, y])))
        self.S = S                      # S[0..n] ; step k (1..n) has stance S[k-1], swing lands at S[k]
        self.k = 0                      # 0 = initial transfer phase
        self.plan_dcm()
        self.xi0 = com_xy.copy()
        self.mode = "walk"
        self.step_t0 = t0 + init_T
        self.swing_p0 = None
        self.landed_early = False

    def begin_step(self):
        """re-plan the DCM from the (adjusted) footsteps and freeze this step's reference."""
        self.plan_dcm()
        k = self.k
        if 1 <= k <= self.n:
            self.xi_ini_frozen = self.xi_ini[k].copy()
            self.xi_end_frozen = self.xi_end[k].copy()
            self.S_nom_k1 = self.S[k][1].copy()

    def T_k(self, k):
        return self.gp.T_step

    def plan_dcm(self, from_k=1):
        w = self.omega(); n = self.n
        vrp = [s[1] for s in self.S]
        xi_end = 0.5 * (vrp[n] + vrp[n - 1])
        self.xi_ini = [None] * (n + 1); self.xi_end = [None] * (n + 1)
        for k in range(n, 0, -1):
            self.xi_end[k] = xi_end
            self.xi_ini[k] = vrp[k - 1] + (xi_end - vrp[k - 1]) * np.exp(-w * self.T_k(k))
            xi_end = self.xi_ini[k]
        self.xi_final = 0.5 * (vrp[n] + vrp[n - 1])

    def stance_sides(self, t):
        """planned contact state: returns (set of stance sides, swing side or None, u_swing)."""
        if self.mode == "stand":
            return {"l", "r"}, None, 0.0
        if t < self.step_t0:
            return {"l", "r"}, None, 0.0
        k = self.k if self.k > 0 else 1
        tau = t - self.step_t0
        T = self.T_k(k); Tds = self.gp.ds_frac * T
        st = self.S[k - 1][0]; sw = self.S[k][0]
        if tau < Tds or k > self.n:
            return {"l", "r"}, None, 0.0
        return {st}, sw, (tau - Tds) / (T - Tds)

    def update(self, t, com, comv, foot_pos, contact_est):
        """Return CoM acc des (3), swing target (p,v,a) or None, stance set, CoP bounds."""
        gp = self.gp; w = self.omega()
        xy, vxy = com[:2], comv[:2]
        xi = xy + vxy / w
        info = {}
        if self.mode == "stand":
            sd = self.stand
            tgt = sd["xy"] if sd.get("xy") is not None else 0.5 * (foot_pos["l"][:2] + foot_pos["r"][:2])
            v, a = sd.get("vxy", np.zeros(2)), sd.get("axy", np.zeros(2))
            xi_ref = tgt + v / w
            dxi_ref = v + a / w
            p_cmd = xi_ref - dxi_ref / w + (1 + gp.k_dcm / w) * (xi - xi_ref)
            stance, sw, u = {"l", "r"}, None, 0.0
        else:
            if self.k == 0:
                if t >= self.step_t0:
                    self.k = 1
                    self.begin_step()
                else:
                    # cubic DCM transfer xi0 -> xi_ini[1]
                    s = (t - self.t0) / self.init_T
                    s = np.clip(s, 0, 1)
                    xe = self.xi_ini[1]; ve = w * (self.xi_ini[1] - self.S[0][1])
                    h00 = 2*s**3 - 3*s**2 + 1; h10 = s**3 - 2*s**2 + s; h01 = -2*s**3 + 3*s**2; h11 = s**3 - s**2
                    d00 = 6*s**2 - 6*s; d10 = 3*s**2 - 4*s + 1; d01 = -6*s**2 + 6*s; d11 = 3*s**2 - 2*s
                    T0 = self.init_T
                    xi_ref = h00 * self.xi0 + h01 * xe + h11 * ve * T0
                    dxi_ref = (d00 * self.xi0 + d01 * xe + d11 * ve * T0) / T0
                    vrp = xi_ref - dxi_ref / w
                    p_cmd = vrp + (1 + gp.k_dcm / w) * (xi - xi_ref)
                    stance, sw, u = {"l", "r"}, None, 0.0
            if self.k >= 1:
                k = self.k; T = self.T_k(k)
                tau = t - self.step_t0
                if k > self.n:
                    xi_ref = self.xi_final; vrp = xi_ref
                    stance, sw, u = {"l", "r"}, None, 0.0
                else:
                    vrp = self.S[k - 1][1]
                    xi_ref = vrp + (self.xi_ini_frozen - vrp) * np.exp(w * min(tau, T))
                    stance, sw, u = self.stance_sides(t)
                p_cmd = vrp + (1 + gp.k_dcm / w) * (xi - xi_ref)
                # ---- capture-point footstep adjustment (only during single support, first 70%)
                # current-step DCM reference is frozen at step start; next footstep is moved by the
                # predicted end-of-step DCM error (keeps the DCM offset of the plan).
                if sw is not None and u < 0.7:
                    t_rem = T - tau
                    pc = self.clamp_cop(p_cmd, stance, foot_pos)
                    xi_end_pred = pc + (xi - pc) * np.exp(w * t_rem)
                    e = xi_end_pred - self.xi_end_frozen
                    e = np.clip(e, [-0.08, -0.06], [0.08, 0.06])
                    newp = self.S_nom_k1 + e
                    st_xy = self.S[k - 1][1]
                    lat = newp[1] - st_xy[1]
                    sgn = 1 if sw == "l" else -1
                    lat = sgn * np.clip(sgn * lat, self.w_min, 0.26)
                    newp[1] = st_xy[1] + lat
                    newp[0] = np.clip(newp[0], st_xy[0] - 0.12, st_xy[0] + 0.22)
                    shift = newp - self.S[k][1]
                    for j in range(k, self.n + 1):
                        self.S[j] = (self.S[j][0], self.S[j][1] + shift)
                    info["step_adjust"] = float(np.linalg.norm(newp - self.S_nom_k1))
                # step transition
                if k <= self.n and (tau >= T or (sw is not None and contact_est.get(sw, False) and u > 0.85)):
                    self.k += 1
                    self.step_t0 = t
                    self.begin_step()
                    self.swing_p0 = None
                    stance, sw, u = {"l", "r"}, None, 0.0
        p_cmd = self.clamp_cop(p_cmd, stance, foot_pos)
        acc = np.zeros(3)
        acc[:2] = w**2 * (xy - p_cmd)
        if self.mode == "stand":
            sd = self.stand
            acc[2] = (120.0 * (gp.com_height + sd.get("dz", 0.0) - com[2]) + 22.0 * (sd.get("vz", 0.0) - comv[2])
                      + sd.get("az", 0.0))
        else:
            acc[2] = 120.0 * (gp.com_height - com[2]) - 22.0 * comv[2]
        info.update(xi=xi, xi_ref=xi_ref, p_cmd=p_cmd)
        swing = None
        if sw is not None:
            swing = self.swing_target(t, sw, u, foot_pos)
        return acc, swing, stance, sw, info

    def clamp_cop(self, p, stance, foot_pos):
        m = self.gp.zmp_margin
        pts = []
        for s in stance:
            c = foot_pos[s][:2]
            pts += [c + np.array([0.086 - m, 0.036 - m]), c + np.array([-0.086 + m, -0.036 + m])]
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        return np.clip(p, lo, hi)

    def swing_target(self, t, sw, u, foot_pos):
        k = self.k; T = self.T_k(k); Tss = T * (1 - self.gp.ds_frac)
        if self.swing_p0 is None:
            self.swing_p0 = foot_pos[sw].copy()
            self.swing_p0[2] = 0.0
        p0 = self.swing_p0; p1 = np.array([*self.S[k][1], 0.0])
        s, ds, dds = minjerk(u)
        h, dh, ddh = self.prof(u)
        p = p0 + (p1 - p0) * s; v = (p1 - p0) * ds / Tss; a = (p1 - p0) * dds / Tss**2
        p[2] = self.gp.swing_height * h; v[2] = self.gp.swing_height * dh / Tss; a[2] = self.gp.swing_height * ddh / Tss**2
        if u >= 1.0:   # late touchdown: keep pressing down slowly
            p[2] = -0.005; v[2] = -0.05; a[2] = 0
        return p, v, a


# =============================================================================
class WBC:
    def __init__(self, model: BodyModel, gp: GaitParams, mu_ctrl=0.6, w_min=0.102, ref=None):
        self.bm = model
        self.gp = gp
        self.mu = mu_ctrl
        self.planner = Planner(gp, w_min)
        r = model.r
        self.feet = {"l": r.foot_bid[0], "r": r.foot_bid[1]}
        self.vadr = r.vadr
        self.ref = ref                     # q_cycle (100 x 20) or None
        self.q_nom = r.default_q()
        self.delay_comp = 0.0
        self.fric_ff = 0.0                 # identified leg Coulomb friction (feed-forward)
        self.qdd_last = np.zeros(model.nv)
        self.last_tau = np.zeros(20)
        self.phase = 0.0
        self.stats = dict(qp_fail=0, n=0)
        self.contact_est = {"l": True, "r": True}
        self.grf_est = {"l": 0.0, "r": 0.0}
        self.ref_on = False

    def set_gait(self, gp):
        self.gp = gp; self.planner.gp = gp

    # ------------------------------------------------------------------ phase & posture
    def posture_ref(self, t):
        q = self.q_nom.copy(); qd = np.zeros(20)
        if self.ref is None or not self.ref_on:
            return q, qd
        pl = self.planner
        if pl.mode != "walk" or pl.k < 1 or pl.k > pl.n:
            return q, qd
        T = pl.T_k(pl.k)
        tau = np.clip((t - pl.step_t0) / T, 0, 1)
        st = pl.S[pl.k - 1][0]
        # phi=0 at left touchdown == start of a step whose stance foot is left
        phi = (0.0 if st == "l" else 0.5) + 0.5 * tau
        self.phase = phi
        n = len(self.ref)
        x = phi * n
        i0 = int(np.floor(x)) % n; i1 = (i0 + 1) % n; a = x - np.floor(x)
        q = (1 - a) * self.ref[i0] + a * self.ref[i1]
        qd = (self.ref[i1] - self.ref[i0]) * n / (2 * T)
        return q, qd

    # ------------------------------------------------------------------ contact estimation
    def estimate_grf(self, o):
        """Vertical GRF per foot from leg joint torques (quasi-static, no F/T sensors)."""
        bm = self.bm; d = bm.d
        M, h = bm.dyn()
        out = {}
        for i, s in enumerate(("l", "r")):
            cols = self.vadr[i * 6:(i + 1) * 6]
            p, J, _ = bm.jac6(self.feet[s], SOLE_PT)
            Jl = J[:, cols]                       # 6x6 leg Jacobian
            tau_leg = o.tau[i * 6:(i + 1) * 6]
            g_leg = h[cols] + M[cols, :] @ self.qdd_last
            try:
                F = np.linalg.solve(Jl.T, g_leg - tau_leg)   # tau = h - J^T F  -> F = J^-T (h - tau)
            except np.linalg.LinAlgError:
                F = np.zeros(6)
            out[s] = float(F[2])
        return out

    # ------------------------------------------------------------------ main
    def control(self, o, t):
        bm = self.bm
        bm.load_state(o, self.delay_comp, self.qdd_last if self.delay_comp > 0 else None)
        d, m = bm.d, bm.m
        nv = bm.nv
        foot_pos = {s: bm.point(b, SOLE_PT) for s, b in self.feet.items()}
        # contact estimation (layer 2): GRF + foot height
        grf = self.estimate_grf(o)
        mg = m.body_subtreemass[0] * G
        for s in ("l", "r"):
            self.grf_est[s] = grf[s]
            on = grf[s] > 0.3 * mg and foot_pos[s][2] < 0.006
            off = grf[s] < 0.08 * mg
            if on: self.contact_est[s] = True
            elif off: self.contact_est[s] = False
        com, Jcom, dJcom = bm.com_jac()
        comv = Jcom @ d.qvel
        acc_com, swing, stance, sw, info = self.planner.update(t, com, comv, foot_pos, self.contact_est)
        M, h = bm.dyn()
        # ---------------- tasks
        A_list, b_list, w_list = [], [], []
        # CoM
        A_list.append(Jcom); b_list.append(acc_com - dJcom); w_list.append(np.array([100, 100, 60.0]))
        # trunk orientation
        R = d.xmat[bm.r.base].reshape(3, 3)
        pitch_t = self.gp.trunk_pitch + (self.planner.stand.get("pitch", 0.0) if self.planner.mode == "stand" else 0.0)
        Rd = rot_y(pitch_t)
        yaw_t = self.planner.stand.get("yaw", 0.0) if self.planner.mode == "stand" else 0.0
        if yaw_t:                                     # Phase-2b trunk yaw twist (feet planted)
            cy_, sy_ = np.cos(yaw_t), np.sin(yaw_t)
            Rd = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1.0]]) @ Rd
        # keep yaw = heading of the feet
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cz, sz = np.cos(yaw * 0.0), np.sin(yaw * 0.0)
        mujoco.mj_jacBody(m, d, bm.jacp, bm.jacr, bm.r.base)
        Jw = bm.jacr.copy()
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        mujoco.mj_jacDot(m, d, jp, jr, d.xpos[bm.r.base], bm.r.base)
        wb = Jw @ d.qvel
        alpha = 150.0 * rot_err(R, Rd) - 22.0 * wb
        A_list.append(Jw); b_list.append(alpha - jr @ d.qvel); w_list.append(np.full(3, 40.0))
        # swing foot
        if sw is not None:
            p, J, dJ = bm.jac6(self.feet[sw], SOLE_PT)
            pd, vd, ad = swing
            v = J @ d.qvel
            Rf = d.xmat[self.feet[sw]].reshape(3, 3)
            a_lin = ad + 500.0 * (pd - p) + 45.0 * (vd - v[:3])
            a_ang = 300.0 * rot_err(Rf, np.eye(3)) - 30.0 * v[3:]
            A_list.append(J); b_list.append(np.concatenate([a_lin, a_ang]) - dJ)
            w_list.append(np.array([60, 60, 60, 20, 20, 20.0]))
        # posture (reference fidelity in the null space)
        q = o.q; qd_j = o.qd
        qr, qdr = self.posture_ref(t)
        Ap = np.zeros((20, nv)); Ap[np.arange(20), self.vadr] = 1.0
        bp = 80.0 * (qr - d.qpos[bm.r.qadr]) + 12.0 * (qdr - d.qvel[self.vadr])
        wp = np.array([0.01] * 12 + [3.0] * 8)
        A_list.append(Ap); b_list.append(bp); w_list.append(wp)
        # ---------------- contacts
        st_list = sorted(stance)
        nc = 4 * len(st_list)
        nx = nv + 3 * nc
        Jc_rows = []; Jfeet = []; dJfeet = []
        for s in st_list:
            b = self.feet[s]
            Rf = d.xmat[b].reshape(3, 3)
            for c in FOOT_CORNERS:
                pc = d.xpos[b] + Rf @ c
                Jc_rows.append(bm.jacp_point(b, pc))
            p, J, dJ = bm.jac6(b, SOLE_PT)
            Jfeet.append(J); dJfeet.append(dJ + 20.0 * (J @ d.qvel))
        Jc = np.vstack(Jc_rows) if Jc_rows else np.zeros((0, nv))
        # ---------------- cost
        Hc = np.zeros((nx, nx)); gc = np.zeros(nx)
        for A, b, w in zip(A_list, b_list, w_list):
            Aw = A * w[:, None]
            Hc[:nv, :nv] += A.T @ Aw
            gc[:nv] += Aw.T @ b
        Hc[:nv, :nv] += 1e-4 * np.eye(nv)
        Hc[nv:, nv:] += 1e-3 * np.eye(3 * nc)
        # torque minimisation (control cost):  tau = Ma qdd + ha - Jca^T f
        Ma = M[6:, :]; ha = h[6:]
        Tq = np.hstack([Ma, -Jc[:, 6:].T]) if nc else Ma
        wt = 2e-3
        Hc += wt * Tq.T @ Tq; gc -= wt * Tq.T @ ha
        # ---------------- equality constraints
        Aeq = [np.hstack([M[:6, :], -Jc[:, :6].T])]
        beq = [-h[:6]]
        for J, dJ in zip(Jfeet, dJfeet):
            Aeq.append(np.hstack([J, np.zeros((6, 3 * nc))])); beq.append(-dJ)
        Aeq = np.vstack(Aeq); beq = np.concatenate(beq)
        # ---------------- inequality constraints  (C x >= c)
        Cin = []; cin = []
        Cin.append(Tq); cin.append(-TAU_MAX[np.argsort(self.vadr)] * 0 - TAU_MAX_DOF(self) - ha)
        Cin.append(-Tq); cin.append(-TAU_MAX_DOF(self) + ha)
        if nc:
            mu = self.mu
            F = np.zeros((5 * nc, 3 * nc))
            for i in range(nc):
                F[5*i+0, 3*i:3*i+3] = [-1, 0, mu]
                F[5*i+1, 3*i:3*i+3] = [1, 0, mu]
                F[5*i+2, 3*i:3*i+3] = [0, -1, mu]
                F[5*i+3, 3*i:3*i+3] = [0, 1, mu]
                F[5*i+4, 3*i:3*i+3] = [0, 0, 1]
            Cin.append(np.hstack([np.zeros((5 * nc, nv)), F])); cin.append(np.zeros(5 * nc))
        # joint limits: q + qd h + 0.5 qdd h^2 in [lo, hi]
        hh = 0.04
        qj = d.qpos[bm.r.qadr]; vj = d.qvel[self.vadr]
        lo = 2 * (bm.r.lo + 0.02 - qj - vj * hh) / hh**2
        hi = 2 * (bm.r.hi - 0.02 - qj - vj * hh) / hh**2
        Cin.append(np.hstack([Ap, np.zeros((20, 3 * nc))])); cin.append(np.minimum(lo, 0) if False else np.minimum(lo, hi - 1e-3))
        Cin.append(np.hstack([-Ap, np.zeros((20, 3 * nc))])); cin.append(-np.maximum(hi, lo + 1e-3))
        Cin = np.vstack(Cin); cin = np.concatenate(cin)
        Cq = np.vstack([Aeq, Cin]).T
        bq = np.concatenate([beq, cin])
        try:
            x = quadprog.solve_qp(Hc, gc, Cq, bq, len(beq))[0]
            ok = True
        except ValueError:
            ok = False
            self.stats["qp_fail"] += 1
            # fallback: drop torque & joint-limit rows (keep dynamics, contacts, friction)
            keep = [Aeq.T]
            bk = [beq]
            if nc:
                keep.append(np.hstack([np.zeros((5 * nc, nv)), F]).T); bk.append(np.zeros(5 * nc))
            try:
                x = quadprog.solve_qp(Hc, gc, np.hstack(keep), np.concatenate(bk), len(beq))[0]
            except ValueError:
                x = np.zeros(nx)
        self.stats["n"] += 1
        qdd = x[:nv]; f = x[nv:]
        tau_dof = Ma @ qdd + ha - (Jc[:, 6:].T @ f if nc else 0)
        tau = tau_dof[self.vadr - 6]
        # friction feed-forward on legs (identified Coulomb friction)
        if self.fric_ff > 0:
            vdes = d.qvel[self.vadr[:12]] + qdd[self.vadr[:12]] * 0.01
            tau[:12] += self.fric_ff * np.tanh(vdes / 0.05)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        self.qdd_last = qdd
        self.last_tau = tau
        # diagnostics for constraint-violation accounting
        fz = f[2::3] if nc else np.zeros(0)
        ft = np.sqrt(f[0::3] ** 2 + f[1::3] ** 2) if nc else np.zeros(0)
        info.update(ok=ok, stance=st_list, swing=sw, f=f, nc=nc, com=com, comv=comv,
                    tau_ratio=float(np.max(np.abs(tau) / TAU_MAX)),
                    fric_ratio=float(np.max(ft / np.maximum(fz, 1e-3))) if nc else 0.0,
                    foot_pos=foot_pos, grf=grf)
        return tau, info


def TAU_MAX_DOF(wbc):
    """torque limits ordered by MuJoCo dof index (6..25)."""
    out = np.zeros(20)
    out[wbc.vadr - 6] = TAU_MAX
    return out
