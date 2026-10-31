"""Joint-torque residual channel for joint zero-offset identification (Phase 2b, torque channel).

Physics (double support, both soles planted, no F/T sensors):

    M(q) qdd + h(q, qd; theta) = S^T tau_j + J_L^T F_L + J_R^T F_R          (18 + arms rows)
    tau_j = tau_meas - b qd - a qdd + f_fric ,   f_fric = -mu_f tanh(qd / v0)  (Coulomb, co-identified)

with q = q_meas - delta (legs), theta = (trunk_dm, trunk_dx, trunk_dy, trunk_dz).  The 12 leg rows
are solved EXACTLY for the two unknown sole wrenches (J_leg,i is 6x6 and well conditioned at the
>= 10 deg knee bend of the safe envelope):

    F_i = J_leg,i^-T ( h_leg,i - tau_j,i )

so the 12 leg torques carry exactly 6 independent equations per sample: the floating-base rows

    r = h_base(q; theta) - J_L,base^T F_L - J_R,base^T F_R          (6: force N, moment N m)

(the other 6 = internal / squeeze wrench between the feet, set by the controller -> free).
The existing SchemaEstimator uses 3 of these 6 (vertical force -> trunk_dm, roll/pitch moment
-> trunk CoM xy); the torque channel uses all 6, evaluated at q_meas - delta, as a residual.

Base velocity / acceleration (only needed for the small dynamic term) come from the left-sole
contact constraint J_L v = 0, J_L qdd + Jdot_L v = 0 with joint qd and a smoothed joint qdd
(Savitzky-Golay on the full-rate qd stream) -> no base-state estimator, no circularity.
IMU: roll/pitch only (base orientation), yaw is irrelevant.

Noise model per joint: sigma^2 = s_tau^2 + (rel tau)^2 + sigma_stick^2(qd), where the stick term
mu_f^2 (1 - tanh^2(qd/v0)) / 3 accounts for the indeterminate static friction of a (nearly)
stalled joint (uniform on [-mu_f, mu_f]); it is propagated to the 6 base residuals through
A_i = d r / d tau_j,i = J_base,i^T J_leg,i^-T  (exact, per sample).
"""
from __future__ import annotations

import numpy as np
import mujoco

from model import PiPlus
from wbc import SOLE_PT

NL = 12
NTR = 4                       # trunk_dm, trunk_dx, trunk_dy, trunk_dz
NRT = 6                       # torque-channel residuals per sample


def sg_derivative(x, dt, half=10):
    """centred Savitzky-Golay (quadratic) first derivative along axis 0 (edges: one-sided)."""
    n = len(x)
    k = np.arange(-half, half + 1)
    w = k / (dt * (k ** 2).sum())
    out = np.zeros_like(x)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        kk = np.arange(lo, hi) - i
        kk = kk - kk.mean()
        out[i] = (kk[:, None] * x[lo:hi]).sum(0) / (dt * (kk ** 2).sum())
    return out


def sg_derivative_at(x, idx, dt, half=10):
    """same quadratic SG first derivative as sg_derivative, evaluated only at rows idx."""
    n = len(x)
    out = np.zeros((len(idx),) + x.shape[1:])
    for o, i in enumerate(idx):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        kk = np.arange(lo, hi) - i
        kk = kk - kk.mean()
        out[o] = (kk[:, None] * x[lo:hi]).sum(0) / (dt * (kk ** 2).sum())
    return out


def whiten(R, J, C):
    """per-sample Cholesky whitening of residuals R (N,6) and Jacobian J (N,6,p) with Cov C."""
    L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[1])[None])
    Rw = np.linalg.solve(L, R[..., None])[..., 0]
    Jw = np.linalg.solve(L, J) if J is not None else None
    return Rw, Jw


class TorqueChannel:
    def __init__(self, s_tau=0.05, rel_tau=0.02, v0=0.02, dynamic=True):
        self.r = PiPlus()
        self.m, self.d = self.r.m, self.r.d
        self.s_tau, self.rel_tau, self.v0, self.dynamic = s_tau, rel_tau, v0, dynamic
        r = self.r
        self.legL, self.legR = r.vadr[:6], r.vadr[6:12]
        self.bdof = np.arange(6)
        self.damp = self.m.dof_damping[r.leg_v].copy()
        self.arm = self.m.dof_armature[r.leg_v].copy()
        self.m0, self.c0 = float(self.m.body_mass[r.base]), self.m.body_ipos[r.base].copy()
        self._jp = np.zeros((3, self.m.nv)); self._jr = np.zeros((3, self.m.nv))
        self._rne = np.zeros(self.m.nv)

    def set_trunk(self, tr):
        m, b = self.m, self.r.base
        m.body_mass[b] = self.m0 + tr[0]
        m.body_ipos[b] = self.c0 + np.array([tr[1], tr[2], tr[3]])

    def _jac(self, i):
        d, b = self.d, self.r.foot_bid[i]
        p = d.xpos[b] + d.xmat[b].reshape(3, 3) @ SOLE_PT
        mujoco.mj_jac(self.m, d, self._jp, self._jr, p, b)
        J = np.vstack([self._jp, self._jr])
        if not self.dynamic:
            return J, None
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacDot(self.m, d, jp, jr, p, b)
        return J, np.vstack([jp, jr])

    def eval(self, q12, qarm, quat, qd, qdd, tau12, fric12):
        """returns r (6), A_L (6x6), A_R (6x6).  fric12 = modelled friction torque (added to tau)."""
        m, d, r = self.m, self.d, self.r
        d.qpos[:] = 0; d.qpos[3:7] = quat
        d.qpos[r.qadr[:NL]] = q12; d.qpos[r.qadr[NL:]] = qarm
        d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        JL, _ = self._jac(0)
        JLb, JLl = JL[:, self.bdof], JL[:, self.legL]
        if self.dynamic:
            d.qvel[r.vadr] = qd
            d.qvel[:6] = -np.linalg.solve(JLb, JLl @ qd[:6])
            mujoco.mj_comVel(m, d)
            JL, JLd = self._jac(0)
            d.qacc[r.vadr] = qdd
            d.qacc[:6] = -np.linalg.solve(JLb, JLl @ qdd[:6] + JLd @ d.qvel)
        JR, _ = self._jac(1)
        mujoco.mj_rne(m, d, 1 if self.dynamic else 0, self._rne)
        h = self._rne
        tj = tau12 + fric12
        if self.dynamic:
            tj = tj - self.damp * qd[:NL] - self.arm * qdd[:NL]
        out = h[self.bdof].copy()
        A = []
        for J, cols, sl in ((JL, self.legL, slice(0, 6)), (JR, self.legR, slice(6, 12))):
            Jl_T = J[:, cols].T
            F = np.linalg.solve(Jl_T, h[cols] - tj[sl])
            out -= J[:, self.bdof].T @ F
            A.append(J[:, self.bdof].T @ np.linalg.inv(Jl_T))      # d r / d tau_j  (= +A)
        return out, A[0], A[1]

    # ------------------------------------------------------------------ batch
    def fric_model(self, qd12, mu):
        return -np.asarray(mu) * np.tanh(qd12 / self.v0)

    def residuals(self, delta, trunk, mu, S, jac=True, h=1e-6, mu_jac=True, v0=None):
        """S: dict of per-sample arrays q(20) qd(20) qdd(20) quat(4) tau(12).
        Returns r (N,6), J (N,6,NL+NTR+nmu[+1]) w.r.t. [delta, trunk, mu(, v0)], Cov (N,6,6) of r.
        v0 given -> the Coulomb transition velocity is co-identified too (extra last column)."""
        mu = np.atleast_1d(np.asarray(mu, float))
        nmu = len(mu)
        N = len(S["q"])
        co_v0 = v0 is not None
        v0 = self.v0 if v0 is None else float(v0)
        npar = NL + NTR + nmu + (1 if co_v0 else 0)
        R = np.zeros((N, NRT)); Jt = np.zeros((N, NRT, npar)) if jac else None
        C = np.zeros((N, NRT, NRT))
        mu12 = np.broadcast_to(mu, (NL,)) if nmu == 1 else mu
        for k in range(N):
            q, qd, qdd, quat, tau = S["q"][k], S["qd"][k], S["qdd"][k], S["quat"][k], S["tau"][k]
            qh = q[:NL] - delta; qarm = q[NL:]
            th = np.tanh(qd[:NL] / v0)
            fr = -mu12 * th
            self.set_trunk(trunk)
            r0, AL, AR = self.eval(qh, qarm, quat, qd, qdd, tau, fr)
            R[k] = r0
            A = np.hstack([AL, AR])                         # 6 x 12  d r / d tau_j
            sig2 = self.s_tau ** 2 + (self.rel_tau * tau) ** 2 + mu12 ** 2 * (1 - th ** 2) / 3.0
            C[k] = (A * sig2[None, :]) @ A.T
            if not jac:
                continue
            for j in range(NL):
                dq = np.zeros(NL); dq[j] = h
                rj, _, _ = self.eval(qh - dq, qarm, quat, qd, qdd, tau, fr)   # d q_true / d delta = -1
                Jt[k, :, j] = (rj - r0) / h
            for j in range(NTR):
                t2 = np.array(trunk, float); t2[j] += h * (1e3 if j == 0 else 1.0)
                self.set_trunk(t2)
                rj, _, _ = self.eval(qh, qarm, quat, qd, qdd, tau, fr)
                Jt[k, :, NL + j] = (rj - r0) / (h * (1e3 if j == 0 else 1.0))
            self.set_trunk(trunk)
            # friction: r depends on tau_j linearly -> d r / d mu = A @ (-tanh)
            if nmu == 1:
                Jt[k, :, NL + NTR] = A @ (-th)
            else:
                Jt[k, :, NL + NTR:NL + NTR + nmu] = A * (-th)[None, :]
            if co_v0:                                       # d fr / d v0 = mu (1 - tanh^2) qd / v0^2
                Jt[k, :, -1] = A @ (mu12 * (1 - th ** 2) * qd[:NL] / v0 ** 2)
        self.set_trunk(np.zeros(NTR))
        return R, Jt, C


def prepare(rec, idx, dt=0.002, use_true=False, half=10):
    """build the per-sample input dict from a full-rate recording at sample indices idx.
    qdd from a Savitzky-Golay derivative of the (measured) full-rate qd stream."""
    qd_all = rec["qd"]
    qdd_all = sg_derivative(qd_all, dt, half)
    S = dict(q=rec["q"][idx].copy(), qd=qd_all[idx].copy(), qdd=qdd_all[idx].copy(),
             quat=rec["quat"][idx].copy(), tau=rec["tau"][idx, :NL].copy())
    if use_true:
        tr = rec["truth"]
        S["q"][:, :NL] = tr[idx, :NL]
        S["qd"][:, :NL] = tr[idx, 20:32]
        S["qdd"][:, :NL] = tr[idx, 40:52]
    return S
