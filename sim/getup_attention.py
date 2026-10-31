"""Interpretable, schema-grounded attention for quasi-static get-up planning (F3).

No learned parameters.  Every attention weight is a closed-form function of explicit body-
schema quantities computed from the internal model (getup_schema.Schema.kin):

HEAD 1  SUPPORT attention  (force / statics understanding: "which body parts should bear load")
    keys   = contact regions r (head, back, pelvis, elbows, hands, thighs, knees, feet, ...)
    query  = the imagined sub-goal of the phase (task Jacobian J_T, residual e_T, target CoM c*)
    score  s_r^des = - w_k * kappa_r * F_bar        statics: joint-torque ratio consumed per
                                                     newton of support through the limb chain
                                                     (kappa_r = ||diag(1/tau_max) J_chain^T e_z||_inf)
                     - w_c * conflict_r              kinematics: fraction of the imagined motion
                                                     that becomes possible only if r is released
                                                     AND that motion lifts r off the floor
                                                     (unilateral: a region the motion presses into
                                                     the floor is a pivot, not an obstacle)
                     + w_u * u_r                     statics: closeness of r to the target CoM
                                                     ground projection (load-sharing utility)
           s_r^cur = s_r^des - max(d_r - eps, 0) / lambda_d     (d_r = height of r above floor)
    a^cur = softmax(s^cur / T)  -> where load CAN and SHOULD go now (MPC force price 1/(a+a0))
    a^des = softmax(s^des / T)  -> where load SHOULD go irrespective of current contact; an
                                   effector with high a^des that is not touching gets a landing
                                   sub-task ("to lift the waist I must put my hands down").

HEAD 2  MOTION attention  (kinematics understanding: "which joints realise the sub-goal")
    v* = (J_T N_S)^+ e_T   minimum-norm motion achieving the sub-goal while every attended
                           support region stays pinned (N_S = null-space projector of the
                           pinned support points);
    a_j = |v*_j| / sum_i |v*_i|   over the 20 actuated joints.
    MPC motion price on joint j = w_m / (a_j + a0): attended joints move, others stay put.

HEAD 3  EFFORT attention  (movement / dynamics-over-the-horizon understanding: "where along the
        imagined motion and in which joint is the body closest to its limits")
    alpha_{k,j} = softmax_{k,j}( |tau_bar_{k,j}| / tau_max_j / T_e )   over horizon nodes k and
    joints j, from the previous MPC solution.  It is exactly the gradient of the log-sum-exp
    (smooth max) of the torque ratios, so weighting the effort term by alpha turns the
    sum-of-squares effort into an iteratively re-weighted minimax (J2) objective.  Its joint
    marginal is the "bottleneck joint" read-out used in the F3 diagnosis.

All weights are logged per MPC tick by the caller for inspection/visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from getup_schema import REGIONS, EFFECTORS, Schema, Kin, EPS_CONTACT
from model import JOINTS


@dataclass
class AttnParams:
    w_kappa: float = 2.0      # statics price weight
    w_conf: float = 4.0       # kinematic task-conflict weight
    w_util: float = 2.0       # support utility weight
    sigma_u: float = 0.12     # [m] utility length scale
    lambda_d: float = 0.02    # [m] proximity length scale
    T: float = 0.5            # support softmax temperature
    T_e: float = 0.1          # effort softmax temperature (in torque-ratio units)


def softmax(x, T=1.0):
    x = np.asarray(x, float) / T
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class SchemaAttention:
    def __init__(self, S: Schema, p: AttnParams = AttnParams()):
        self.S, self.p = S, p
        self.nR = len(REGIONS)

    # ---------------------------------------------------------------- features
    def region_features(self, k: Kin):
        S = self.S
        d = S.region_height(k)
        xy = S.region_xy(k)
        low = np.zeros(self.nR, int)
        kappa = np.zeros(self.nR)
        ez = np.array([0, 0, 1.0])
        for i, r in enumerate(REGIONS):
            idx = S.region_points[r]
            j = idx[np.argmin(k.z[idx])]
            low[i] = j
            tau = (k.Jp[j].T @ ez)[S.jv]               # chain torques per newton (ancestors only)
            kappa[i] = np.max(np.abs(tau) / S.tau_max)
        return d, xy, low, kappa

    @staticmethod
    def _progress(JT, eT, Jc):
        """Fraction of the imagined task motion achievable while the rows Jc are held."""
        if Jc is None or len(Jc) == 0:
            N = np.eye(JT.shape[1])
        else:
            U, s, Vt = np.linalg.svd(Jc)
            rank = int((s > 1e-6 * max(s[0], 1e-12)).sum())
            V0 = Vt[rank:].T
            N = V0 @ V0.T
        A = JT @ N
        v = np.linalg.pinv(A, rcond=1e-6) @ eT
        ach = A @ v
        return float(np.linalg.norm(ach) / max(np.linalg.norm(eT), 1e-12)), N @ v

    # ---------------------------------------------------------------- heads
    def support(self, k: Kin, JT, eT, c_star, eps=EPS_CONTACT + 0.004):
        p = self.p
        d, xy, low, kappa = self.region_features(k)
        F_bar = 0.5 * self.S.mg
        touching = d <= eps
        # conflict: progress gain from releasing region r (others pinned at their lowest point)
        pin = {i: k.Jp[low[i]] for i in range(self.nR) if touching[i]}
        def stack(excl=None):
            rows = [J for i, J in pin.items() if i != excl]
            return np.vstack(rows) if rows else None
        pi_all, _ = self._progress(JT, eT, stack())
        conflict = np.zeros(self.nR)
        for i in pin:
            pi_wo, v = self._progress(JT, eT, stack(i))
            # unilateral contact: releasing r only helps if the imagined motion LIFTS r off the
            # floor (if it would push r into the floor, r is the pivot, not an obstacle)
            zdot = float(k.Jp[low[i]][2] @ v)
            if zdot > 1e-6:
                conflict[i] = max(0.0, pi_wo - pi_all)
        u = np.exp(-np.sum((xy - c_star[None, :2]) ** 2, 1) / (2 * p.sigma_u ** 2))
        s_des = -p.w_kappa * kappa * F_bar - p.w_conf * conflict + p.w_util * u
        prox = np.maximum(d - eps, 0) / p.lambda_d
        s_cur = s_des - prox
        a_cur = softmax(s_cur, p.T)
        a_des = softmax(s_des, p.T)
        return dict(a_cur=a_cur, a_des=a_des, d=d, kappa=kappa, conflict=conflict, util=u,
                    touching=touching, low=low, pi_all=pi_all, s_des=s_des, s_cur=s_cur)

    def motion(self, k: Kin, JT, eT, support_regions):
        S = self.S
        rows = [k.Jp[S.region_points[REGIONS[i]][np.argmin(k.z[S.region_points[REGIONS[i]]])]]
                for i in support_regions]
        Jc = np.vstack(rows) if rows else None
        pi, v = self._progress(JT, eT, Jc)
        vj = np.abs(v[S.jv])
        a = vj / vj.sum() if vj.sum() > 1e-12 else np.full(20, 1 / 20)
        return dict(a_joint=a, progress=pi, v=v)

    def effort(self, tau_bar):
        """tau_bar: (N, 20) torques of the previous plan -> alpha (N, 20), mean 1."""
        if tau_bar is None:
            return None
        r = np.abs(tau_bar) / self.S.tau_max[None, :]
        a = softmax(r.ravel(), self.p.T_e).reshape(r.shape)
        return a * r.size
