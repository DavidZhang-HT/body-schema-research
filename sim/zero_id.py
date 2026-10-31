"""Joint zero-offset identification from double-support loop closure (proprioception + IMU).

Measurement model:  q_meas = q_true + delta  (delta = per-joint encoder zero offset, 12 leg joints)

Constraints that hold during Phase-2 double-support exploration (both soles flat on a flat floor,
no slip), evaluated with the controller's kinematic model at q_meas - delta:

  (G) gravity / IMU :  each sole is level:  (R_bf^T g_b)_{x,y} = 0      2 per foot   (sigma_g)
        g_b = IMU 'up' vector in the base frame (roll/pitch only; IMU yaw is NOT used)
  (R) relative rot  :  R_LR = R_bL^T R_bR = Rz(psi)                      3            (sigma_ra)
  (P) relative pos  :  R_bL^T (p_R - p_L) = (x, y, 0)                    3            (sigma_p)

  (x, y, psi) = relative foot placement, unknown but constant -> nuisance parameters.
  The base position estimate is deliberately NOT used: on a real robot it comes from leg
  kinematics itself (circular).  A 'known stance' variant (calibration jig) fixes (x, y, psi).

Estimator: MAP Gauss-Newton over [delta(12), x, y, psi] with prior delta ~ N(0, sigma_prior^2),
outlier rejection, Schur-complement information analysis.

Gating (Phase-4 lesson):
  * sample gate  : planned double support, both feet estimated in contact and loaded
                   (> 10 % body weight each), quasi-static (|qd_leg| < 1 rad/s, |gyro| < 0.4 rad/s,
                   |acc| ~ g), >= 0.15 s after the last touchdown -> no impact/transient samples.
  * commit gate  : >= N_min samples; only the components of delta along eigen-directions of the
                   (marginal, misfit-scaled) data information that are >= kappa x the prior
                   information are written; everything else stays on the prior (0);
                   and the committed update must pass an evidence test (chi2 drop > 2 dof).
"""
from __future__ import annotations

import numpy as np
import mujoco

from model import PiPlus, LEG_JOINTS
from wbc import SOLE_PT, quat_to_mat, G
from zero_torque import TorqueChannel, sg_derivative_at, whiten, NTR, NRT

NL = 12
NP = NL + 3                         # delta + nuisance (x, y, psi)
NR = 10                             # residuals per sample: G(4) R(3) P(3)
NA = NP + NR                        # + constant systematic bias per residual channel


def _vee_skew(R):
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class ZeroEstimator:
    def __init__(self, prior_std=np.deg2rad(2.0), sig_g=3e-3, sig_ra=1e-3, sig_p=1e-3,
                 decim=5, kappa=16.0, n_min=150, sys_g=1e-6, sys_ra=1e-6, sys_p=1e-6, use_tau=True,
                 torque=False, s_tau=0.05, rel_tau=0.02, mu_prior=(0.2, 0.3), co_v0=True, v0_prior=(0.03, 0.02),
                 trunk_prior_std=(0.6, 0.02, 0.015, 0.03), tq_gate=6.0, dt=0.002):
        self.use_tau = use_tau
        # ---- joint-torque residual channel (Phase 2b torque channel, see zero_torque.py).
        # Default OFF -> the loop-closure-only estimator and every published number are unchanged.
        self.torque = torque
        if torque:
            self.tc = TorqueChannel(s_tau=s_tau, rel_tau=rel_tau)
            self.mu_prior = np.array(mu_prior, float)            # frictionloss (mean, std) [N m]
            # Coulomb transition velocity v0 of the friction law -mu tanh(qd / v0): co-identified
            # by default (a fixed v0 = 0.02 rad/s biases mu low by 13-26 %, see results.md)
            self.co_v0, self.v0_prior = co_v0, np.array(v0_prior, float)
            self.nf = 2 if co_v0 else 1                          # friction unknowns [mu(, v0)]
            self.iMU = NA + NTR
            self.trunk_prior_std = np.array(trunk_prior_std, float)
            self.tq_gate, self.dt = tq_gate, dt
            self._tick_qd = []                                  # full-rate joint velocity stream
            self.S = dict(q=[], qd=[], quat=[], tau=[], tick=[])
        self.r = PiPlus()
        self.m, self.d = self.r.m, self.r.d
        self.prior_std = prior_std
        self.sig = np.array([sig_g] * 4 + [sig_ra] * 3 + [sig_p] * 3)
        # optional constant per-episode systematic bias per channel (IMU mounting etc.).
        # Default OFF (sigma ~ 0): tested, it absorbs the absolute gravity / relative-tilt
        # constraints that carry most of the information (see results.md, q0 appendix).
        self.sig_sys = np.array([sys_g] * 4 + [sys_ra] * 3 + [sys_p] * 3)
        self.decim, self.kappa, self.n_min = decim, kappa, n_min
        self.Q, self.Gb, self.T = [], [], []
        self.n_seen = 0; self.n_gate_rej = 0
        self.last_td = -1.0; self._prev_stance = None
        self._k = 0

    # ------------------------------------------------------------------ sample gate
    def admissible(self, o, info, wbc, t):
        st = info.get("stance", [])
        if self._prev_stance is not None and len(st) > len(self._prev_stance):
            self.last_td = t
        self._prev_stance = st
        if len(st) != 2 or t - self.last_td < 0.15:
            return False
        if not (wbc.contact_est["l"] and wbc.contact_est["r"]):
            return False
        mg = wbc.bm.m.body_subtreemass[0] * G
        if min(wbc.grf_est["l"], wbc.grf_est["r"]) < 0.10 * mg:
            return False
        an = np.linalg.norm(o.acc)
        if abs(an - G) > 0.6 or np.abs(o.qd[:NL]).max() > 1.0 or np.linalg.norm(o.gyro) > 0.4:
            return False
        return True

    def add(self, o, info, wbc, t):
        self.n_seen += 1
        if self.torque:
            self._tick_qd.append(np.asarray(o.qd, float).copy())
        if not self.admissible(o, info, wbc, t):
            self.n_gate_rej += 1
            return False
        self._k += 1
        if self._k % self.decim:
            return False
        self.Q.append(o.q[:NL].copy())
        self.Gb.append(quat_to_mat(o.base_quat).T @ np.array([0, 0, 1.0]))
        self.T.append(t)
        if self.torque:
            for k, v in (("q", o.q), ("qd", o.qd), ("quat", o.base_quat), ("tau", o.tau[:NL])):
                self.S[k].append(np.asarray(v, float).copy())
            self.S["tick"].append(len(self._tick_qd) - 1)
        return True

    def load_recording(self, rec):
        """fill the estimator from a full-rate recording (zero_torque_record.py npz): the gated
        rows are exactly the samples add() accepted in that run (offline re-solve, no re-sim)."""
        idx = np.flatnonzero(rec["gated"])
        self.Q = list(rec["q"][idx, :NL]); self.T = list(rec["t"][idx])
        self.Gb = [quat_to_mat(qq).T @ np.array([0, 0, 1.0]) for qq in rec["quat"][idx]]
        if self.torque:
            self._tick_qd = list(rec["qd"])
            self.S = dict(q=list(rec["q"][idx]), qd=list(rec["qd"][idx]), quat=list(rec["quat"][idx]),
                          tau=list(rec["tau"][idx, :NL]), tick=list(idx))

    # ------------------------------------------------------------------ kinematics
    def _feet(self, q12):
        d = self.d
        d.qpos[:] = 0; d.qpos[3] = 1.0
        d.qpos[self.r.qadr[:NL]] = q12
        mujoco.mj_kinematics(self.m, d)
        out = []
        for b in self.r.foot_bid:
            R = d.xmat[b].reshape(3, 3).copy()
            out.append((d.xpos[b] + R @ SOLE_PT, R))
        return out

    @staticmethod
    def _res(fl, fr, g, nuis):
        (pL, RL), (pR, RR) = fl, fr
        x, y, psi = nuis
        rg = np.concatenate([(RL.T @ g)[:2], (RR.T @ g)[:2]])
        rr = _vee_skew(_rz(psi).T @ (RL.T @ RR))
        rp = RL.T @ (pR - pL) - np.array([x, y, 0.0])
        return np.concatenate([rg, rr, rp])

    def residuals(self, theta, Q, Gb, jac=True, h=1e-6):
        """r: (N, NR) unweighted; J: (N, NR, NP) d r / d theta."""
        delta, nuis = theta[:NL], theta[NL:]
        N = len(Q)
        r = np.zeros((N, NR)); J = np.zeros((N, NR, NP)) if jac else None
        for k in range(N):
            qt = Q[k] - delta
            fl, fr = self._feet(qt)
            r[k] = self._res(fl, fr, Gb[k], nuis)
            if not jac:
                continue
            for j in range(6):           # perturb left j and right j together (legs decoupled)
                dq = np.zeros(NL); dq[j] = h; dq[6 + j] = h
                fl2, fr2 = self._feet(qt - dq)      # d(q_true)/d(delta) = -1
                rl = self._res(fl2, fr, Gb[k], nuis)
                rrr = self._res(fl, fr2, Gb[k], nuis)
                J[k, :, j] = (rl - r[k]) / h
                J[k, :, 6 + j] = (rrr - r[k]) / h
            for j in range(3):
                n2 = np.array(nuis, float); n2[j] += h
                J[k, :, NL + j] = (self._res(fl, fr, Gb[k], n2) - r[k]) / h
        return r, J

    # ------------------------------------------------------------------ solve
    def _augment(self, r, J, theta):
        """residual model r_k = r_kin(delta, nuis) - b ;  b = constant systematic bias per channel."""
        b = theta[NP:]
        Ja = np.concatenate([J, np.broadcast_to(-np.eye(NR), (len(r), NR, NR))], axis=2)
        return r - b[None, :], Ja

    def _prior(self):
        P = np.zeros(NA); P[:NL] = 1.0 / self.prior_std ** 2; P[NL:NP] = 1e-6; P[NP:] = 1.0 / self.sig_sys ** 2
        return P

    def _gn(self, Q, Gb, theta, free, iters=8, keep=None):
        """Gauss-Newton MAP on the 'free' subset of theta = [delta, nuis, bias]."""
        W = 1.0 / self.sig
        P = self._prior()
        fixed_keep = keep is not None
        keep = np.ones(len(Q), bool) if keep is None else keep
        for it in range(iters):
            r, J = self._augment(*self.residuals(theta[:NP], Q, Gb), theta)
            if it == 2 and not fixed_keep:                # robust: drop gross outliers once
                z = np.abs(r * W).max(1)
                keep = z < max(6.0, np.percentile(z, 99))
            rw = (r * W)[keep].reshape(-1); Jw = (J * W[None, :, None])[keep].reshape(-1, NA)
            Jf = Jw[:, free]
            H = Jf.T @ Jf + np.diag(P[free])
            g = Jf.T @ rw + P[free] * theta[free]
            step = np.linalg.solve(H, g)
            theta = theta.copy(); theta[free] -= step
            if np.abs(step).max() < 1e-8:
                break
        r, J = self._augment(*self.residuals(theta[:NP], Q, Gb), theta)
        return theta, r, J, keep

    def nominal_stance(self):
        fl, fr = self._feet(self.r.default_q()[:NL])
        (pL, RL), (pR, RR) = fl, fr
        rel = RL.T @ (pR - pL)
        return np.array([rel[0], rel[1], np.arctan2((RL.T @ RR)[1, 0], (RL.T @ RR)[0, 0])])

    @staticmethod
    def tau_int(r, max_lag=200, cut=0.05):
        """Integrated autocorrelation time (in samples) of each residual channel after the fit.
        Information from N correlated samples ~ information from N / tau_int independent ones."""
        out = np.ones(r.shape[1])
        for c in range(r.shape[1]):
            x = r[:, c] - r[:, c].mean()
            v = (x * x).mean()
            if v <= 0:
                continue
            t = 1.0
            for L in range(1, min(max_lag, len(x) // 3)):
                rho = (x[L:] * x[:-L]).mean() / v
                if rho < cut:
                    break
                t += 2 * rho
            out[c] = t
        return out

    def information(self, theta, Q, Gb, keep=None, known_stance=False, scale=1.0, tau=None):
        """Marginal data information on delta: nuisance (x, y, psi) and systematic bias b are
        Schur-complemented out (b with its sigma_sys prior), misfit-scaled and divided per
        channel by the residual integrated autocorrelation time tau (correlated samples)."""
        r, J = self._augment(*self.residuals(theta[:NP], Q, Gb), theta)
        keep = np.ones(len(Q), bool) if keep is None else keep
        tau = np.ones(NR) if tau is None else tau
        Jw = (J / (self.sig * np.sqrt(tau))[None, :, None])[keep].reshape(-1, NA)
        H = Jw.T @ Jw / scale
        P = self._prior(); P[:NL] = 0.0                # data-only on delta, priors on nuisances
        H = H + np.diag(P)
        oth = np.arange(NP, NA) if known_stance else np.arange(NL, NA)
        Hdd, Hdn, Hnn = H[:NL, :NL], H[:NL][:, oth], H[np.ix_(oth, oth)]
        return Hdd - Hdn @ np.linalg.solve(Hnn + 1e-9 * np.eye(len(oth)), Hdn.T)

    def loop_consistency(self, Q, Gb, tol_ang=np.deg2rad(1.0), tol_z=0.005):
        """Sample gate #2 (needs no offset knowledge): with constant offsets the loop-closure
        residuals at delta=0 vary only weakly with posture; a sole that rolls onto its edge or
        slips shows up as a jump of the IMU-free relative foot tilt / height or of the gravity
        residual away from its robust median -> reject (non-flat / transient contact)."""
        th = np.zeros(NP); th[NL:] = self.nominal_stance()
        r, _ = self.residuals(th, Q, Gb, jac=False)
        ch = np.r_[0:4, 4, 5]                           # gravity(4), rel roll, rel pitch
        dev = np.abs(r[:, ch] - np.median(r[:, ch], 0)).max(1)
        devz = np.abs(r[:, 9] - np.median(r[:, 9]))
        return (dev < tol_ang) & (devz < tol_z)

    def solve(self, known_stance=False, subsample=None):
        Q = np.array(self.Q); Gb = np.array(self.Gb)
        if subsample and len(Q) > subsample:
            idx = np.linspace(0, len(Q) - 1, subsample).astype(int)
            Q, Gb = Q[idx], Gb[idx]
        out = dict(n=int(len(Q)), n_seen=int(self.n_seen), n_gate_rej=int(self.n_gate_rej), committed=False)
        delta0 = np.zeros(NL)
        sel = np.arange(len(Q))
        if len(Q) >= 10:
            ok_loop = self.loop_consistency(Q, Gb)
            self.loop_mask = ok_loop
            out["n_loop_rej"] = int((~ok_loop).sum())
            Q, Gb = Q[ok_loop], Gb[ok_loop]
            sel = sel[ok_loop]
        n = len(Q)
        if n < self.n_min:
            out["reason"] = f"too few gated samples ({n} < {self.n_min})"
            return delta0, out
        if self.torque:
            assert not subsample, "torque channel: no subsampling"
            return self._solve_torque(Q, Gb, sel, known_stance, out)
        return self._solve_lc(Q, Gb, known_stance, out)

    def _solve_lc(self, Q, Gb, known_stance, out):
        """loop-closure-only solve on gated samples (the published Phase-2b estimator)."""
        delta0 = np.zeros(NL)
        W = 1.0 / self.sig
        theta = np.zeros(NA)
        theta[NL:NP] = self.nominal_stance()
        free_n = np.r_[NP:NA] if known_stance else np.r_[NL:NA]          # everything but delta
        free_all = np.r_[0:NL, NP:NA] if known_stance else np.r_[0:NA]
        # H1: MAP delta (with outlier rejection)
        th1, r1, J1, keep = self._gn(Q, Gb, theta, free_all)
        # H0: delta = 0, nuisances fitted on the same samples
        th0, r0, _, _ = self._gn(Q, Gb, theta, free_n, iters=4, keep=keep)
        chi0 = float(((r0 * W)[keep] ** 2).sum()); chi1 = float(((r1 * W)[keep] ** 2).sum())
        dof = max(int(keep.sum()) * NR - NA, 1)
        s2 = max(1.0, chi1 / dof)                      # misfit scale (unmodelled systematics)
        tau = self.tau_int(r1[keep]) if self.use_tau else np.ones(NR)   # correlated-residual inflation
        Hm = self.information(th1, Q, Gb, keep, known_stance, scale=s2, tau=tau)
        lam, V = np.linalg.eigh(Hm)
        prior_info = 1.0 / self.prior_std ** 2
        obs = lam >= self.kappa * prior_info
        Vo = V[:, obs]
        d_map = th1[:NL]
        d_commit = Vo @ (Vo.T @ d_map)
        # posterior (data + prior) per-joint std
        Cpost = np.linalg.inv(Hm + prior_info * np.eye(NL))
        # evidence: chi2 at the committed (projected) delta, nuisances re-fitted
        thc = th1.copy(); thc[:NL] = d_commit
        thc, rc, _, _ = self._gn(Q, Gb, thc, free_n, iters=3, keep=keep)
        chic = float(((rc * W)[keep] ** 2).sum())
        evid = (chi0 - chic) / s2 / float(np.mean(tau))
        ok = obs.sum() > 0 and evid > 2.0 * obs.sum()
        rms = lambda r: dict(grav_deg=float(np.rad2deg(np.sqrt((r[keep, :4] ** 2).mean()))),
                             relrot_deg=float(np.rad2deg(np.sqrt((r[keep, 4:7] ** 2).mean()))),
                             relpos_mm=float(1e3 * np.sqrt((r[keep, 7:] ** 2).mean())))
        out.update(
            committed=bool(ok), n_used=int(keep.sum()), n_obs_dirs=int(obs.sum()),
            eig_std_deg=np.rad2deg(1.0 / np.sqrt(np.maximum(lam, 1e-12))).round(4).tolist(),
            eigvecs=V.round(3).tolist(), obs_mask=obs.tolist(),
            delta_map_deg=np.rad2deg(d_map).round(3).tolist(),
            delta_commit_deg=np.rad2deg(d_commit if ok else delta0).round(3).tolist(),
            post_std_deg=np.rad2deg(np.sqrt(np.diag(Cpost))).round(3).tolist(),
            nuisance=th1[NL:NP].round(5).tolist(), bias=th1[NP:].round(5).tolist(), misfit_scale=float(s2),
            tau_int=tau.round(2).tolist(),
            rms_before=rms(r0), rms_after=rms(rc), evidence_chi2=float(evid))
        return (d_commit if ok else delta0), out

    # ================================================================== torque channel
    # theta_t = [delta(12), nuis(3), bias(10), trunk(4: dm dx dy dz), mu(1: leg frictionloss)]
    def _torque_samples(self, sel):
        S = {k: np.array(self.S[k])[sel] for k in ("q", "qd", "quat", "tau")}
        S["qdd"] = sg_derivative_at(np.array(self._tick_qd), np.array(self.S["tick"])[sel], self.dt)
        return S

    def torque_consistency(self, S, win=15):
        """Sample gate #3 (torque-channel consistency, needs no offset knowledge): at delta = 0,
        nominal trunk and prior friction, the 6 floating-base torque residuals of a quasi-static,
        flat-footed double stance vary SMOOTHLY with posture (wrong trunk mass / CoM and wrong
        offsets enter through the slowly changing leg geometry).  Each residual is detrended with
        a running median over +-win gated samples; a sample whose detrended residual exceeds
        tq_gate x the robust (MAD) scale in any channel (contact transient, a sole losing its edge,
        an unmodelled push, an SG-acceleration spike) violates the rigid-body + 2-wrench model
        -> reject.  (A noise-model-normalised test against the global median was tried first:
        it rejects ~50 % of normal samples, because the offset-induced posture dependence itself
        is many noise sigmas -> not usable before the offsets are known.)"""
        R, _, _ = self.tc.residuals(np.zeros(NL), np.zeros(NTR), [self.mu_prior[0]], S, jac=False)
        n = len(R)
        trend = np.array([np.median(R[max(0, k - win):k + win + 1], 0) for k in range(n)])
        e = R - trend
        sc = 1.4826 * np.median(np.abs(e - np.median(e, 0)), 0) + 1e-9
        dev = (np.abs(e) / sc[None, :]).max(1)
        return dev < self.tq_gate

    def _prior_t(self):
        P = np.r_[self._prior(), 1.0 / self.trunk_prior_std ** 2, 1.0 / self.mu_prior[1] ** 2]
        if self.co_v0:
            P = np.r_[P, 1.0 / self.v0_prior[1] ** 2]
        m = np.zeros(len(P)); m[self.iMU] = self.mu_prior[0]
        if self.co_v0:
            m[-1] = self.v0_prior[0]
        return P, m

    def _stack_t(self, theta, Q, Gb, S):
        """whitened residuals + Jacobians (w.r.t. theta_t) of both channels."""
        nat = NA + NTR + self.nf
        r, J = self._augment(*self.residuals(theta[:NP], Q, Gb), theta[:NA])
        W = 1.0 / self.sig
        rl = r * W
        Jl = np.zeros((len(Q), NR, nat)); Jl[:, :, :NA] = J * W[None, :, None]
        R, Jt, C = self.tc.residuals(theta[:NL], theta[NA:NA + NTR], [theta[self.iMU]], S,
                                     v0=theta[-1] if self.co_v0 else None)
        rt, Jtw = whiten(R, Jt, C)
        Jtf = np.zeros((len(Q), NRT, nat)); Jtf[:, :, :NL] = Jtw[:, :, :NL]; Jtf[:, :, NA:] = Jtw[:, :, NL:]
        return rl, Jl, rt, Jtf

    def _gn_t(self, Q, Gb, S, theta, free, iters=8, keep=None):
        P, m = self._prior_t()
        fixed = keep is not None
        N = len(Q)
        kl, kt = (np.ones(N, bool), np.ones(N, bool)) if keep is None else keep
        for it in range(iters):
            rl, Jl, rt, Jt = self._stack_t(theta, Q, Gb, S)
            if it == 2 and not fixed:                       # robust: drop gross outliers once per channel
                zl = np.abs(rl).max(1); kl = zl < max(6.0, np.percentile(zl, 99))
                zt = np.abs(rt).max(1); kt = zt < max(6.0, np.percentile(zt, 99))
            rw = np.r_[rl[kl].ravel(), rt[kt].ravel()]
            Jf = np.vstack([Jl[kl].reshape(-1, len(theta)), Jt[kt].reshape(-1, len(theta))])[:, free]
            H = Jf.T @ Jf + np.diag(P[free])
            g = Jf.T @ rw + P[free] * (theta[free] - m[free])
            step = np.linalg.solve(H, g)
            theta = theta.copy(); theta[free] -= step
            theta[self.iMU] = max(theta[self.iMU], 0.0)     # friction >= 0
            if self.co_v0:
                theta[-1] = max(theta[-1], 0.005)           # transition velocity > 0
            if np.abs(step).max() < 1e-8:
                break
        return theta, self._stack_t(theta, Q, Gb, S), (kl, kt)

    def _solve_torque(self, Q, Gb, sel, known_stance, out):
        """loop closure + joint-torque channel, same triple gating + a 4th (channel-agreement) check."""
        delta0 = np.zeros(NL)
        nat = NA + NTR + self.nf
        # LC-only reference solve (published estimator) on the same loop-gated samples
        _, out_lc = self._solve_lc(Q, Gb, known_stance, dict(out))
        S = self._torque_samples(sel)
        ok_t = self.torque_consistency(S)
        out["n_torque_rej"] = int((~ok_t).sum())
        full = np.zeros(len(self.Q), bool); full[sel[ok_t]] = True
        self.loop_mask = full
        Q, Gb = Q[ok_t], Gb[ok_t]
        S = {k: v[ok_t] for k, v in S.items()}
        n = len(Q)
        out["lc_only"] = {k: out_lc.get(k) for k in ("committed", "n_obs_dirs", "delta_commit_deg", "eig_std_deg", "reason")}
        if n < self.n_min:
            out["reason"] = f"too few gated samples after torque gate ({n} < {self.n_min})"
            return delta0, out
        theta = np.zeros(nat)
        theta[NL:NP] = self.nominal_stance(); theta[self.iMU] = self.mu_prior[0]
        if self.co_v0:
            theta[-1] = self.v0_prior[0]
        free_n = np.r_[NP:nat] if known_stance else np.r_[NL:nat]
        free_all = np.r_[0:NL, NP:nat] if known_stance else np.r_[0:nat]
        th1, (rl1, Jl1, rt1, Jt1), keep = self._gn_t(Q, Gb, S, theta, free_all)
        kl, kt = keep
        th0, (rl0, _, rt0, _), _ = self._gn_t(Q, Gb, S, theta, free_n, iters=4, keep=keep)
        chi = lambda a, b: (float((a[kl] ** 2).sum()), float((b[kt] ** 2).sum()))
        c1, c0 = chi(rl1, rt1), chi(rl0, rt0)
        s2l = max(1.0, c1[0] / max(int(kl.sum()) * NR - NA, 1))
        s2t = max(1.0, c1[1] / max(int(kt.sum()) * NRT - (NL + NTR + self.nf), 1))
        taul = self.tau_int(rl1[kl]) if self.use_tau else np.ones(NR)
        taut = self.tau_int(rt1[kt]) if self.use_tau else np.ones(NRT)
        Jw = np.vstack([(Jl1[kl] / np.sqrt(taul * s2l)[None, :, None]).reshape(-1, nat),
                        (Jt1[kt] / np.sqrt(taut * s2t)[None, :, None]).reshape(-1, nat)])
        P, _ = self._prior_t(); P[:NL] = 0.0
        H = Jw.T @ Jw + np.diag(P)
        oth = np.r_[NP:nat] if known_stance else np.r_[NL:nat]
        Hdn = H[:NL][:, oth]
        Hm = H[:NL, :NL] - Hdn @ np.linalg.solve(H[np.ix_(oth, oth)] + 1e-9 * np.eye(len(oth)), Hdn.T)
        lam, V = np.linalg.eigh(Hm)
        prior_info = 1.0 / self.prior_std ** 2
        obs = lam >= self.kappa * prior_info
        Vo = V[:, obs]
        d_map = th1[:NL]
        d_commit = Vo @ (Vo.T @ d_map)
        Cpost = np.linalg.inv(Hm + prior_info * np.eye(NL))
        thc = th1.copy(); thc[:NL] = d_commit
        thc, (rlc, _, rtc, _), _ = self._gn_t(Q, Gb, S, thc, free_n, iters=3, keep=keep)
        cc = chi(rlc, rtc)
        evid = (c0[0] - cc[0]) / s2l / float(np.mean(taul)) + (c0[1] - cc[1]) / s2t / float(np.mean(taut))
        ok = obs.sum() > 0 and evid > 2.0 * obs.sum()
        # gate #4 (channel agreement): on every direction the loop closure ALONE commits, the
        # combined estimate must agree with the LC-only estimate within 3 sigma (LC-only formal
        # std).  A torque-model systematic (torque-constant error, unmodelled payload / link
        # inertia, wrong friction law) that drags the well-observed subspace -> torque channel
        # vetoed, LC-only result kept.  Guarantees no regression on the LC-observable subspace.
        agree, zmax = True, 0.0
        if out_lc.get("committed"):
            Vl = np.array(out_lc["eigvecs"]); ol = np.array(out_lc["obs_mask"], bool)
            sl = np.deg2rad(np.array(out_lc["eig_std_deg"]))[ol]
            dl = np.deg2rad(np.array(out_lc["delta_commit_deg"]))
            z = np.abs(Vl[:, ol].T @ (d_commit - dl)) / sl
            zmax = float(z.max()) if len(z) else 0.0
            agree = zmax < 3.0
        rms = lambda a, b: dict(grav_deg=float(np.rad2deg(np.sqrt(((a[kl, :4] * self.sig[:4]) ** 2).mean()))),
                                relrot_deg=float(np.rad2deg(np.sqrt(((a[kl, 4:7] * self.sig[4:7]) ** 2).mean()))),
                                relpos_mm=float(1e3 * np.sqrt(((a[kl, 7:] * self.sig[7:]) ** 2).mean())),
                                torque_white_rms=float(np.sqrt((b[kt] ** 2).mean())))
        use = ok and agree
        out.update(
            committed=bool(use), torque_evidence_ok=bool(ok), channel_agree=bool(agree), agree_zmax=zmax,
            n_used=int(kl.sum()), n_used_torque=int(kt.sum()), n_obs_dirs=int(obs.sum()),
            eig_std_deg=np.rad2deg(1.0 / np.sqrt(np.maximum(lam, 1e-12))).round(4).tolist(),
            eigvecs=V.round(3).tolist(), obs_mask=obs.tolist(),
            delta_map_deg=np.rad2deg(d_map).round(3).tolist(),
            delta_commit_deg=np.rad2deg(d_commit if use else delta0).round(3).tolist(),
            post_std_deg=np.rad2deg(np.sqrt(np.diag(Cpost))).round(3).tolist(),
            nuisance=th1[NL:NP].round(5).tolist(), bias=th1[NP:NA].round(5).tolist(),
            trunk_hat=th1[NA:NA + NTR].round(4).tolist(), mu_hat=float(th1[self.iMU]),
            v0_hat=float(th1[-1]) if self.co_v0 else self.tc.v0,
            trunk_std=np.sqrt(np.diag(np.linalg.inv(H))[NA:NA + NTR]).round(4).tolist(),
            mu_std=float(np.sqrt(np.linalg.inv(H)[self.iMU, self.iMU])),
            misfit_scale=float(s2l), misfit_scale_torque=float(s2t),
            tau_int=taul.round(2).tolist(), tau_int_torque=taut.round(2).tolist(),
            rms_before=rms(rl0, rt0), rms_after=rms(rlc, rtc), evidence_chi2=float(evid))
        if ok and not agree:
            # veto: fall back to the LC-only commit (gates of the published estimator)
            out["delta_commit_deg"] = out_lc.get("delta_commit_deg", [0.0] * NL)
            out["committed"] = bool(out_lc.get("committed"))
            out["fallback"] = "lc_only (channel disagreement)"
            out["torque_eig_std_deg"] = out["eig_std_deg"]
            for k in ("eig_std_deg", "eigvecs", "obs_mask", "n_obs_dirs"):
                out[k] = out_lc.get(k)
            return np.deg2rad(np.array(out["delta_commit_deg"])), out
        return (d_commit if use else delta0), out


def label_dir(v, thr=0.25):
    """Human-readable description of an eigen-direction over the 12 leg offsets."""
    names = [j.replace("_joint", "") for j in LEG_JOINTS]
    idx = np.argsort(-np.abs(v))
    return " ".join(f"{v[i]:+.2f}{names[i]}" for i in idx if abs(v[i]) >= thr)
