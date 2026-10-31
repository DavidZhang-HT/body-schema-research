"""Observability scan of joint zero offsets WITH the joint-torque residual channel (Phase 2b arbiter).

Input: the full-rate Phase-2 calibration recordings (zero_torque_record.py ->
results/q0/samples_tau_<mode>_<seed>.npz), evaluated on exactly the gated samples the deployed
loop-closure estimator used.

Unknowns: delta (12 leg offsets) + trunk (dm, dx, dy, dz) + friction mu (1 common or 12 per joint)
+ loop-closure nuisances (x, y, psi).  Trunk / mu / nuisances carry their Phase-1 priors
(scenarios.PRIOR_STD) and are Schur-complemented out; delta is data-only (same convention as the
deployed ZeroEstimator.information -> eig_std_deg).

Two noise conventions, reported side by side for BOTH channels:
  white    : per-sample noise model only (loop closure: sigma_g/ra/p; torque: s_tau, rel_tau and
             the stick-friction term of TorqueChannel), samples treated as independent
             -> optimistic upper bound on information.
  deployed : as ZeroEstimator.solve: misfit scale s2 = max(1, chi2/dof) and per-channel
             integrated autocorrelation time tau_int of the whitened residuals, both measured at
             the TRUE parameters (delta, trunk, mu all at truth -> pure model / sensor error).
             For loop closure the deployed s2 / tau_int of the published calib run are used.

Diagnostics:
  * residual budget of the torque channel at the truth: measured-state vs true-state input,
    modelled friction vs MuJoCo's actual frictionloss torque (truth snapshot);
  * trunk-CoM vs offset separation: correlation of each weak direction with trunk_dx/dy/dz/dm
    in the torque information, and its configuration dependence along the trajectory.

Usage: .venv/bin/python zero_torque_scan.py [seed ...]   -> results/q0/torque_scan.json
"""
import os, sys, json
import numpy as np
from scenarios import true_params_q0, PRIOR_STD
from zero_id import ZeroEstimator, NL, NP, NR, label_dir
from zero_torque import TorqueChannel, prepare, NTR, NRT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "q0")
TR_PRIOR = np.array([PRIOR_STD["trunk_dm"], PRIOR_STD["trunk_dx"], PRIOR_STD["trunk_dy"], PRIOR_STD["trunk_dz"]])
MU_PRIOR = PRIOR_STD["frictionloss"]


def whiten(R, J, C):
    """per-sample Cholesky whitening: returns Rw (N,6), Jw (N,6,p)."""
    L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[1])[None])
    Rw = np.linalg.solve(L, R[..., None])[..., 0]
    Jw = np.linalg.solve(L, J)
    return Rw, Jw


def marginal(H, nd=NL):
    Hdd, Hdn, Hnn = H[:nd, :nd], H[:nd, nd:], H[nd:, nd:]
    return Hdd - Hdn @ np.linalg.solve(Hnn, Hdn.T)


def stds(Hm, W):
    """formal std (deg) of eigen-directions of Hm (ascending) and along the fixed directions W."""
    lam, V = np.linalg.eigh(Hm)
    lam = np.maximum(lam, 1e-30)
    eig = np.rad2deg(1 / np.sqrt(lam))
    # along a fixed direction w: marginal std with all other delta directions also unknown
    # (data only -> pseudo-inverse restricted to the well-defined part; report inf if ~null)
    Ci = np.linalg.pinv(Hm, rcond=1e-14)
    along = [float(np.rad2deg(np.sqrt(max(w @ Ci @ w, 0)))) for w in W.T]
    return eig, V, along


def scan(seed, mode="active", mu_per_joint=False):
    tp = true_params_q0(seed)
    d_true = np.array(tp.q0[:NL])
    tr_true = np.array([tp.trunk_dm, tp.trunk_dx, tp.trunk_dy, tp.trunk_dz])
    rec = np.load(os.path.join(OUT, f"samples_tau_{mode}_{seed}.npz"))
    pub = json.load(open(os.path.join(OUT, f"calib_{mode}_{seed}.json")))["free"]
    idx = np.flatnonzero(rec["gated"])
    est = ZeroEstimator(use_tau=False)
    Qm = rec["q"][idx, :NL]
    Gb = np.array([_up(qq) for qq in rec["quat"][idx]])
    if len(idx) >= 10:
        lm = est.loop_consistency(Qm, Gb)
        idx, Qm, Gb = idx[lm], Qm[lm], Gb[lm]
    N = len(idx)
    out = dict(seed=seed, mode=mode, n=N, fell=bool(rec["fell"]))
    if N < 20:
        out["reason"] = "too few gated samples"
        return out

    # ---------------- loop closure (at truth, measured q incl. backlash / noise)
    th = np.zeros(NP); th[:NL] = d_true; th[NL:] = est.nominal_stance()
    th = est._gn(Qm, Gb, np.r_[th, np.zeros(NR)], np.r_[NL:NP], iters=4)[0][:NP]   # fit nuisances only
    _, Jlc = est.residuals(th, Qm, Gb)
    s2_lc = float(pub.get("misfit_scale", 1.0)); tau_lc = np.array(pub.get("tau_int", [1.0] * NR))

    def H_lc(deployed):
        sc = est.sig * (np.sqrt(tau_lc * s2_lc) if deployed else 1.0)
        Jw = (Jlc / sc[None, :, None]).reshape(-1, NP)
        return Jw.T @ Jw

    # ---------------- torque channel (at truth)
    tc = TorqueChannel()
    nmu = NL if mu_per_joint else 1
    mu0 = np.full(nmu, tp.frictionloss)
    S = prepare(rec, idx)
    R, Jt, C = tc.residuals(d_true, tr_true, mu0, S)
    Rw, Jtw = whiten(R, Jt, C)
    s2_t = max(1.0, float((Rw ** 2).sum() / max(Rw.size - (NL + NTR + nmu), 1)))
    tau_t = ZeroEstimator.tau_int(Rw)
    # residual budget diagnostics (no Jacobians)
    St = prepare(rec, idx, use_true=True)
    R_true_state, _, C2 = tc.residuals(np.zeros(NL), tr_true, mu0, St, jac=False)
    fr_true = rec["truth"][idx, 80:92]     # MuJoCo frictionloss torque on the leg dofs (efc force)
    tau_true = rec["truth"][idx, 60:72]    # applied (delayed) motor torque, noise-free
    tc_nf = TorqueChannel()
    R_true_fric = np.zeros_like(R)
    for k in range(N):
        tc_nf.set_trunk(tr_true)
        R_true_fric[k] = tc_nf.eval(St["q"][k, :NL], St["q"][k, NL:], St["quat"][k], St["qd"][k], St["qdd"][k],
                                    tau_true[k], fr_true[k])[0]
    tc_nf.set_trunk(np.zeros(NTR))
    rms = lambda X: np.sqrt((X ** 2).mean(0)).round(3).tolist()
    out["residual_rms_truth"] = dict(measured_state=rms(R), true_state=rms(R_true_state),
                                     true_state_true_tau_true_friction=rms(R_true_fric),
                                     noise_model_std=np.sqrt(np.einsum("kii->ki", C).mean(0)).round(3).tolist())
    out["torque_misfit_scale"] = s2_t
    out["torque_tau_int"] = tau_t.round(2).tolist()

    def H_t(deployed):
        sc = np.sqrt(tau_t * s2_t) if deployed else np.ones(NRT)
        Jw = (Jtw / sc[None, :, None]).reshape(-1, NL + NTR + nmu)
        return Jw.T @ Jw

    # ---------------- combine: theta = [delta(12), trunk(4), mu(nmu), nuis(3)]
    p = NL + NTR + nmu + 3
    iN = slice(NL + NTR + nmu, p)
    pri = np.r_[np.zeros(NL), 1 / TR_PRIOR ** 2, np.full(nmu, 1 / MU_PRIOR ** 2), np.full(3, 1e-6)]

    def assemble(use_lc, use_t, deployed, trunk_prior=True):
        H = np.zeros((p, p))
        if use_lc:
            Hl = H_lc(deployed)
            sel = np.r_[0:NL, NL + NTR + nmu:p]
            H[np.ix_(sel, sel)] += Hl
        if use_t:
            H[:NL + NTR + nmu, :NL + NTR + nmu] += H_t(deployed)
        P = pri.copy()
        if not trunk_prior:
            P[NL:NL + NTR] = 1e-8
        return H + np.diag(P)

    # weak directions W1..W4 = 4 weakest eigenvectors of the deployed loop-closure-only information
    Hm_lc = marginal(assemble(True, False, True))
    lam, V = np.linalg.eigh(Hm_lc)
    W = V[:, :4]
    out["W_dirs"] = [label_dir(W[:, i]) for i in range(4)]
    res = {}
    for name, (lc, t, dep, tpri) in {
            "lc_white": (True, False, False, True), "lc_deployed": (True, False, True, True),
            "lc+tau_white": (True, True, False, True), "lc+tau_deployed": (True, True, True, True),
            "lc+tau_white_flat_trunk": (True, True, False, False),
            "tau_only_white": (False, True, False, True)}.items():
        H = assemble(lc, t, dep, tpri)
        Hm = marginal(H)
        eig, Ve, along = stds(Hm, W)
        res[name] = dict(W_std_deg=np.round(along, 3).tolist(),
                         weakest4_eig_std_deg=np.round(eig[:4], 3).tolist(),
                         weakest_dir=label_dir(Ve[:, 0]),
                         n_dirs_below_0p5=int((eig < 0.5).sum()), n_dirs_below_2=int((eig < 2.0).sum()))
    out["scan"] = res

    # ---------------- trunk-CoM vs offset separation (torque channel, white)
    Ht = H_t(False)
    Cfull = np.linalg.pinv(Ht[:NL + NTR, :NL + NTR] + np.diag(np.r_[np.zeros(NL), 1e-8 * np.ones(NTR)]), rcond=1e-14)
    corr = []
    for i in range(4):
        w = np.r_[W[:, i], np.zeros(NTR)]
        vw = w @ Cfull @ w
        corr.append([float((w @ Cfull[:, NL + j]) / np.sqrt(max(vw * Cfull[NL + j, NL + j], 1e-300))) for j in range(NTR)])
    out["torque_corr_W_vs_trunk"] = np.round(corr, 3).tolist()
    # configuration dependence: per-sample residual signature of 1 deg along W_i and of the trunk
    # parameter (1 cm / 0.1 kg) -> pitch-moment row (index 4: base moment y) along the trajectory
    sig = {}
    for i in range(4):
        s_w = Jt[:, :, :NL] @ W[:, i] * np.deg2rad(1.0)               # (N,6) per 1 deg along W_i
        sig[f"W{i+1}_per_deg"] = dict(mean=np.abs(s_w).mean(0).round(4).tolist(),
                                      std_over_traj=s_w.std(0).round(4).tolist())
    for j, (nm, u) in enumerate((("dm_per_0.1kg", 0.1), ("dx_per_cm", 0.01), ("dy_per_cm", 0.01), ("dz_per_cm", 0.01))):
        s_t = Jt[:, :, NL + j] * u
        sig[nm] = dict(mean=np.abs(s_t).mean(0).round(4).tolist(), std_over_traj=s_t.std(0).round(4).tolist())
    out["signature"] = sig
    # range of leg configuration visited (knee bend, hip pitch) -> how far from "straight"
    qt = rec["truth"][idx, :NL]
    out["knee_deg_range"] = [float(np.rad2deg(qt[:, 3].min())), float(np.rad2deg(qt[:, 3].max()))]
    out["hip_pitch_deg_range"] = [float(np.rad2deg(qt[:, 0].min())), float(np.rad2deg(qt[:, 0].max()))]
    return out


def _up(quat):
    from wbc import quat_to_mat
    return quat_to_mat(quat).T @ np.array([0, 0, 1.0])


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(5))
    allout = []
    for seed in seeds:
        for mode in ("active", "quiet"):
            for mpj in (False, True):
                a = scan(seed, mode, mu_per_joint=mpj)
                a["mu_per_joint"] = mpj
                allout.append(a)
                if "scan" not in a:
                    print(seed, mode, "skip:", a.get("reason"), "n", a["n"], flush=True); break
                print(f"\nseed {seed} {mode} mu_per_joint={mpj} n={a['n']} knee {a['knee_deg_range']}", flush=True)
                if not mpj:
                    print("  W dirs:", *a["W_dirs"], sep="\n    ")
                    print("  torque residual RMS at truth (Fx Fy Fz Mx My Mz):")
                    for k, v in a["residual_rms_truth"].items():
                        print(f"    {k:26s} {v}")
                    print("  torque misfit", round(a["torque_misfit_scale"], 2), "tau_int", a["torque_tau_int"])
                    print("  corr(W_i, trunk dm dx dy dz):", a["torque_corr_W_vs_trunk"])
                for k, v in a["scan"].items():
                    print(f"  {k:24s} W1-4 std {v['W_std_deg']}  weakest eig {v['weakest4_eig_std_deg']} "
                          f"#<0.5 {v['n_dirs_below_0p5']} #<2 {v['n_dirs_below_2']}", flush=True)
    json.dump(allout, open(os.path.join(OUT, "torque_scan.json"), "w"), indent=1)
