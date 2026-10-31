"""Noise-free observability scan of SAFE double-support exploratory motions for joint-zero ID.

Question: does any double-support motion (both soles flat, no stepping, bounded base height /
tilt, torques within limits) lift the common-mode intra-leg allocation directions
(calf vs ankle pitch, hip pitch vs calf+ankle) above the 2 deg prior?

Method (pure kinematics, deterministic, no simulation):
  * a motion = base pose trajectory (x, y, z, roll, pitch, yaw)(t) relative to two soles that are
    fixed flat on the floor at a chosen stance (width, stagger, yaw);
  * leg IK (6-DoF per leg, Newton on the controller's MuJoCo model) gives the TRUE q(t);
    gravity in the base frame is exact;
  * ZeroEstimator.residuals() Jacobian at the truth -> marginal information on the 12 offsets
    (nuisance x, y, psi Schur-complemented; the same sigmas as zero_observability.py),
    sampled at 100 Hz for T = 6 s (= the deployed calibration window and decimation);
  * safety / feasibility per sample: joint limits (2 deg margin), quasi-static torques
    (min-norm 2-foot wrench split) < 20 Nm, CoP of each foot inside its sole (5 mm margin),
    CoM inside the support polygon (>= 2 cm), base tilt <= 6 deg.

Metric per candidate:
  * eigen-spectrum (rel. sv, std) of the 6 s noiseless information;
  * marginal formal std (noiseless, 6 s, white) along the BASELINE weak directions, i.e. the 2
    weakest eigenvectors of the replica of the current Phase-2 sway (sqrt(v^T H^-1 v));
  * 'deployed-equivalent' std = noiseless std x K, K = (deployed estimator weak std) /
    (noiseless 6 s weak std) for the baseline, from the cached seed-0 calibration (accounts
    for tau_int inflation + misfit scaling).  Verdict thresholds: 2 deg (prior) and 0.5 deg
    (commit gate kappa=16 -> std <= prior/4).

Usage:  .venv/bin/python zero_motion_scan.py   -> results/q0/motion_scan.json (+ table on stdout)
"""
import os, json
import numpy as np
import mujoco

from model import PiPlus, LEG_JOINTS
from wbc import SOLE_PT
from zero_id import ZeroEstimator, NL, NP, label_dir
from model import FOOT_CORNERS, SOLE_CENTER, SOLE_HALF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "q0")
FS, T = 100.0, 6.0
TAU_LIM = 20.0
EPS = 1.0 / np.deg2rad(100.0) ** 2          # caps reported std at ~100 deg (exact nullspace)


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class Kin:
    def __init__(self):
        self.r = PiPlus()
        self.m, self.d = self.r.m, self.r.d
        self.q_nom = self.r.default_q()
        self.est = ZeroEstimator(use_tau=False)
        fl, fr = self.est._feet(self.q_nom[:NL])
        self.z0 = -fl[0][2]                       # base height above soles (nominal)
        self.x0 = fl[0][0]                        # sole-centre x in base frame (nominal)
        self.w0 = fl[0][1]                        # half stance width (nominal)
        self.mass = float(self.m.body_subtreemass[0])

    # --- base frame at origin, identity
    def _fk(self, q20):
        d = self.d
        d.qpos[:] = 0; d.qpos[3] = 1.0
        d.qpos[self.r.qadr] = q20
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)

    def ik(self, q20, targets, iters=200):
        """targets: [(p_sole, R_foot)] for l, r in BASE frame."""
        q = q20.copy()
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self._fk(q)
            err_max = 0.0
            for i, (pt, Rt) in enumerate(targets):
                b = self.r.foot_bid[i]
                R = self.d.xmat[b].reshape(3, 3)
                p = self.d.xpos[b] + R @ SOLE_PT
                ep = pt - p
                Re = R.T @ Rt
                eo = R @ (0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]))
                mujoco.mj_jac(self.m, self.d, jp, jr, p, b)
                cols = self.r.vadr[i * 6:(i + 1) * 6]
                J = np.vstack([jp[:, cols], jr[:, cols]])
                dq = np.linalg.solve(J.T @ J + 1e-8 * np.eye(6), J.T @ np.r_[ep, eo])
                dq *= min(1.0, 0.2 / max(np.abs(dq).max(), 1e-12))
                q[i * 6:(i + 1) * 6] += dq
                err_max = max(err_max, np.abs(ep).max(), np.abs(eo).max())
            if err_max < 1e-10:
                break
        return q, err_max

    def statics(self, q20, Rb):
        """quasi-static joint torques with the base orientation Rb in the world: min-norm split of
        the two 6-D foot wrenches balancing gravity.  Returns tau(12), CoP margins, CoM margin."""
        d, m = self.d, self.m
        d.qpos[:] = 0; d.qpos[3:7] = 0
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, Rb.reshape(-1)); d.qpos[3:7] = quat
        d.qpos[self.r.qadr] = q20
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        h = d.qfrc_bias.copy()                    # gravity only (qvel = 0)
        Js = []; P = []
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        for b in self.r.foot_bid:
            R = d.xmat[b].reshape(3, 3); p = d.xpos[b] + R @ SOLE_PT
            mujoco.mj_jac(m, d, jp, jr, p, b)
            Js.append(np.vstack([jp, jr]).copy()); P.append(p)
        Jc = np.vstack(Js)                        # 12 x nv  (force at sole centre + moment)
        # min-norm split, moments weighted as forces at a 2 cm lever (keeps CoPs central, like
        # the WBC's force-minimising QP)
        Wi = np.tile(np.r_[np.ones(3), np.full(3, 0.02)], 2)
        F = Wi * np.linalg.lstsq(Jc[:, :6].T * Wi[None, :], h[:6], rcond=None)[0]
        tau = h[6:] - Jc[:, 6:].T @ F
        tau12 = tau[self.r.vadr[:NL] - 6]
        cop = []
        for i in range(2):
            f, mo = F[6 * i:6 * i + 3], F[6 * i + 3:6 * i + 6]
            if f[2] < 1e-3:
                cop.append(-1.0); continue
            cx, cy = -mo[1] / f[2], mo[0] / f[2]   # CoP offset from sole centre (world ~ foot)
            cop.append(min(SOLE_HALF[0] - abs(cx), SOLE_HALF[1] - abs(cy)))
        com = d.subtree_com[0]
        # support polygon = convex hull of the 8 sole corners (world)
        pts = np.vstack([d.xpos[b] + FOOT_CORNERS @ d.xmat[b].reshape(3, 3).T for b in self.r.foot_bid])[:, :2]
        com_m = _hull_margin(pts, com[:2])
        fz = (F[2], F[8])
        return tau12, cop, com_m, fz


def _hull_margin(pts, c):
    from itertools import combinations
    # convex hull (monotone chain)
    P = sorted(map(tuple, pts))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    H = np.array(lo[:-1] + up[:-1])
    m = np.inf
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        e = b - a; n = np.array([-e[1], e[0]]) / np.linalg.norm(e)   # inward normal (CCW)
        m = min(m, float(n @ (c - a)))
    return m


# ------------------------------------------------------------------ motions
def sway(t, amp=1.0):
    """replica of experiment.exploration_targets (active) applied to the base (no ramp)."""
    f = np.array([0.3, 0.7, 1.3]); A = np.array([0.012, 0.008, 0.004]); ph = np.array([0.0, 1.1, 2.3])
    x = np.sum(A * np.sin(2 * np.pi * f * t + ph))
    y = 0.045 * np.sin(2 * np.pi * 0.25 * t) + 0.004 * np.sin(2 * np.pi * 1.1 * t)
    dz = -0.012 * (1 - np.cos(2 * np.pi * 0.4 * t))
    pitch = 0.06 * np.sin(2 * np.pi * 0.55 * t)
    return amp * np.array([x, y, dz, 0.0, pitch, 0.0])


def make_motion(sway_amp=1.0, squat=0.0, squat_cyc=0.0, yaw_amp=0.0, yaw_f=0.35, roll_amp=0.0,
                roll_f=0.45, base_dx=0.0, stagger=0.0, width=0.0, foot_yaw=0.0):
    """Returns (feet world poses, base pose fn).  squat: static extra base lowering (m, >0 down);
    squat_cyc: amplitude of an extra slow (0.2 Hz) squat cycle; yaw_amp: base yaw twist (rad);
    roll_amp: base roll (rad); base_dx: static fore-aft base offset over the feet; stagger: L foot
    forward / R foot back by +-stagger/2 (operator-placed stance); width: extra half-width (m);
    foot_yaw: toe-out of both feet (rad)."""
    def base(t):
        s = sway(t, sway_amp)
        s[0] += base_dx
        s[2] += -squat - squat_cyc * (1 - np.cos(2 * np.pi * 0.2 * t))
        s[3] += roll_amp * np.sin(2 * np.pi * roll_f * t + 0.7)
        s[5] += yaw_amp * np.sin(2 * np.pi * yaw_f * t + 0.3)
        return s
    feet = dict(stagger=stagger, width=width, foot_yaw=foot_yaw)
    return feet, base


def trajectory(K, motion):
    feet, base = motion
    w = K.w0 + feet["width"]
    # world: soles on z = 0; base nominal at (x0 offset) height z0
    Fp = [np.array([feet["stagger"] / 2, w, 0.0]), np.array([-feet["stagger"] / 2, -w, 0.0])]
    FR = [rpy(0, 0, feet["foot_yaw"]), rpy(0, 0, -feet["foot_yaw"])]
    ts = np.arange(0, T, 1 / FS)
    q = K.q_nom.copy()
    Q, Gb, worst = [], [], dict(tau=0.0, cop=np.inf, com=np.inf, lim=np.inf, ik=0.0, fz_min=np.inf, tilt=0.0)
    for t in ts:
        x, y, dz, r_, p_, y_ = base(t)
        pb = np.array([-K.x0 + x, y, K.z0 + dz]); Rb = rpy(r_, p_, y_)
        tg = [(Rb.T @ (Fp[i] - pb), Rb.T @ FR[i]) for i in range(2)]
        q, err = K.ik(q, tg)
        Q.append(q[:NL].copy()); Gb.append(Rb.T @ np.array([0, 0, 1.0]))
        tau, cop, comm, fz = K.statics(q, Rb)
        lim = np.min(np.r_[q[:NL] - K.r.lo[:NL], K.r.hi[:NL] - q[:NL]])
        worst["tau"] = max(worst["tau"], float(np.abs(tau).max()))
        worst["cop"] = min(worst["cop"], float(min(cop)))
        worst["com"] = min(worst["com"], comm)
        worst["lim"] = min(worst["lim"], float(np.rad2deg(lim)))
        worst["ik"] = max(worst["ik"], float(err))
        worst["knee"] = min(worst.get("knee", np.inf), float(np.rad2deg(min(q[3], q[9]))))
        worst["fz_min"] = min(worst["fz_min"], float(min(fz)) / (K.mass * 9.81))
        worst["tilt"] = max(worst["tilt"], float(np.rad2deg(np.arccos(np.clip(Rb[2, 2], -1, 1)))))
    return np.array(Q), np.array(Gb), worst


def info(K, Q, Gb, known_stance=False):
    est = K.est
    th = np.zeros(NP); th[NL:] = est.nominal_stance()
    _, J = est.residuals(th, Q, Gb)
    Jw = (J / est.sig[None, :, None]).reshape(-1, NP)
    H = Jw.T @ Jw
    if known_stance:
        return H[:NL, :NL]
    return H[:NL, :NL] - H[:NL, NL:] @ np.linalg.solve(H[NL:, NL:], H[NL:, :NL])


def safe(w):
    return (w["tau"] < TAU_LIM and w["cop"] > 0.005 and w["com"] > 0.02 and w["lim"] > 2.0
            and w["ik"] < 1e-6 and w["knee"] > 10.0 and w["tilt"] < 6.0 and w["fz_min"] > 0.05)


D2R = np.deg2rad
S0 = 0.02     # base 2 cm below the URDF default stand: in the simulated Phase-2 runs the knees stay
              # at 19-58 deg (cached samples); at the URDF height the +-4.5 cm lateral unloading
              # would straighten the loaded knee to ~6 deg (singular) in a pure-kinematic replica.
CANDIDATES = [
    ("baseline: Phase-2 sway (replica)", dict(squat=S0)),
    ("quiet stand (no motion)", dict(sway_amp=0.0, squat=S0)),
    # squat depth sweep (static extra lowering under the sway)
    ("sway + squat -2 cm (total -4)", dict(squat=S0 + 0.02)),
    ("sway + squat -4 cm (total -6)", dict(squat=S0 + 0.04)),
    ("sway + squat -6 cm (total -8)", dict(squat=S0 + 0.06)),
    ("sway + squat -8 cm (total -10)", dict(squat=S0 + 0.08)),
    ("sway + squat cycle 0..-6 cm (0.2 Hz)", dict(squat=S0, squat_cyc=0.03)),
    ("sway + squat cycle 0..-10 cm (0.2 Hz)", dict(squat=S0, squat_cyc=0.05)),
    # fore-aft: base over the feet (= both feet shifted back / forward relative to base)
    ("sway + base +2 cm fwd", dict(squat=S0, base_dx=0.02)),
    ("sway + base -2 cm back", dict(squat=S0, base_dx=-0.02)),
    ("sway + base +2 cm fwd, squat -4", dict(squat=S0 + 0.04, base_dx=0.02)),
    # lateral stance width (operator-placed)
    ("sway + width +3 cm", dict(squat=S0, width=0.015)),
    ("sway + width +6 cm", dict(squat=S0 + 0.01, width=0.03)),
    ("sway + width -2 cm", dict(squat=S0, width=-0.01)),
    # asymmetric stances (operator-placed stagger / toe-out)
    ("sway + stagger 4 cm (L fwd)", dict(squat=S0, stagger=0.04)),
    ("sway + stagger 8 cm (L fwd)", dict(squat=S0 + 0.01, stagger=0.08)),
    ("sway + stagger 12 cm (L fwd)", dict(squat=S0 + 0.02, stagger=0.12)),
    ("sway + toe-out 10 deg", dict(squat=S0, foot_yaw=D2R(10))),
    # in-place asymmetry WITHOUT re-placing the feet: base yaw twist / base roll
    ("sway + yaw twist +-5 deg", dict(squat=S0, yaw_amp=D2R(5))),
    ("sway + yaw twist +-10 deg", dict(squat=S0, yaw_amp=D2R(10))),
    ("sway + yaw twist +-15 deg", dict(squat=S0, yaw_amp=D2R(15))),
    ("sway + yaw twist +-20 deg", dict(squat=S0, yaw_amp=D2R(20))),
    ("sway + yaw twist +-15 deg @0.6 Hz", dict(squat=S0, yaw_amp=D2R(15), yaw_f=0.6)),
    ("sway + base roll +-3 deg", dict(squat=S0, roll_amp=D2R(3))),
    ("sway + base roll +-5 deg", dict(squat=S0, roll_amp=D2R(5))),
    # combinations
    ("sway + squat cyc 0..-6 + yaw +-10 deg", dict(squat=S0, squat_cyc=0.03, yaw_amp=D2R(10))),
    ("sway + squat cyc 0..-6 + yaw +-15 deg", dict(squat=S0, squat_cyc=0.03, yaw_amp=D2R(15))),
    ("sway + squat -2 (tot -4) + yaw +-15 deg", dict(squat=S0 + 0.02, yaw_amp=D2R(15))),
    ("sway + squat -2 (tot -4) + yaw +-20 deg", dict(squat=S0 + 0.02, yaw_amp=D2R(20))),
    ("sway + squat cyc 0..-6 + stagger 8 cm", dict(squat=S0 + 0.01, squat_cyc=0.03, stagger=0.08)),
    ("sway + yaw +-10 + roll +-3 deg", dict(squat=S0, yaw_amp=D2R(10), roll_amp=D2R(3))),
    ("yaw twist +-10 deg only (no sway)", dict(sway_amp=0.0, squat=S0, yaw_amp=D2R(10))),
    ("yaw twist +-15 deg only (no sway)", dict(sway_amp=0.0, squat=S0, yaw_amp=D2R(15))),
]


def baseline_cached(K, n_weak=4):
    """weak directions = the n_weak weakest eigenvectors of the noiseless information of the
    ACTUAL seed-0 Phase-2 calibration (cached simulated samples, true q), i.e. the directions the
    deployed estimator left on the prior (seed 0 committed 8/12).  K factor per direction =
    deployed-estimator marginal std / noiseless 6 s marginal std (tau_int + misfit inflation)."""
    rep = json.load(open(os.path.join(OUT, "calib_active_0.json")))["free"]
    Vd = np.array(rep["eigvecs"]); sd = np.array(rep["eig_std_deg"])
    from scenarios import true_params_q0
    z = np.load(os.path.join(OUT, "samples_active_0.npz"))
    Qt = z["Q"] - np.array(true_params_q0(0).q0[:12])[None, :]
    H = info(K, Qt, z["Gb"]) * (FS * T / len(Qt))          # rescale to 600 samples / 6 s
    lam, V = np.linalg.eigh(H)
    Vw = V[:, :n_weak]
    Hi = np.linalg.inv(H + EPS * np.eye(NL))
    Hdi = np.linalg.inv(Vd @ np.diag(1.0 / np.deg2rad(sd) ** 2) @ Vd.T + EPS * np.eye(NL))
    s_nl = np.array([np.sqrt(v @ Hi @ v) for v in Vw.T]); s_dep = np.array([np.sqrt(v @ Hdi @ v) for v in Vw.T])
    std_eig = np.rad2deg(1 / np.sqrt(np.maximum(lam, 0) + EPS))
    return Vw, s_dep / s_nl, np.rad2deg(s_nl), np.rad2deg(s_dep), std_eig


def main(cands=CANDIDATES):
    K = Kin()
    rows = []
    Vw, kfac, sd_nl_cached, sd_dep, std_eig_cached = baseline_cached(K)
    kmax = float(kfac.max())
    for name, kw in cands:
        Q, Gb, w = trajectory(K, make_motion(**kw))
        H = info(K, Q, Gb)
        lam, V = np.linalg.eigh(H); lam = np.maximum(lam, 1e-30)
        Hi = np.linalg.inv(H + EPS * np.eye(NL))
        wstd = [float(np.rad2deg(np.sqrt(max(v @ Hi @ v, 0)))) for v in Vw.T]
        Hj = info(K, Q, Gb, known_stance=True); Hji = np.linalg.inv(Hj + EPS * np.eye(NL))
        wstd_jig = [float(np.rad2deg(np.sqrt(max(v @ Hji @ v, 0)))) for v in Vw.T]
        std_all = np.rad2deg(1 / np.sqrt(lam + EPS))
        row = dict(name=name, kw={k: float(v) for k, v in kw.items()}, safe=bool(safe(w)), worst=w,
                   rel_sv=(np.sqrt(lam / lam.max())).tolist(), std_eig_deg=std_all.tolist(),
                   weak_dir_std_deg=wstd, weak_dir_std_jig_deg=wstd_jig,
                   weak_dir_std_deployed_eq_deg=[float(s * k) for s, k in zip(wstd, kfac)],
                   worst2_eig_std_deg=std_all[:2].tolist(),
                   worst2_eig_std_deployed_eq_deg=(std_all[:2] * kmax).tolist(),
                   n_dirs_below_0p5_deployed_eq=int((std_all * kmax < 0.5).sum()),
                   n_dirs_below_2_deployed_eq=int((std_all * kmax < 2.0).sum()),
                   weakest_dirs=[label_dir(V[:, i]) for i in range(3)])
        rows.append(row)
        print(f"{name:42s} safe={row['safe']!s:5s} tau={w['tau']:5.1f} cop={1e3*w['cop']:5.1f}mm com={1e3*w['com']:5.1f}mm "
              f"lim={w['lim']:5.1f} knee={w['knee']:4.1f} ik={w['ik']:.0e} tilt={w['tilt']:4.1f} | rel.sv min {row['rel_sv'][0]:.1e} {row['rel_sv'][1]:.1e} | "
              f"weak-dir std dep-eq " + " ".join("%6.2f" % x for x in row['weak_dir_std_deployed_eq_deg']) +
              f" | worst eig dep-eq {row['worst2_eig_std_deployed_eq_deg'][0]:7.2f} | n<0.5: {row['n_dirs_below_0p5_deployed_eq']} n<2: {row['n_dirs_below_2_deployed_eq']}",
              flush=True)
    meta = dict(fs=FS, T=T, weak_dirs_baseline=[label_dir(v) for v in Vw.T], weak_dirs_vec=Vw.T.round(4).tolist(),
                deployed_factor=kfac.tolist(), baseline_weak_std_noiseless_from_cached_samples=sd_nl_cached.tolist(),
                baseline_weak_std_deployed_seed0=sd_dep.tolist(), cached_eig_std_noiseless_6s=std_eig_cached.tolist())
    print("weak dirs:", meta["weak_dirs_baseline"], "deployed factor K:", kfac,
          "cached-sample noiseless std:", sd_nl_cached, "deployed:", sd_dep)
    return rows, meta


if __name__ == "__main__":
    rows, meta = main()
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(OUT, "motion_scan.json"), "w"), indent=1)
