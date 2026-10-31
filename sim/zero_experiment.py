"""Validation of joint zero-offset identification (calibration only, deterministic, CPU).

Usage:  .venv/bin/python zero_experiment.py calib <seed> {active|quiet}
        .venv/bin/python zero_experiment.py calib_all            (seeds 0-4, both modes)
Writes results/q0/calib_<mode>_<seed>.json

Plant = scenarios.true_params_q0(seed): the base mismatched plant of that seed (mass, CoM,
delay, friction, soft pads, mu, backlash) + KNOWN leg joint zero offsets.  Controller =
URDF-nominal, offsets unknown (q0_hat = 0).  Same 1 s stand + 6 s double-support calibration
as experiment.episode; 'active' = Phase-2 multi-frequency sway/squat/unloading, 'quiet' =
quiet standing (passive baseline).
"""
import os, sys, json, time
import numpy as np
import mujoco
from model import RealRobot, BodyParams, JOINTS, LEG_JOINTS
from wbc import WBC, BodyModel, GaitParams, SOLE_PT
from scenarios import true_params_q0, Q0_PRIOR_STD_DEG
from experiment import exploration_targets, REF
from zero_id import ZeroEstimator, label_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "q0")
NAMES = [j.replace("_joint", "") for j in LEG_JOINTS]
# Phase-2b motion variants selected by zero_motion_scan.py (trunk yaw twist about planted feet)
MODES = {"active": {}, "quiet": {},
         "yaw15": dict(yaw_twist=np.deg2rad(15.0), yaw_f=0.35),
         "yaw15sq": dict(yaw_twist=np.deg2rad(15.0), yaw_f=0.35, squat_cyc=0.03),
         "yaw25": dict(yaw_twist=np.deg2rad(25.0), yaw_f=0.35)}


def _truth_snapshot(rob):
    """ground truth at the instant the last observation was produced (evaluation only):
    true leg q, qd, qacc, applied torque and MuJoCo joint-friction (frictionloss) torque."""
    d, r = rob.d, rob.r
    fr = np.zeros(rob.m.nv)
    for i in range(d.nefc):
        if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF:
            fr[d.efc_id[i]] += d.efc_force[i]
    return np.concatenate([d.qpos[r.qadr], d.qvel[r.vadr], d.qacc[r.vadr], rob.tau_applied, fr[r.vadr]])


def calibrate(seed, mode, T0=1.0, T_cal=6.0, record=False):
    """record=True additionally stores the FULL-RATE proprioceptive stream (q, qd, tau, IMU)
    and ground-truth diagnostics -> samples_tau_<mode>_<seed>.npz (torque-channel scan).
    Recording consumes no RNG -> the run is bit-identical to record=False."""
    tp = true_params_q0(seed)
    rob = RealRobot(tp, seed=seed)
    bm = BodyModel(BodyParams())
    wbc = WBC(bm, GaitParams(stride=0.06), mu_ctrl=0.6, ref=REF)
    zest = ZeroEstimator(prior_std=np.deg2rad(Q0_PRIOR_STD_DEG))
    o = rob.observe(); mid = None
    R = dict(t=[], q=[], qd=[], tau=[], quat=[], gyro=[], acc=[], gated=[], truth=[])
    truth_o = _truth_snapshot(rob)
    ank = [i for i, j in enumerate(LEG_JOINTS) if "ankle" in j]
    bl_bias, tilt, foot0, slip = [], [], None, 0.0
    while rob.t < T0 + T_cal:
        t = rob.t
        if t >= T0:
            if mid is None:
                bm.load_state(o)
                mid = 0.5 * (bm.point(wbc.feet["l"], SOLE_PT)[:2] + bm.point(wbc.feet["r"], SOLE_PT)[:2])
            wbc.planner.stand = exploration_targets(t - T0, T_cal, mid, active=(mode != "quiet"),
                                                    **MODES.get(mode, {}))
        tau, info = wbc.control(o, t)
        gated = False
        if t > T0 + 0.3:
            # ground-truth diagnostics (evaluation only): effective ankle backlash encoder bias
            b = np.zeros(12); b[ank] = 0.5 * tp.backlash * np.tanh(rob.tau_applied[ank] / 0.3)
            if zest.add(o, info, wbc, t):
                gated = True
                bl_bias.append(b)
                zs = [rob.d.xmat[f].reshape(3, 3)[:, 2] for f in rob.r.foot_bid]
                tilt.append([np.rad2deg(np.arccos(np.clip(z[2], -1, 1))) for z in zs])
                fp = np.array([rob.d.xpos[f][:2] for f in rob.r.foot_bid])
                foot0 = fp if foot0 is None else foot0
                slip = max(slip, float(np.abs(fp - foot0).max()))
        if record and t >= T0:
            for k, v in (("t", t), ("q", o.q), ("qd", o.qd), ("tau", o.tau), ("quat", o.base_quat),
                         ("gyro", o.gyro), ("acc", o.acc), ("gated", gated), ("truth", truth_o)):
                R[k].append(np.copy(v))
        o = rob.step(tau)
        if record:
            truth_o = _truth_snapshot(rob)
        if rob.fallen():
            break
    if record:
        np.savez(os.path.join(OUT, f"samples_tau_{mode}_{seed}.npz"), fell=rob.fallen(),
                 **{k: np.array(v) for k, v in R.items()})
    t0 = time.time()
    np.savez(os.path.join(OUT, f"samples_{mode}_{seed}.npz"), Q=np.array(zest.Q), Gb=np.array(zest.Gb),
             tilt=np.array(tilt), bl=np.array(bl_bias), n_seen=zest.n_seen, n_gate_rej=zest.n_gate_rej,
             fell=rob.fallen(), slip=slip)
    return analyse(zest, tp, seed, mode, tilt, bl_bias, slip, bool(rob.fallen()))


def resolve(seed, mode, **kw):
    """re-run only the estimator on cached gated samples (no re-simulation)."""
    z = np.load(os.path.join(OUT, f"samples_{mode}_{seed}.npz"))
    zest = ZeroEstimator(prior_std=np.deg2rad(Q0_PRIOR_STD_DEG), **kw)
    zest.Q, zest.Gb = list(z["Q"]), list(z["Gb"])
    zest.n_seen, zest.n_gate_rej = int(z["n_seen"]), int(z["n_gate_rej"])
    return analyse(zest, true_params_q0(seed), seed, mode, list(z["tilt"]), list(z["bl"]),
                   float(z["slip"]), bool(z["fell"]))


def analyse(zest, tp, seed, mode, tilt, bl_bias, slip, fell):
    t0 = time.time()
    dhat, info_free = zest.solve()
    lm = getattr(zest, "loop_mask", np.ones(len(tilt), bool))
    tilt_kept = np.array(tilt)[lm] if len(tilt) else np.zeros((0, 2))
    _, info_jig = zest.solve(known_stance=True)
    true = np.rad2deg(np.array(tp.q0[:12]))
    bl = np.rad2deg(np.mean(np.array(bl_bias)[lm], 0)) if len(bl_bias) else np.zeros(12)
    rep = dict(seed=seed, mode=mode, fell=fell, solve_s=time.time() - t0,
               true_deg=true.round(3).tolist(), backlash_bias_deg=bl.round(3).tolist(),
               true_foot_tilt_deg_max=float(np.max(tilt)) if len(tilt) else None,
               true_foot_tilt_deg_max_kept=float(np.max(tilt_kept)) if len(tilt_kept) else None,
               true_foot_slip_mm=1e3 * slip, free=info_free, jig=info_jig)
    for key, inf in (("free", info_free), ("jig", info_jig)):
        dm = np.array(inf.get("delta_map_deg", [0] * 12)); dc = np.array(inf.get("delta_commit_deg", [0] * 12))
        rep[key]["err_map_deg"] = (dm - true).round(3).tolist()
        rep[key]["err_commit_deg"] = (dc - true).round(3).tolist()
        rep[key]["err_commit_vs_true_plus_backlash_deg"] = (dc - true - bl).round(3).tolist()
        if "eigvecs" in inf:
            V = np.array(inf["eigvecs"])
            # per eigen-direction: formal std, committed?, truth / truth+backlash / MAP projections
            rep[key]["dirs"] = [dict(std_deg=s, obs=o_, dir=label_dir(V[:, i]),
                                     true=float(V[:, i] @ true), true_bl=float(V[:, i] @ (true + bl)),
                                     map=float(V[:, i] @ dm), commit=float(V[:, i] @ dc))
                                for i, (s, o_) in enumerate(zip(inf["eig_std_deg"], inf["obs_mask"]))]
            obs = np.array(inf["obs_mask"], bool)
            Vo, Vn = V[:, obs], V[:, ~obs]
            # error split: committed (observable) subspace vs. uncommitted subspace (stays on prior)
            e = dc - true
            rep[key]["err_rms_obs_subspace_deg"] = float(np.linalg.norm(Vo.T @ e) / np.sqrt(12))
            rep[key]["err_rms_obs_subspace_vs_true_plus_bl_deg"] = float(np.linalg.norm(Vo.T @ (dc - true - bl)) / np.sqrt(12))
            rep[key]["err_rms_null_subspace_deg"] = float(np.linalg.norm(Vn.T @ e) / np.sqrt(12))
            rep[key]["err_rms_deg"] = float(np.sqrt((e ** 2).mean()))
            rep[key]["prior_err_rms_deg"] = float(np.sqrt((true ** 2).mean()))
    return rep


def summary_line(rep, key="free"):
    e = np.array(rep[key]["err_commit_deg"]); t = np.array(rep["true_deg"])
    return (f"seed {rep['seed']} {rep['mode']:6s} {key:4s} n={rep[key]['n']} obs_dirs={rep[key].get('n_obs_dirs')} "
            f"commit={rep[key]['committed']} RMS err {np.sqrt((e**2).mean()):.2f} deg (prior {np.sqrt((t**2).mean()):.2f}; "
            f"obs-subspace {rep[key].get('err_rms_obs_subspace_deg', np.nan):.2f}, null {rep[key].get('err_rms_null_subspace_deg', np.nan):.2f}) | "
            + " ".join(f"{n}:{x:+.2f}" for n, x in zip(NAMES, e)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fn = calibrate
    if sys.argv[1] in ("resolve", "resolve_all"):
        fn = resolve
    if sys.argv[1] in ("calib", "resolve"):
        jobs = [(int(sys.argv[2]), sys.argv[3])]
    else:
        jobs = [(s, m) for s in range(5) for m in ("active", "quiet")]
    for seed, mode in jobs:
        t0 = time.time()
        rep = fn(seed, mode)
        rep["wall_s"] = time.time() - t0
        json.dump(rep, open(os.path.join(OUT, f"calib_{mode}_{seed}.json"), "w"), indent=1)
        print(summary_line(rep, "free"), flush=True)
        print(summary_line(rep, "jig"), flush=True)
        print("   rms before/after", rep["free"].get("rms_before"), rep["free"].get("rms_after"),
              "tilt(all/kept)", rep["true_foot_tilt_deg_max"], rep["true_foot_tilt_deg_max_kept"],
              "loop_rej", rep["free"].get("n_loop_rej"), "fell", rep["fell"], "slip_mm", round(rep["true_foot_slip_mm"], 2),
              "wall", round(rep["wall_s"], 1), flush=True)
