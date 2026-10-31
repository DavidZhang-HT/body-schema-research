#!/usr/bin/env python
"""E2 Step 2: dynamic positive control for SUPINE->SIDE on scheme A's 20-DoF plant.

Method (choice (b) from the task: sampling MPC / CEM over trajectory-parameterized
actions; documented in EXP_E2_REPORT.md):
  * MuJoCo forward dynamics on model.PiPlus(NOMINAL, dt=0.001, shell="full").
  * Receding-horizon blocks: T_BLOCK s each; per block, CEM optimizes
    (n_wp x 20) joint-waypoint DELTAS (executed by a joint PD:
    tau = Kp*(q_t - q) - Kd*qd, clipped to TAU_MAX).  Waypoint deltas are
    clipped to +/-DELTA and joint limits.
  * Cost rewards roll progress p = sign*R[2,1] (SIDE needs p >= 0.7):
      cost = -W_END*p_end - W_MAX*p_max - W_MEAN*p_mean
             + W_TAU*mean|tau| + W_PEN*selfpen_depth
    NaN/unstable rollouts get cost = INF.
  * After each executed block, the geometric SIDE predicate (mode_check with a
    trivially-feasible cert) is evaluated.  On first satisfaction, a 1.5 s
    zero-torque settle runs; if the predicate still holds afterwards, the start
    is SUCCESS.  The full mode_check (with the real statics LP) is reported.
  * Per block logged: peak KE [J], peak CoM speed [m/s], peak |qvel|,
    max/mean/end roll progress, min self-dist, predicate result.

Usage:
  python step2.py --sweep      # random-waypoint sweep: can anything tip it? (diagnostic)
  python step2.py --start 0    # run one start (index into make_starts order)
  python step2.py --all        # all 6 starts, sequential
  python step2.py --all --workers 2   # all 6 starts, 2 parallel processes
"""
import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

from common import Plant, ModeEval, make_starts, rollout_block
from model import JOINTS

# ------------------------------------------------------------------ config: STEP 2 (dynamic, no KE cap)
# Method: Phase 1 (scripted, quasi-static) moves the down-side arm forward so it
# is not trapped under the trunk -- diag_side_pose.py showed the trapped arm
# destabilizes SIDE (topples to prone) while arm-forward makes SIDE a stable
# attractor (pred=True after 3 s zero-torque settle).  Phase 2 is the searched
# dynamic roll (CEM, no KE cap).  The down-side arm is HELD at its Phase-1 pose
# during Phase 2 (mask) so the stabilizer is not undone by the search.
CFG = dict(
    T_PHASE1=2.0,
    T_BLOCK=1.5, CTRL_DT=0.02, N_WP=3, DELTA=1.0,
    N=64, ELITE=10, ITERS=5, SIGMA0=0.5, SIGMA_MIN=0.05, SIGMA_MAX=0.6,
    W_END=30.0, W_MAX=15.0, W_MEAN=5.0, W_OVER=20.0, W_KEEND=3.0,
    W_TAU=0.01, W_PEN=500.0, W_KE=0.0,
    MAX_BLOCKS=12, SETTLE_T=1.5, SETTLE_KE=0.1,
)
# down-side arm joints held at Phase-1 pose during Phase 2 (per roll direction)
HOLD_JOINTS = {
    +1: ["r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint"],
    -1: ["l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint"],
}
# Phase-1 target for the down-side arm (forward across the chest)
PHASE1_ARM = {"shoulder_pitch_joint": -1.5, "elbow_joint": 0.8}
INF = 1e9


def block_cost(trace, sign, cfg):
    if not trace["ok"]:
        return INF
    r21 = np.array(trace["R21"]) * sign
    p_end, p_max, p_mean = r21[-1], r21.max(), r21.mean()
    tau = np.abs(np.array(trace["tau"])).mean()
    pen = trace["max_selfpen_m"]
    ke_term = cfg["W_KE"] * trace["peak_ke"]
    over = max(0.0, float(p_max) - 0.95)          # bonus for clearing the ~90° hump
    ke_end_shaped = cfg["W_KEEND"] * trace["ke_end"] / (1.0 + np.exp(-10.0 * (p_end - 0.6)))
    return (-cfg["W_END"] * p_end - cfg["W_MAX"] * p_max - cfg["W_MEAN"] * p_mean
            - cfg["W_OVER"] * over
            + cfg["W_TAU"] * tau + cfg["W_PEN"] * pen + ke_term + ke_end_shaped)


def phase1_scripted(plant, sign, cfg):
    """Quasi-static preparatory motion: bring the down-side arm forward across
    the chest so it is not trapped under the trunk during the roll."""
    j = {n: i for i, n in enumerate(JOINTS)}
    side = "r" if sign == +1 else "l"
    q_cur = plant.d.qpos[plant.qadr].copy()
    q_tgt = q_cur.copy()
    for jname, val in PHASE1_ARM.items():
        q_tgt[j[f"{side}_{jname}"]] = val
    q_tgt = np.clip(q_tgt, plant.lo, plant.hi)
    tr = rollout_block(plant, None, np.array([q_tgt]), cfg["T_PHASE1"], ctrl_dt=cfg["CTRL_DT"])
    return tr


def cem_block(plant, mev, sign, seed, cfg, log_prefix="", hold_idx=None):
    """One CEM-optimized block from the plant's current state.

    hold_idx: joint indices (JOINTS order) held fixed at current q (overridden
    after sampling, so the search cannot undo the Phase-1 stabilizer)."""
    rng = np.random.default_rng(seed)
    snap = plant.snapshot()
    q0 = plant.d.qpos[plant.qadr].copy()
    lo = np.maximum(plant.lo, q0 - cfg["DELTA"])
    hi = np.minimum(plant.hi, q0 + cfg["DELTA"])
    D = cfg["N_WP"] * 20
    mu = np.zeros(D)
    sigma = np.full(D, cfg["SIGMA0"])
    best = {"cost": INF, "delta": None}
    for it in range(cfg["ITERS"]):
        deltas = np.clip(rng.normal(mu, sigma, (cfg["N"], D)), -cfg["DELTA"], cfg["DELTA"])
        costs = np.empty(cfg["N"])
        for n in range(cfg["N"]):
            w = np.clip(q0 + deltas[n].reshape(cfg["N_WP"], 20), lo, hi)
            if hold_idx is not None:
                w[:, hold_idx] = q0[hold_idx][None, :]
            plant.restore(snap)
            tr = rollout_block(plant, None, w, cfg["T_BLOCK"], ctrl_dt=cfg["CTRL_DT"])
            c = block_cost(tr, sign, cfg)
            costs[n] = c
            if c < best["cost"]:
                best = {"cost": float(c), "delta": deltas[n].copy()}
        elite = np.argsort(costs)[:cfg["ELITE"]]
        mu = deltas[elite].mean(axis=0)
        sigma = np.clip(deltas[elite].std(axis=0), cfg["SIGMA_MIN"], cfg["SIGMA_MAX"])
        print(f"{log_prefix}  CEM it{it}: best={costs.min():+.2f} "
              f"elite_mean={costs[elite].mean():+.2f}", flush=True)
    w_best = np.clip(q0 + best["delta"].reshape(cfg["N_WP"], 20), lo, hi)
    if hold_idx is not None:
        w_best[:, hold_idx] = q0[hold_idx][None, :]
    plant.restore(snap)
    trace = rollout_block(plant, None, w_best, cfg["T_BLOCK"], ctrl_dt=cfg["CTRL_DT"])
    info = {"cem_best_cost": best["cost"], "waypoints": w_best.tolist()}
    return w_best, trace, info


def settle(plant, mev, dst, T):
    """Zero-torque settle from a snapshot; restores the snapshot afterwards.

    Returns (result_dict, was_already_saved_state).  The caller must restore the
    pre-settle snapshot on failure so a rock that falls back does not destroy
    the block-end state (receding-horizon chaining)."""
    import mujoco
    snap = plant.snapshot()
    plant.d.ctrl[:] = 0.0
    sub = int(round(0.02 / plant.m.opt.timestep))
    n = int(round(T / 0.02))
    for _ in range(n):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
    q = plant.d.qpos.copy()
    ok_pred, why_pred, _ = mev.predicate(q, dst)
    ok_full, why_full, cert, _ = mev.full_check(q, dst)
    res = {"predicate_ok": bool(ok_pred), "predicate_why": why_pred,
           "full_ok": bool(ok_full), "full_why": why_full,
           "rho": float(cert["rho"]), "cert_feasible": bool(cert["feasible"])}
    return res, snap


def run_start(tag, q_start, dst, sign, seed, cfg, outdir, mev=None):
    t0 = time.time()
    plant = Plant(timestep=0.001)
    if mev is None:
        mev = ModeEval()
    plant.spawn(q_start)
    j = {n: i for i, n in enumerate(JOINTS)}
    hold_idx = [j[n] for n in HOLD_JOINTS[sign]]
    rec = {"tag": tag, "dst": dst, "sign": sign, "seed": seed,
           "cfg": cfg, "blocks": [], "status": "RUNNING"}
    print(f"\n===== start {tag} -> {dst} (sign {sign:+d}, seed {seed}) =====", flush=True)
    all_qpos, all_qvel = [], []
    # ---- Phase 1: scripted arm-forward (quasi-static preparation)
    tr1 = phase1_scripted(plant, sign, cfg)
    all_qpos += tr1["qpos"]
    all_qvel += tr1["qvel"]
    rec["phase1"] = {"peak_ke_J": tr1["peak_ke"], "ke_end_J": tr1["ke_end"],
                     "max_selfpen_mm": tr1["max_selfpen_m"] * 1e3}
    print(f"[{tag}] phase1 done: peakKE={tr1['peak_ke']:.4f}J", flush=True)
    # ---- Phase 2: CEM dynamic roll (down-side arm held)
    success = False
    for b in range(cfg["MAX_BLOCKS"]):
        w, trace, info = cem_block(plant, mev, sign,
                                  seed * 1000 + b, cfg,
                                  log_prefix=f"[{tag} block {b}]",
                                  hold_idx=hold_idx)
        q_end = plant.d.qpos.copy()
        ok_pred, why_pred, _ = mev.predicate(q_end, dst)
        r21 = np.array(trace["R21"]) * sign
        blk = {
            "block": b,
            "peak_ke_J": trace["peak_ke"],
            "ke_end_J": trace["ke_end"],
            "peak_com_speed_ms": trace["peak_com_speed"],
            "peak_qvel": float(np.abs(np.array(trace["qvel"])).max()),
            "roll_end": float(r21[-1]), "roll_max": float(r21.max()),
            "roll_mean": float(r21.mean()),
            "min_self_dist_mm": float(np.array(trace["selfd"]).min() * 1e3),
            "max_selfpen_mm": float(trace["max_selfpen_m"] * 1e3),
            "cem_best_cost": info["cem_best_cost"],
            "predicate_ok": bool(ok_pred),
            "predicate_why": why_pred,
            "wall_s": round(time.time() - t0, 1),
        }
        rec["blocks"].append(blk)
        all_qpos += trace["qpos"]
        all_qvel += trace["qvel"]
        print(f"[{tag} block {b}] peakKE={blk['peak_ke_J']:.3f}J "
              f"roll end/max={blk['roll_end']:+.3f}/{blk['roll_max']:+.3f} "
              f"predicate={ok_pred}", flush=True)
        if ok_pred:
            # Settle test only when the block ended calm; a high-KE transient
            # (mid-rock) chains to the next block instead of being reset.
            if trace["ke_end"] < cfg["SETTLE_KE"]:
                s, snap_pre = settle(plant, mev, dst, cfg["SETTLE_T"])
                blk["settle"] = s
                print(f"[{tag}] predicate satisfied (ke_end={trace['ke_end']:.3f}) -> settle: "
                      f"predicate_ok={s['predicate_ok']} full_ok={s['full_ok']} rho={s['rho']:.3f}",
                      flush=True)
                if s["predicate_ok"]:
                    success = True
                    rec["settle"] = s
                    break
                plant.restore(snap_pre)  # failed settle: keep block-end state, chain on
            else:
                print(f"[{tag}] predicate transient (ke_end={trace['ke_end']:.3f}) -> chaining",
                      flush=True)
    rec["status"] = "SUCCESS" if success else "FAIL_STALL"
    rec["n_blocks"] = len(rec["blocks"])
    rec["wall_s"] = round(time.time() - t0, 1)
    qf = plant.d.qpos.copy()
    ok_pred, why_pred, _ = mev.predicate(qf, dst)
    ok_full, why_full, cert, _ = mev.full_check(qf, dst)
    rec["final"] = {"predicate_ok": bool(ok_pred), "predicate_why": why_pred,
                    "full_ok": bool(ok_full), "full_why": why_full,
                    "rho": float(cert["rho"]), "cert_feasible": bool(cert["feasible"]),
                    "R21": float(plant.base_R()[2, 1])}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"step2_{tag}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.join(outdir, f"traj_step2_{tag}.npz"),
                        qpos=np.array(all_qpos), qvel=np.array(all_qvel))
    print(f"[{tag}] {rec['status']} after {rec['n_blocks']} blocks, "
          f"wall={rec['wall_s']:.0f}s", flush=True)
    return rec


def sweep(n_samples=200, seed=7):
    """Diagnostic: random waypoint blocks (no optimization) -- can anything tip it?"""
    mev = ModeEval()
    starts, _ = make_starts(mev)
    tag, q_start, dst, sign, _ = starts[0]
    rng = np.random.default_rng(seed)
    cfg = CFG
    plant = Plant(timestep=0.001)
    best = 0.0
    dist = []
    for n in range(n_samples):
        plant.spawn(q_start)
        q0 = plant.d.qpos[plant.qadr].copy()
        deltas = np.clip(rng.normal(0, 0.5, (cfg["N_WP"], 20)), -1.0, 1.0)
        w = np.clip(q0 + deltas,
                    np.maximum(plant.lo, q0 - 1.0), np.minimum(plant.hi, q0 + 1.0))
        tr = rollout_block(plant, None, w, cfg["T_BLOCK"], ctrl_dt=cfg["CTRL_DT"])
        m = float((np.array(tr["R21"]) * sign).max()) if tr["ok"] else -9
        dist.append(m)
        best = max(best, m)
        if (n + 1) % 40 == 0:
            print(f"sweep {n+1}/{n_samples}: best max|R21|={best:.3f}", flush=True)
    dist = np.array(dist)
    print(f"SWEEP done: best={dist.max():.3f} mean={dist.mean():.3f} "
          f"frac>0.3={(dist > 0.3).mean():.3f} frac>0.5={(dist > 0.5).mean():.3f} "
          f"frac>0.7={(dist > 0.7).mean():.3f}")


def _worker(args):
    tag, q_start, dst, sign, seed, outdir = args
    try:
        return run_start(tag, q_start, dst, sign, seed, CFG, outdir)
    except Exception:
        return {"tag": tag, "status": "ERROR", "error": traceback.format_exc(limit=5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--outdir", default=os.path.join(HERE, "step2"))
    a = ap.parse_args()

    if a.sweep:
        sweep()
        return

    mev = ModeEval()
    starts, verified = make_starts(mev)
    print("start verification:", json.dumps(verified, indent=1)[:800], flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "starts.json"), "w") as f:
        json.dump(verified, f, indent=1)

    tasks = [(tag, q, dst, sign, seed, a.outdir) for (tag, q, dst, sign, seed) in starts]
    if a.start is not None:
        tasks = [tasks[a.start]]
    elif not a.all:
        tasks = tasks[:1]

    os.makedirs(a.outdir, exist_ok=True)
    if a.workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(f"{r['tag']}: {r['status']} blocks={r.get('n_blocks')} "
              f"wall={r.get('wall_s')}", flush=True)


if __name__ == "__main__":
    main()
