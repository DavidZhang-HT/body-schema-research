#!/usr/bin/env python
"""E2 Steps 2 & 3: direct trajectory optimization for SUPINE->SIDE.

Method (documented in EXP_E2_REPORT.md):
  Phase 1 (scripted, quasi-static, 2.0 s): move the down-side arm forward across
    the chest.  Justified by diag_side_pose.py: with the arm alongside, the
    trapped arm destabilizes SIDE (90-deg placement topples to prone); with the
    arm forward, SIDE is a stable attractor (3-s zero-torque settle: pred=True).
  Phase 2 (searched): CEM over an N_WP-waypoint joint trajectory (PD-tracked),
    down-side arm HELD at its Phase-1 pose (mask).  Cost uses a 1.0-s
    zero-torque settle simulated inside every sample:
        cost = -W_SETTLE*p_settle - W_END*p_end - W_MAX*p_max
               + W_TAU*mean|tau| + W_PEN*selfpen + W_KE*peak_KE
    i.e. the true objective (end up settled in SIDE) is optimized directly.
    The discovered Step-2 motion is a wind-up (counter-rock to -0.4) then a
    ballistic swing over the top (peak KE ~3.4 J).

  Step 2 (dynamic): T=4.0 s, N_WP=6, DELTA=1.0, SIGMA0=0.6, W_KE=0 (no cap).
  Step 3 (ablation): T=6.0 s, N_WP=8, DELTA=0.35, SIGMA0=0.12, W_KE=3000, and
    a HARD KE cap at trajectory end (<=0.005 J, ENFORCED): samples with
    peak_ke > 0.03 J are rejected in-CEM; after CEM the best trajectory is
    executed and if ke_end > 0.005 J it is shrunk (deltas *= 0.5, re-execute,
    up to 3 tries); persistent violation -> start FAILs.

Success (both steps): 1.5-s zero-torque settle from trajectory end keeps the
geometric SIDE predicate; the full mode_check (real statics LP) is reported.

Usage: python run_direct.py --step 2 --all [--workers 2]
       python run_direct.py --step 3 --start 0
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
import mujoco

from common import Plant, ModeEval, make_starts, rollout_block
from step2 import phase1_scripted, CFG as CFG2_BASE, HOLD_JOINTS, INF
from model import JOINTS

CFG2 = dict(T_TRAJ=4.0, N_WP=6, DELTA=1.0, SIGMA0=0.6, N=96, ELITE=12, ITERS=8,
            W_SETTLE=50.0, W_END=10.0, W_MAX=5.0, W_TAU=0.01, W_PEN=500.0,
            W_KE=0.0, SETTLE_T=1.0, FINAL_SETTLE_T=1.5)
CFG3 = dict(T_TRAJ=6.0, N_WP=8, DELTA=0.6, SIGMA0=0.12, N=96, ELITE=12, ITERS=8,
            W_SETTLE=50.0, W_END=10.0, W_MAX=5.0, W_TAU=0.01, W_PEN=500.0,
            W_KE=500.0, SETTLE_T=1.0, FINAL_SETTLE_T=1.5,
            KE_CAP=0.005, PEAK_REJECT=0.03, SHRINK_TRIES=3)


def cost_with_settle(plant, mev, w, sign, dst, cfg):
    snap = plant.snapshot()
    tr = rollout_block(plant, None, w, cfg["T_TRAJ"], ctrl_dt=0.02)
    if not tr["ok"]:
        plant.restore(snap)
        return INF, None
    if cfg.get("PEAK_REJECT") and tr["peak_ke"] > cfg["PEAK_REJECT"]:
        plant.restore(snap)
        return INF, None
    r21 = np.array(tr["R21"]) * sign
    plant.d.ctrl[:] = 0.0
    sub = 20
    for _ in range(int(cfg["SETTLE_T"] / 0.02)):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
        if not plant.finite():
            plant.restore(snap)
            return INF, None
    q = plant.d.qpos.copy()
    ok_pred, _, _ = mev.predicate(q, dst)
    ok_full, _, cert, _ = mev.full_check(q, dst)
    p_settle = float(sign * plant.base_R()[2, 1])
    tau = np.abs(np.array(tr["tau"])).mean()
    c = (-cfg["W_SETTLE"] * p_settle - cfg["W_END"] * r21[-1] - cfg["W_MAX"] * r21.max()
         + cfg["W_TAU"] * tau + cfg["W_PEN"] * tr["max_selfpen_m"]
         + cfg["W_KE"] * tr["peak_ke"])
    info = {"p_settle": p_settle, "pred": bool(ok_pred), "full": bool(ok_full),
            "rho": float(cert["rho"]), "peak_ke": float(tr["peak_ke"]),
            "ke_end": float(tr["ke_end"]), "p_max": float(r21.max()),
            "p_end": float(r21[-1])}
    plant.restore(snap)
    return c, info


def run_direct_start(tag, q_start, dst, sign, seed, cfg, step, outdir):
    t0 = time.time()
    plant = Plant(timestep=0.001)
    mev = ModeEval()
    plant.spawn(q_start)
    j = {n: i for i, n in enumerate(JOINTS)}
    hold_idx = [j[n] for n in HOLD_JOINTS[sign]]
    rec = {"tag": tag, "dst": dst, "sign": sign, "seed": seed, "step": step,
           "cfg": {k: v for k, v in cfg.items()}, "status": "RUNNING"}

    print(f"\n===== step{step} start {tag} -> {dst} (sign {sign:+d}, seed {seed}) =====", flush=True)
    # Phase 1
    tr1 = phase1_scripted(plant, sign, CFG2_BASE)
    rec["phase1"] = {"peak_ke_J": float(tr1["peak_ke"]), "ke_end_J": float(tr1["ke_end"])}
    print(f"[{tag}] phase1: peakKE={tr1['peak_ke']:.4f}J", flush=True)
    qpos_all = [np.array(tr1["qpos"])]
    qvel_all = [np.array(tr1["qvel"])]

    # Phase 2: CEM
    snap0 = plant.snapshot()
    q0 = plant.d.qpos[plant.qadr].copy()
    lo = np.maximum(plant.lo, q0 - cfg["DELTA"])
    hi = np.minimum(plant.hi, q0 + cfg["DELTA"])
    D = cfg["N_WP"] * 20
    rng = np.random.default_rng(seed)
    mu = np.zeros(D)
    sigma = np.full(D, cfg["SIGMA0"])
    best = {"cost": INF, "delta": None, "info": None}
    for it in range(cfg["ITERS"]):
        deltas = np.clip(rng.normal(mu, sigma, (cfg["N"], D)), -cfg["DELTA"], cfg["DELTA"])
        # elitism: keep the best so far in the population
        if best["delta"] is not None:
            deltas[0] = best["delta"]
        costs = np.empty(cfg["N"])
        infos = []
        for n in range(cfg["N"]):
            w = np.clip(q0 + deltas[n].reshape(cfg["N_WP"], 20), lo, hi)
            w[:, hold_idx] = q0[hold_idx][None, :]
            plant.restore(snap0)
            c, info = cost_with_settle(plant, mev, w, sign, dst, cfg)
            costs[n] = c
            infos.append(info)
            if c < best["cost"]:
                best = {"cost": float(c), "delta": deltas[n].copy(), "info": info}
        elite = np.argsort(costs)[:cfg["ELITE"]]
        mu = deltas[elite].mean(axis=0)
        sigma = np.clip(deltas[elite].std(axis=0), 0.03, 0.7)
        bi = best["info"]
        print(f"[{tag}] it{it}: best={best['cost']:+.2f} p_settle={bi['p_settle']:+.3f} "
              f"pred={bi['pred']} peakKE={bi['peak_ke']:.3f}J", flush=True)

    # Execute best; Step 3: enforce the KE cap (shrink on violation)
    w_best = np.clip(q0 + best["delta"].reshape(cfg["N_WP"], 20), lo, hi)
    w_best[:, hold_idx] = q0[hold_idx][None, :]
    cap_ok = True
    shrink = 0
    tr = None
    while True:
        plant.restore(snap0)
        tr = rollout_block(plant, None, w_best, cfg["T_TRAJ"], ctrl_dt=0.02)
        ke_end = tr["ke_end"]
        if step == 3 and ke_end > cfg["KE_CAP"] and shrink < cfg["SHRINK_TRIES"]:
            shrink += 1
            scale = 0.5
            w_best = q0 + (w_best - q0) * scale  # shrink toward q0
            w_best[:, hold_idx] = q0[hold_idx][None, :]
            print(f"[{tag}] KE cap violated (ke_end={ke_end:.4f}J) -> shrink x{scale} (try {shrink})",
                  flush=True)
            continue
        if step == 3 and ke_end > cfg["KE_CAP"]:
            cap_ok = False
        break
    rec["ke_shrinks"] = shrink
    rec["ke_cap_ok"] = bool(cap_ok)
    rec["cem_best"] = best["info"]
    qpos_all.append(np.array(tr["qpos"]))
    qvel_all.append(np.array(tr["qvel"]))

    # Final: 1.5-s zero-torque settle from trajectory end
    plant.d.ctrl[:] = 0.0
    sub = 20
    for _ in range(int(cfg["FINAL_SETTLE_T"] / 0.02)):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
    qf = plant.d.qpos.copy()
    ok_pred, why_pred, _ = mev.predicate(qf, dst)
    ok_full, why_full, cert, _ = mev.full_check(qf, dst)
    rec["final_settle"] = {"predicate_ok": bool(ok_pred), "predicate_why": why_pred,
                           "full_ok": bool(ok_full), "full_why": why_full,
                           "rho": float(cert["rho"]),
                           "cert_feasible": bool(cert["feasible"]),
                           "R21": float(sign * plant.base_R()[2, 1])}
    rec["phase2_exec"] = {"peak_ke_J": float(tr["peak_ke"]), "ke_end_J": float(tr["ke_end"]),
                          "max_selfpen_mm": float(tr["max_selfpen_m"] * 1e3)}
    success = bool(ok_pred) and (step == 2 or cap_ok)
    # Step 3 must ALSO satisfy the cap to count as success
    rec["status"] = "SUCCESS" if success else ("FAIL_KE_CAP" if not cap_ok else "FAIL_STALL")
    rec["wall_s"] = round(time.time() - t0, 1)
    print(f"[{tag}] {rec['status']}: pred={ok_pred} full={ok_full} "
          f"rho={cert['rho']:.3f} peakKE={tr['peak_ke']:.3f}J ke_end={tr['ke_end']:.4f}J "
          f"wall={rec['wall_s']:.0f}s", flush=True)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"step{step}_{tag}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.join(outdir, f"traj_step{step}_{tag}.npz"),
                        qpos=np.vstack(qpos_all), qvel=np.vstack(qvel_all),
                        waypoints=w_best)
    return rec


def _worker(args):
    try:
        return run_direct_start(*args)
    except Exception:
        import traceback
        return {"tag": args[0], "status": "ERROR", "error": traceback.format_exc(limit=5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[2, 3])
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = CFG2 if a.step == 2 else CFG3
    outdir = os.path.join(HERE, f"step{a.step}")

    mev = ModeEval()
    starts, verified = make_starts(mev)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "starts.json"), "w") as f:
        json.dump(verified, f, indent=1)

    tasks = [(tag, q, dst, sign, seed, cfg, a.step, outdir)
             for (tag, q, dst, sign, seed) in starts]
    if a.start is not None:
        tasks = [tasks[a.start]]
    elif not a.all:
        tasks = tasks[:1]

    if a.workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(f"{r['tag']}: {r['status']} wall={r.get('wall_s')}", flush=True)


if __name__ == "__main__":
    main()
