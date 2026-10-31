#!/usr/bin/env python
"""Direct trajectory optimization for the roll: CEM over a 4s waypoint program,
cost = post-settle SIDE predicate (the true objective).  From post-Phase-1 state.
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start, rollout_block
from step2 import phase1_scripted, CFG, HOLD_JOINTS, INF
from model import JOINTS

T_TRAJ = 4.0
N_WP = 6
N = 96
ELITE = 12
ITERS = 6
SIGMA0 = 0.6
SETTLE_T = 1.0


def cost_with_settle(plant, mev, w, sign, dst):
    snap = plant.snapshot()
    tr = rollout_block(plant, None, w, T_TRAJ, ctrl_dt=0.02)
    if not tr["ok"]:
        plant.restore(snap)
        return INF, None
    r21 = np.array(tr["R21"]) * sign
    # settle (blind, no logging)
    plant.d.ctrl[:] = 0.0
    sub = 20
    for _ in range(int(SETTLE_T / 0.02)):
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
    pen = tr["max_selfpen_m"]
    c = (-50.0 * p_settle - 10.0 * r21[-1] - 5.0 * r21.max()
         + 0.01 * tau + 500.0 * pen)
    plant.restore(snap)
    return c, {"p_settle": p_settle, "pred": ok_pred, "full": ok_full,
               "rho": float(cert["rho"]), "peak_ke": tr["peak_ke"],
               "p_max": float(r21.max())}


def main():
    plant = Plant(timestep=0.001)
    mev = ModeEval()
    plant.spawn(load_clean_start())
    sign, dst = +1, "SIDE"
    tr1 = phase1_scripted(plant, sign, CFG)
    print(f"phase1 done: peakKE={tr1['peak_ke']:.4f}J", flush=True)
    j = {n: i for i, n in enumerate(JOINTS)}
    hold_idx = [j[n] for n in HOLD_JOINTS[sign]]

    snap0 = plant.snapshot()
    q0 = plant.d.qpos[plant.qadr].copy()
    lo = np.maximum(plant.lo, q0 - 1.0)
    hi = np.minimum(plant.hi, q0 + 1.0)
    D = N_WP * 20
    rng = np.random.default_rng(11)
    mu = np.zeros(D)
    sigma = np.full(D, SIGMA0)
    best = {"cost": INF, "delta": None, "info": None}
    for it in range(ITERS):
        deltas = np.clip(rng.normal(mu, sigma, (N, D)), -1.0, 1.0)
        costs = np.empty(N)
        infos = []
        for n in range(N):
            w = np.clip(q0 + deltas[n].reshape(N_WP, 20), lo, hi)
            w[:, hold_idx] = q0[hold_idx][None, :]
            plant.restore(snap0)
            c, info = cost_with_settle(plant, mev, w, sign, dst)
            costs[n] = c
            infos.append(info)
            if c < best["cost"]:
                best = {"cost": float(c), "delta": deltas[n].copy(), "info": info}
        elite = np.argsort(costs)[:ELITE]
        mu = deltas[elite].mean(axis=0)
        sigma = np.clip(deltas[elite].std(axis=0), 0.05, 0.7)
        bi = infos[int(np.argmin(costs))]
        print(f"it{it}: best={costs.min():+.2f} p_settle={bi['p_settle']:+.3f} "
              f"pred={bi['pred']} peakKE={bi['peak_ke']:.2f}J", flush=True)
    print("BEST:", best["cost"], best["info"])
    # execute best for real and save
    w_best = np.clip(q0 + best["delta"].reshape(N_WP, 20), lo, hi)
    w_best[:, hold_idx] = q0[hold_idx][None, :]
    plant.restore(snap0)
    tr = rollout_block(plant, None, w_best, T_TRAJ, ctrl_dt=0.02)
    np.savez_compressed("/tmp/direct_traj.npz", qpos=np.array(tr["qpos"]),
                        qvel=np.array(tr["qvel"]), waypoints=w_best)
    # final settle check
    plant.d.ctrl[:] = 0.0
    sub = 20
    for _ in range(int(1.5 / 0.02)):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
    ok, why, _ = mev.predicate(plant.d.qpos.copy(), dst)
    okf, whyf, cert, _ = mev.full_check(plant.d.qpos.copy(), dst)
    print(f"FINAL settle: pred={ok} full={okf} rho={cert['rho']:.3f} R21={plant.base_R()[2,1]:+.3f}")
    if not ok:
        print("why:", why)


if __name__ == "__main__":
    main()
