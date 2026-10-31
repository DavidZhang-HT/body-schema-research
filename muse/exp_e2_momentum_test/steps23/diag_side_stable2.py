#!/usr/bin/env python
"""Gentle placement at 90/100 deg roll (no drop): is SIDE a stable rest pose?"""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start

plant = Plant(timestep=0.001)
mev = ModeEval()
q0 = load_clean_start()

for deg in (90, 100):
    plant.spawn(q0)
    th = np.radians(deg)
    qx = np.array([np.cos(th / 2), np.sin(th / 2), 0, 0])
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, qx, q0[3:7])
    # gentle: use reset_pose-like placement (lowest shell point at z=0)
    plant.r.reset_pose(q0[plant.qadr], qq, plant.d)
    plant.d.qvel[:] = 0
    plant.d.ctrl[:] = 0.0
    R_init = plant.base_R()
    print(f"place {deg}deg: initial R21={R_init[2,1]:+.3f} chest_z={R_init[2,0]:+.3f} base_z={plant.d.qpos[2]:.3f}")
    sub = 20
    for s in range(150):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
        if (s + 1) % 30 == 0:
            R = plant.base_R()
            print(f"   t={(s+1)*0.02:4.2f}s R21={R[2,1]:+.3f} chest_z={R[2,0]:+.3f} "
                  f"base_z={plant.d.qpos[2]:.3f} KE={plant.kinetic_energy():.4f}")
    R = plant.base_R()
    ok, why, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
    okf, whyf, cert, _ = mev.full_check(plant.d.qpos.copy(), "SIDE")
    print(f"   FINAL pred={ok} full={okf} rho={cert['rho']:.3f}")
    if not ok:
        print(f"   why: {why}")
