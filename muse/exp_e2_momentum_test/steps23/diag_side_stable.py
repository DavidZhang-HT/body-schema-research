#!/usr/bin/env python
"""Is SIDE (90 deg roll) a stable resting pose? Drop test at 90/100/110 deg roll."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start

plant = Plant(timestep=0.001)
mev = ModeEval()
q0 = load_clean_start()

for deg in (80, 90, 100, 110):
    plant.spawn(q0)
    # premultiply base quat by world-frame Rx(deg): q' = qx \otimes q
    th = np.radians(deg)
    qx = np.array([np.cos(th / 2), np.sin(th / 2), 0, 0])
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, qx, q0[3:7])
    plant.d.qpos[3:7] = qq
    plant.d.qpos[2] = 0.30  # drop from 0.3 m
    plant.d.qvel[:] = 0
    mujoco.mj_forward(plant.m, plant.d)
    plant.d.ctrl[:] = 0.0
    sub = 20
    for s in range(150):  # 3 s
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
    R = plant.base_R()
    ok, why, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
    okf, whyf, cert, _ = mev.full_check(plant.d.qpos.copy(), "SIDE")
    print(f"drop {deg:3d}deg -> R21={R[2,1]:+.3f} chest_z={R[2,0]:+.3f} "
          f"base_z={plant.d.qpos[2]:.3f} KE={plant.kinetic_energy():.4f} "
          f"pred={ok} full={okf} rho={cert['rho']:.3f}")
    if not ok:
        print(f"    why: {why}")
