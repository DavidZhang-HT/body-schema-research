#!/usr/bin/env python
"""Search for a STABLE side-lying pose: 90deg placement x arm configurations."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start
from model import JOINTS

plant = Plant(timestep=0.001)
mev = ModeEval()
q0 = load_clean_start()
j = {n: i for i, n in enumerate(JOINTS)}
qj0 = q0[plant.qadr].copy()

configs = {
    "arms_across_chest": {"l_shoulder_pitch_joint": -1.5, "r_shoulder_pitch_joint": -1.5,
                          "l_elbow_joint": 0.8, "r_elbow_joint": 0.8},
    "arms_up": {"l_shoulder_pitch_joint": -2.5, "r_shoulder_pitch_joint": -2.5},
    "R_arm_forward": {"r_shoulder_pitch_joint": -1.5, "r_elbow_joint": 0.8},
    "arms_tuck": {"l_shoulder_roll_joint": 0.5, "r_shoulder_roll_joint": -0.5,
                  "l_elbow_joint": 1.5, "r_elbow_joint": 1.5},
}
for name, kw in configs.items():
    qj = qj0.copy()
    for k, v in kw.items():
        qj[j[k]] = v
    q = q0.copy()
    q[plant.qadr] = qj
    th = np.radians(90)
    qx = np.array([np.cos(th / 2), np.sin(th / 2), 0, 0])
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, qx, q[3:7])
    plant.r.reset_pose(qj, qq, plant.d)
    plant.d.qvel[:] = 0
    plant.d.ctrl[:] = 0.0
    sub = 20
    for s in range(150):
        for _ in range(sub):
            mujoco.mj_step(plant.m, plant.d)
    R = plant.base_R()
    ok, why, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
    print(f"{name:20s} -> R21={R[2,1]:+.3f} chest_z={R[2,0]:+.3f} "
          f"base_z={plant.d.qpos[2]:.3f} pred={ok}")
    if not ok:
        print(f"    why: {why[0][:100]}")
