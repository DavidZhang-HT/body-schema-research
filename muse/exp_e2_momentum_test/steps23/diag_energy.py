#!/usr/bin/env python
"""Map the energy landscape: CoM height vs roll angle (arm-forward config)."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start
from model import JOINTS

plant = Plant(timestep=0.001)
q0 = load_clean_start()
j = {n: i for i, n in enumerate(JOINTS)}
# arm-forward config
qj = q0[plant.qadr].copy()
qj[j["r_shoulder_pitch_joint"]] = -1.5
qj[j["r_elbow_joint"]] = 0.8
q = q0.copy(); q[plant.qadr] = qj

print(f"{'deg':>5} {'com_z':>8} {'base_z':>8} (quasi-static placement, arm forward)")
for deg in (0, 30, 45, 60, 75, 80, 85, 90, 95, 100, 110):
    th = np.radians(deg)
    qx = np.array([np.cos(th / 2), np.sin(th / 2), 0, 0])
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, qx, q[3:7])
    plant.r.reset_pose(qj, qq, plant.d)
    plant.d.qvel[:] = 0
    mujoco.mj_forward(plant.m, plant.d)
    com = plant.com()
    print(f"{deg:5d} {com[2]:8.4f} {plant.d.qpos[2]:8.4f}")
