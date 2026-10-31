#!/usr/bin/env python
"""Diagnose left-arm reach: for a grid of (shoulder_roll, elbow), print hand world pos via FK."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, load_clean_start
from model import JOINTS

plant = Plant(timestep=0.001)
q0 = load_clean_start()
plant.spawn(q0)
j = {n: i for i, n in enumerate(JOINTS)}
m = plant.m
# hand: use the elbow link body origin + the "hand" sphere offset (0,0,-0.110 in elbow frame)
elbow_bid = m.body("l_elbow_link").id

qj0 = plant.d.qpos[plant.qadr].copy()
print(f"{'sr':>6} {'elb':>6} | {'hand_x':>7} {'hand_y':>7} {'hand_z':>7} | R21")
for sr in [0.0, 0.5, 1.0, 1.3, 1.6, 2.0, 2.5]:
    for elb in [0.0, 0.6, 1.2]:
        plant.spawn(q0)
        qj = qj0.copy()
        qj[j["l_shoulder_roll_joint"]] = sr
        qj[j["l_elbow_joint"]] = elb
        plant.d.qpos[plant.qadr] = qj
        mujoco.mj_forward(m, plant.d)
        R = plant.d.xmat[elbow_bid].reshape(3, 3)
        hp = plant.d.xpos[elbow_bid] + R @ np.array([0, 0, -0.110])
        r21 = plant.base_R()[2, 1]
        print(f"{sr:6.1f} {elb:6.1f} | {hp[0]:7.3f} {hp[1]:7.3f} {hp[2]:7.3f} | {r21:+.3f}")
