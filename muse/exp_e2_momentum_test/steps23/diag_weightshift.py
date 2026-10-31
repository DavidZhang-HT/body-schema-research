#!/usr/bin/env python
"""Sanity: hand-designed SLOW weight-shift. Can quasi-static CoM shifting tip it?"""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
from common import Plant, ModeEval, load_clean_start, rollout_block
from step2 import phase1_scripted, CFG as CFG2_BASE
from model import JOINTS

plant = Plant(timestep=0.001)
mev = ModeEval()
plant.spawn(load_clean_start())
tr1 = phase1_scripted(plant, +1, CFG2_BASE)
print(f"phase1 done, KE={tr1['peak_ke']:.4f}J")

j = {n: i for i, n in enumerate(JOINTS)}
q0 = plant.d.qpos[plant.qadr].copy()
# slow weight shift: move limb mass to the right (-y body? No: right = -y? Body +y=left, so right=-y.
# To shift CoM right, move limbs toward -y.)
# Legs: hip_roll negative swings legs toward -y (right) for... left leg range [-0.31,3.14],
# right leg [-3.14,0.31]. Both can go negative (toward right).
q_tgt = q0.copy()
q_tgt[j["l_hip_roll_joint"]] = -0.30   # max adduction (toward right)
q_tgt[j["r_hip_roll_joint"]] = -1.20   # abduct right leg out to the right
q_tgt[j["l_shoulder_roll_joint"]] = -0.30  # left arm toward right (adduct)
q_tgt[j["l_shoulder_pitch_joint"]] = -1.0  # left arm across body
q_tgt = np.clip(q_tgt, plant.lo, plant.hi)
tr = rollout_block(plant, None, np.array([q_tgt]), 8.0, ctrl_dt=0.02)
R21 = np.array(tr["R21"])
com = np.array(tr["com"])
print(f"weight-shift: peakKE={tr['peak_ke']:.4f}J")
print(f"  R21: start {R21[0]:+.3f} -> end {R21[-1]:+.3f}, max {R21.max():+.3f}")
print(f"  CoM y: start {com[0,1]:+.3f} -> end {com[-1,1]:+.3f} (body +y=left; want -y=right)")
print(f"  CoM z: start {com[0,2]:+.3f} -> end {com[-1,2]:+.3f}")
ok, why, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
print(f"  predicate: {ok}")
