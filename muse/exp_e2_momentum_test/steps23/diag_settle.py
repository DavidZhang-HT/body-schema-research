#!/usr/bin/env python
"""Reproduce one CEM block-0 (start clean_R_a, seed 11), save end state, instrument settle."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
import mujoco
from common import Plant, ModeEval, load_clean_start
from step2 import cem_block, CFG

plant = Plant(timestep=0.001)
mev = ModeEval()
plant.spawn(load_clean_start())
w, trace, info = cem_block(plant, mev, +1, 11 * 1000 + 0, CFG, log_prefix="[repro]")
print(f"end roll={(np.array(trace['R21'])).max():+.3f} peakKE={trace['peak_ke']:.3f}J")
q_end = plant.d.qpos.copy()
qv_end = plant.d.qvel.copy()
ok, why, _ = mev.predicate(q_end, "SIDE")
print("predicate at block end:", ok, why)

# instrumented settle: continue from end state WITH its velocity, zero torque
plant.d.ctrl[:] = 0.0
sub = 20
print(f"{'t':>5} {'R21':>7} {'chest_z':>8} {'head_z':>8} {'base_z':>7} {'KE':>8} pred?")
for s in range(200):
    for _ in range(sub):
        mujoco.mj_step(plant.m, plant.d)
    if s % 10 == 0:
        R = plant.base_R()
        okp, whyp, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
        print(f"{s*0.02:5.2f} {R[2,1]:+7.3f} {R[2,0]:+8.3f} {R[2,2]:+8.3f} "
              f"{plant.d.qpos[2]:7.3f} {plant.kinetic_energy():8.4f} {okp}")
        if s == 0:
            print("   why:", whyp)
np.save("/tmp/b0_end.npy", np.concatenate([q_end, qv_end]))
