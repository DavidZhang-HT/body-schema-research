#!/usr/bin/env python
"""Smoke: phase1 (arm forward) + one CEM block + settle, start clean_R_a."""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
import numpy as np
from common import Plant, ModeEval, load_clean_start
from step2 import phase1_scripted, cem_block, settle, CFG, HOLD_JOINTS
from model import JOINTS

plant = Plant(timestep=0.001)
mev = ModeEval()
plant.spawn(load_clean_start())
tr1 = phase1_scripted(plant, +1, CFG)
print(f"phase1: peakKE={tr1['peak_ke']:.4f}J ke_end={tr1['ke_end']:.5f}J")
j = {n: i for i, n in enumerate(JOINTS)}
hold_idx = [j[n] for n in HOLD_JOINTS[+1]]
w, trace, info = cem_block(plant, mev, +1, 11 * 1000, CFG, log_prefix="[smoke]", hold_idx=hold_idx)
r21 = np.array(trace["R21"])
print(f"block: peakKE={trace['peak_ke']:.3f}J roll end/max={r21[-1]:+.3f}/{r21.max():+.3f}")
ok, why, _ = mev.predicate(plant.d.qpos.copy(), "SIDE")
print("predicate:", ok)
if ok:
    s, snap = settle(plant, mev, "SIDE", CFG["SETTLE_T"])
    print("settle:", s["predicate_ok"], "full:", s["full_ok"], "rho:", round(s["rho"], 3))
