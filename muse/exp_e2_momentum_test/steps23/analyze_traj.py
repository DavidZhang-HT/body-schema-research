#!/usr/bin/env python
"""Analyze the successful direct trajectory: what moves, when does it tip?"""
import sys
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23")
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/sim")
import numpy as np
from model import JOINTS

tr = np.load("/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/steps23/step2/traj_direct_clean_R_a.npz")
qpos, qvel, wp = tr["qpos"], tr["qvel"], tr["waypoints"]
# qpos[:,7:] are joints? No: qpos is (27,): [0:3] pos, [3:7] quat, [7:] joints? 
# Actually qadr maps JOINTS->qpos indices. For this model, free joint is first (7), then joints.
# Let me just use indices 7:27 as joints in JOINTS order (verify via qadr monotonic).
print("n steps:", len(qpos), "dt=0.02 ->", len(qpos)*0.02, "s")
jq = qpos[:, 7:27]
# range of motion per joint
rom = jq.max(axis=0) - jq.min(axis=0)
order = np.argsort(rom)[::-1]
print("top joints by range of motion:")
for i in order[:10]:
    print(f"  {JOINTS[i]:28s} rom={rom[i]:.3f} rad  q0={jq[0,i]:+.3f} -> qend={jq[-1,i]:+.3f}")
# base orientation trace (R21) - need base quat -> R. Approx from qpos[3:7].
import mujoco
R21 = []
for k in range(len(qpos)):
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, qpos[k, 3:7])
    R21.append(Rm.reshape(3,3)[2,1])
R21 = np.array(R21)
print("R21 at t=0,1,2,3,4s:", [f"{R21[int(t/0.02)]:+.3f}" for t in (0,1,2,3,3.98)])
# when does it cross 0.7?
cross = np.where(R21 > 0.7)[0]
print("first cross 0.7 at t=%.2fs" % (cross[0]*0.02) if len(cross) else "never")
# peak joint speeds
jsp = np.abs(qvel[:, 6:26]).max(axis=0)
print("peak joint speeds (top):")
for i in np.argsort(jsp)[::-1][:6]:
    print(f"  {JOINTS[i]:28s} {jsp[i]:.2f} rad/s")
