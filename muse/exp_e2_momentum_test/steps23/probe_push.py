#!/usr/bin/env python
"""Scripted probe: can a hand-designed left-arm push roll the supine pi+ to SIDE?

Phase 1 (0-1s): abduct left arm (shoulder_roll -> +/-1.3).
Phase 2 (1-2.5s): press into ground (shoulder_roll -> +/-1.7, elbow adjusts).
Hold to 4s.  Records R[2,1] (SIDE needs >= 0.7).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from common import Plant, ModeEval, load_clean_start, rollout_block
from model import JOINTS

VENV = "/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python"


def run_trial(sign, T1=1.0, T2=1.5, T3=1.5, abduct=1.3, press=1.7):
    plant = Plant(timestep=0.001)
    q0 = load_clean_start()
    plant.spawn(q0)
    j = {n: i for i, n in enumerate(JOINTS)}
    q_start = plant.d.qpos[plant.qadr].copy()

    def wp(**kw):
        w = q_start.copy()
        for k, v in kw.items():
            w[j[k]] = v
        return w

    # phase 1: abduct
    w1 = wp(l_shoulder_roll_joint=q_start[j["l_shoulder_roll_joint"]] + sign * abduct,
            l_elbow_joint=0.6)
    # phase 2: press down into the ground
    w2 = wp(l_shoulder_roll_joint=q_start[j["l_shoulder_roll_joint"]] + sign * press,
            l_elbow_joint=0.25,
            l_shoulder_pitch_joint=q_start[j["l_shoulder_pitch_joint"]] - 0.3)
    # phase 3: hold press
    traces = []
    for T, w in ((T1, w1), (T2, w2), (T3, w2)):
        tr = rollout_block(plant, None, np.array([w]), T, ctrl_dt=0.02)
        traces.append(tr)
    R = plant.base_R()
    return {
        "sign": sign,
        "R21_end": float(R[2, 1]),
        "R21_trace": [x for tr in traces for x in tr["R21"]],
        "peak_ke": max(tr["peak_ke"] for tr in traces),
        "max_selfpen_mm": max(tr["max_selfpen_m"] for tr in traces) * 1e3,
        "base_z_end": float(plant.d.qpos[2]),
    }


def main():
    mev = ModeEval()
    ok, why, cert, _ = mev.full_check(load_clean_start(), "SUPINE")
    print(f"start SUPINE check: ok={ok} rho={cert['rho']:.4f}", flush=True)
    for sign in (+1.0, -1.0):
        t0 = time.time()
        r = run_trial(sign)
        print(f"sign {sign:+.0f}: R21_end={r['R21_end']:+.3f} "
              f"peak_KE={r['peak_ke']:.4f}J selfpen={r['max_selfpen_mm']:.1f}mm "
              f"base_z={r['base_z_end']:.3f} wall={time.time()-t0:.1f}s", flush=True)
        tr = r["R21_trace"]
        print(f"   R21 trace (every 10th): {[f'{x:+.2f}' for x in tr[::10]]}", flush=True)


if __name__ == "__main__":
    main()
