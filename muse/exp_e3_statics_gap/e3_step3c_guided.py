"""E3 step 3c (diagnostic arm, post-hoc, NOT in the pre-registered verdict rule): certificate-guided
execution.  From the plateau snapshot, command B's own PD/lag/torque-envelope actuation (GroundWorld.advance,
unchanged) with a FULL 24-D joint-space basis (B := I, BOUNDS := 3 rad) toward the step-2 tier-A
certified feet-only equilibrium, linearly interpolated over K blocks of 0.6 s, then 3 zero-action
hold blocks.  Tests whether the certified target is physically reachable through B's actuation when the
action basis is not a restriction.  Control: B's nominal standing pose held with zero action
(does the PD hold itself keep a standing robot up?).
"""
import json
import numpy as np
from pathlib import Path
import mujoco as mj
import e3_plant
from e3_plant import GroundWorld, score
import ground_schema as gs
from e3_step3_search import handover, strict

OUT = Path(__file__).resolve().parent / 'results'


def main():
    gs.B = np.eye(24); gs.BOUNDS = np.full(24, 3.0)
    w = GroundWorld(); res = {}
    # control: standing + zero-action hold
    import experiment
    mj.mj_resetData(w.m, w.d); w.d.qpos[:] = experiment.standing(w.m); w.actual[:] = 0; mj.mj_forward(w.m, w.d)
    hist = []
    for k in range(5):
        w.advance(np.zeros(24)); hist.append(w.metrics())
    res['control_standing_hold'] = dict(history=hist, handover_all=all(handover(r) for r in hist))
    print('control standing hold: handover each block', [handover(r) for r in hist], 'base', [round(r['base_height'], 3) for r in hist])
    snap0 = np.load(OUT / 'plateau_snapshot.npy')
    targets = {}
    for nm in ('feet_only_eq_tierA',):
        f = OUT / f'{nm}.npy'
        if f.exists(): targets[nm] = np.load(f)
    targets['nominal_standing'] = experiment.standing(w.m)
    for tn, qt in targets.items():
        for K in (1, 2, 4, 8):
            w.restore(snap0); q_start = w.d.qpos[7:].copy(); hist = []
            for k in range(1, K + 1):
                tgt = q_start + (qt[7:] - q_start) * k / K
                w.advance(tgt - w.d.qpos[7:]); r = w.metrics(); r['score'] = float(score(w)); hist.append(r)
            for k in range(3):
                w.advance(np.zeros(24)); r = w.metrics(); r['score'] = float(score(w)); hist.append(r)
            succ = all(handover(r) for r in hist[-3:])
            res[f'{tn}_K{K}'] = dict(history=hist, success_hold=succ, strict_end=strict(hist[-1]), final_qpos=w.d.qpos.tolist())
            print(f'{tn} K={K}: success(handover held 3 blocks)={succ} end base {hist[-1]["base_height"]:.3f} up {hist[-1]["up_z"]:.3f} '
                  f'feet {hist[-1]["foot_load"]:.1f} nonfoot {hist[-1]["nonfoot_load"]:.1f} score {hist[-1]["score"]:.3f} '
                  f'max score {max(h["score"] for h in hist):.3f}', flush=True)
    (OUT / 'step3c_guided.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
