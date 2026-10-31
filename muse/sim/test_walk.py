import sys, time
import numpy as np
from model import RealRobot, BodyParams, JOINTS
from wbc import WBC, BodyModel, GaitParams
import json, os

def run(true_p=BodyParams(), model_p=BodyParams(), gp=GaitParams(), stride=0.0, nsteps=8, T_stand=1.0, verbose=True, seed=0):
    rob = RealRobot(true_p, seed=seed)
    bm = BodyModel(model_p)
    z = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "ref_walk.npz"))
    wbc = WBC(bm, gp, mu_ctrl=0.6, ref=z["q_cycle"])
    o = rob.observe()
    t0 = time.time(); walking = False; log = []
    tend = T_stand + 0.6 + nsteps * gp.T_step + 1.0
    while rob.t < tend:
        t = rob.t
        if not walking and t >= T_stand:
            bm.load_state(o)
            from wbc import SOLE_PT
            fp = {s: bm.point(b, SOLE_PT)[:2] for s, b in wbc.feet.items()}
            com = bm.d.subtree_com[0][:2].copy()
            wbc.planner.start_walk(t, (fp["l"], fp["r"]), com, nsteps, stride)
            wbc.ref_on = True
            walking = True
        tau, info = wbc.control(o, t)
        o = rob.step(tau)
        log.append((t, *rob.d.qpos[:3], info["tau_ratio"], info["fric_ratio"], len(info["stance"])))
        if rob.fallen():
            if verbose: print("FELL at", round(t, 3))
            break
    wall = time.time() - t0
    L = np.array(log)
    if verbose:
        print(f"sim {rob.t:.2f}s wall {wall:.1f}s qp_fail {wbc.stats['qp_fail']}/{wbc.stats['n']} final base {np.round(rob.d.qpos[:3],3)} "
              f"max tau ratio {L[:,4].max():.2f} max fric {L[:,5].max():.2f}")
    return rob, wbc, L

if __name__ == "__main__":
    stride = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(stride=stride, nsteps=n)
