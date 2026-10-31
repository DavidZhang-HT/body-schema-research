"""G0 probe: drop pi+ supine on the full contact shell, check that multi-link ground
contacts form, and that the per-link contact loads are readable through the torque channel.

Torque-channel read-out (generalisation of WBC.estimate_grf / TorqueChannel to arbitrary sigma):
    J_c(sigma, q)^T F = M(q) qdd + h(q, qd) - S^T tau_meas
with F = [6-D wrench of the trunk (rigid shell, several points -> net wrench only, probe 2 set G)]
       + [3-D force for every other contacting link (point contact)].
Uses only *measured* quantities (noisy encoders, motor-current torque, estimator base pose,
IMU quat) and the nominal internal model; compared against MuJoCo ground-truth contact forces.
Deterministic (seeded noise).  Run:  sim/.venv/bin/python sim/getup_g0_probe.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PiPlus, RealRobot, BodyParams, JOINTS, ROOT, getup_state, GetupMonitor  # noqa: E402


def truth_link_forces(rob):
    """World-frame net contact force per body + contact points (ground truth)."""
    m, d = rob.m, rob.d
    out = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if rob.r.floor_gid not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == rob.r.floor_gid else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]          # contact frame -> world (force on geom2)
        if c.geom1 != rob.r.floor_gid:                 # force on the robot geom
            fw = -fw
        b = m.body(m.geom_bodyid[g]).name
        e = out.setdefault(b, {"F": np.zeros(3), "M": np.zeros(3), "pts": []})
        e["F"] += fw
        e["pts"].append((c.pos.copy(), fw))
    return out


def torque_channel_readout(r: PiPlus, obs, links, ref_pts):
    """Least-squares contact loads from measured torques on the internal (nominal) model."""
    m, d = r.m, mujoco.MjData(r.m)
    d.qpos[0:3] = obs.base_pos
    d.qpos[3:7] = obs.base_quat
    d.qpos[r.qadr] = obs.q
    d.qvel[r.vadr] = obs.qd
    R = np.zeros(9); mujoco.mju_quat2Mat(R, obs.base_quat)
    d.qvel[3:6] = obs.gyro
    d.qvel[0:3] = obs.base_linvel
    mujoco.mj_forward(m, d)
    h = d.qfrc_bias.copy()                    # quasi-static: qdd ~ 0 (rest)
    rhs = h.copy()
    rhs[r.vadr] -= obs.tau
    cols, layout = [], []
    for b in links:
        bid = m.body(b).id
        p = ref_pts[b]
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, p, bid)
        if b == "base_link":
            cols.append(np.vstack([jp, jr]).T); layout.append((b, 6))
        else:
            cols.append(jp.T); layout.append((b, 3))
    A = np.hstack(cols)
    sv = np.linalg.svd(A, compute_uv=False)
    F, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    res = A @ F - rhs
    out, k = {}, 0
    for b, n in layout:
        out[b] = F[k:k + n]; k += n
    return out, dict(rank=int((sv > 1e-6 * sv[0]).sum()), n=A.shape[1], cond=float(sv[0] / sv[-1]),
                     resid=float(np.linalg.norm(res)))


def J(name):
    return JOINTS.index(name)


def scenarios(r):
    """Drop postures (qj, base quat wxyz).  All deterministic."""
    sq = r.supine_q()
    sup = np.array([np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0])     # chest up
    pro = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])       # chest down
    out = {"supine": (sq, sup)}
    q = sq.copy()                                                       # arms abducted on floor
    q[J("l_shoulder_roll_joint")], q[J("r_shoulder_roll_joint")] = 1.4, -1.4
    q[J("l_elbow_joint")] = q[J("r_elbow_joint")] = 0.0
    out["supine_arms_out"] = (q, sup)
    q = sq.copy()                                                       # long-sit, hands behind
    for s_ in "lr":
        q[J(f"{s_}_hip_pitch_joint")] = -1.45
        q[J(f"{s_}_shoulder_pitch_joint")] = 0.6
    out["long_sit"] = (q, np.array([1.0, 0, 0, 0]))
    q = np.zeros(20)                                                    # prone hands-and-knees
    for s_ in "lr":
        q[J(f"{s_}_hip_pitch_joint")] = -1.57
        q[J(f"{s_}_calf_joint")] = 1.57
        q[J(f"{s_}_shoulder_pitch_joint")] = -1.57
    out["prone_hands_feet"] = (q, pro)
    q = q.copy()                                                        # prone kneel (knee shell)
    for s_ in "lr":
        q[J(f"{s_}_hip_pitch_joint")] = -1.2
        q[J(f"{s_}_ankle_pitch_joint")] = 0.9
    out["prone_kneel"] = (q, pro)
    return out


def run_scenario(name, qj, quat, shell="full"):
    p = BodyParams()
    rob = RealRobot(p, seed=0, shell=shell)
    rob.reset_supine(qj, quat)
    r = rob.r
    mon = GetupMonitor(timeout=1e9)
    kp, kd = 15.0, 0.5
    for i in range(1500):                     # settle 3 s under a soft PD hold
        o = rob.observe()
        rob.step(kp * (qj - o.q) - kd * o.qd)
        mon.update(rob)
    truth = truth_link_forces(rob)
    log = dict(scenario=name, shell=shell, state=getup_state(r, rob.d, rob.contact_links()),
               monitor=mon.status, base_z=float(rob.d.qpos[2]), contact_bodies=sorted(truth),
               n_contacts=int(rob.d.ncon))
    links = sorted(truth, key=lambda b: (b != "base_link", b))
    if not links:
        return log
    ref = {b: (np.mean([pp for pp, _ in truth[b]["pts"]], axis=0) if b != "base_link"
               else rob.d.xpos[r.base].copy()) for b in links}
    ests = []
    for i in range(100):                      # 0.2 s of measurements, averaged
        o = rob.observe()
        o = rob.step(kp * (qj - o.q) - kd * o.qd)
        est, info = torque_channel_readout(r, o, links, ref)
        ests.append(est)
    truth = truth_link_forces(rob)
    mg = r.total_mass0 * 9.81
    rows = []
    for b in links:
        e = np.mean([x[b] for x in ests], axis=0)
        t = truth[b]["F"] if b in truth else np.zeros(3)
        rows.append(dict(link=b, est_F=e[:3].round(2).tolist(), true_F=t.round(2).tolist(),
                         err_N=round(float(np.linalg.norm(e[:3] - t)), 2)))
    log.update(readout=rows, readout_info=info,
               sum_true_Fz_over_mg=round(float(sum(truth[b]["F"][2] for b in truth) / mg), 4),
               max_err_over_mg=round(max(x["err_N"] for x in rows) / mg, 4))
    return log


def main():
    r = PiPlus(shell="full")
    logs = []
    for name, (q, quat) in scenarios(r).items():
        logs.append(run_scenario(name, q, quat))
    # legacy shell, same hands-and-knees drop: limbs cannot touch (the G0 defect)
    q, quat = scenarios(r)["prone_kneel"]
    logs.append(run_scenario("prone_kneel", q, quat, shell="feet"))
    for L in logs:
        print(f"[{L['scenario']:18s} shell={L['shell']:4s}] state={L['state']:10s} z={L['base_z']:.3f} "
              f"ncon={L['n_contacts']:2d} bodies={L['contact_bodies']}")
        for x in L.get("readout", []):
            print(f"     {x['link']:22s} est={x['est_F']}  true={x['true_F']}  err={x['err_N']} N")
        if "readout_info" in L:
            print("     rank %d/%d cond %.1f  max err/mg %.3f" % (L["readout_info"]["rank"], L["readout_info"]["n"],
                                                             L["readout_info"]["cond"], L["max_err_over_mg"]))
    os.makedirs(os.path.join(ROOT, "results", "f3"), exist_ok=True)
    json.dump(logs, open(os.path.join(ROOT, "results", "f3", "g0_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
