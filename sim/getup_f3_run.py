"""F3 full run: supine -> standing quasi-static keyframe search with the attention MPC.

Usage:  python getup_f3_run.py TAG        (writes results/f3/run_TAG.npz + run_TAG.json)

Deterministic: no randomness anywhere (dense quadprog QP, HiGHS LP, single-threaded BLAS is
forced by the caller via OMP/OPENBLAS/MKL_NUM_THREADS=1).  Every MPC tick logs the three
attention heads (support a_cur/a_des over 18 contact regions, motion a_joint over 20 joints,
effort alpha joint marginal), the certified statics (rho, binding joint, loads per region) and q.
Keyframes = the configuration at the end of each phase (imagined sub-goal).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from getup_schema import Schema, REGIONS
from getup_attention import SchemaAttention
from getup_mpc import QSMPC, MPCParams, make_phases, run_phase, certify
from model import JOINTS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "f3")


def jl(x):
    if isinstance(x, dict):
        return {str(k): jl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jl(v) for v in x]
    if isinstance(x, np.ndarray):
        return jl(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main(tag):
    S = Schema()
    A = SchemaAttention(S)
    p = MPCParams()
    mpc = QSMPC(S, A, p)
    q = np.load(os.path.join(ROOT, "q_supine_settled.npy"))
    phases = make_phases(S)
    logs = []
    t_wall = time.time()
    cert0 = certify(S, q, p.mu)
    keyframes = [dict(name="K0_supine", q=q.copy(), cert=cert0, outcome="start", ticks=0, t=0.0)]
    verdict = "FOUND"
    fail = None
    for ph in phases:
        n0 = len(logs)
        tp = time.time()
        q, out = run_phase(S, mpc, q, ph, p.mu, logs)
        seg = logs[n0:]
        rhos = [L["cert"]["rho"] for L in seg if L.get("cert") and np.isfinite(L["cert"]["rho"])]
        peak = int(np.argmax(rhos)) if rhos else None
        kf = dict(name=f"K_{ph.name}", q=q.copy(), cert=certify(S, q, p.mu), outcome=out,
                  ticks=len(seg), t=time.time() - tp,
                  phase_peak_rho=(max(rhos) if rhos else None),
                  phase_peak_binding=(seg[peak]["cert"]["binding"] if rhos else None))
        keyframes.append(kf)
        print(f"{ph.name}: {out} ticks={len(seg)} peak_rho={kf['phase_peak_rho']} "
              f"({kf['phase_peak_binding']}) t={kf['t']:.1f}s", flush=True)
        if out != "DONE":
            verdict = "NOT_FOUND"
            L = seg[-1]
            fail = dict(phase=ph.name, outcome=out, tick=L["tick"],
                        infeasible_diag=L.get("infeasible_diag"), elastic=L.get("elastic"),
                        cert=L.get("cert"), landing=L.get("landing"),
                        task_cost_last=L.get("task_cost_after"))
            break
    t_total = time.time() - t_wall

    # ---------------- per-tick arrays
    nT = len(logs)
    arr = dict(
        phase=np.array([L["phase"] for L in logs]),
        tick=np.array([L["tick"] for L in logs]),
        a_cur=np.array([L["sup"]["a_cur"] if "sup" in L else np.full(len(REGIONS), np.nan) for L in logs]),
        a_des=np.array([L["sup"]["a_des"] if "sup" in L else np.full(len(REGIONS), np.nan) for L in logs]),
        a_joint=np.array([L["mot"]["a_joint"] if "mot" in L else np.full(20, np.nan) for L in logs]),
        alpha=np.array([np.asarray(L["alpha"]).mean(0) if "alpha" in L else np.full(20, np.nan) for L in logs]),
        rho=np.array([L["cert"]["rho"] if L.get("cert") else np.nan for L in logs]),
        ratio=np.array([L["cert"]["ratio"] if L.get("cert") and L["cert"].get("ratio") is not None
                        else np.full(20, np.nan) for L in logs]),
        loads=np.array([[L["cert"]["loads"].get(r, 0.0) if L.get("cert") else np.nan for r in REGIONS]
                        for L in logs]),
        z=np.array([L["q"][2] if "q" in L else np.nan for L in logs]),
        q=np.array([L["q"] if "q" in L else np.full(len(keyframes[0]["q"]), np.nan) for L in logs]),
        qp_t=np.array([L.get("t", np.nan) for L in logs]),
    )
    os.makedirs(ROOT, exist_ok=True)
    np.savez(os.path.join(ROOT, f"run_{tag}.npz"), **arr,
             kf_names=np.array([k["name"] for k in keyframes]),
             kf_q=np.array([k["q"] for k in keyframes]))
    summ = dict(
        tag=tag, verdict=verdict, wall_clock_s=t_total, n_ticks=nT, rho_cap=p.rho_cap, mu_plan=p.mu,
        tau_max=dict(zip(JOINTS, S.tau_max.tolist())), regions=REGIONS, joints=JOINTS,
        mpc_params=p.__dict__,
        keyframes=[dict(name=k["name"], outcome=k["outcome"], ticks=k["ticks"], t=k["t"],
                        z=float(k["q"][2]), rho=k["cert"]["rho"], binding=k["cert"]["binding"],
                        loads={r: v for r, v in k["cert"]["loads"].items() if v > 1e-3},
                        feet_only=k["cert"]["feet_only"], pen=k["cert"]["pen"], selfd=k["cert"]["selfd"],
                        phase_peak_rho=k.get("phase_peak_rho"), phase_peak_binding=k.get("phase_peak_binding"))
                   for k in keyframes],
        n_gate_open=int(getattr(mpc, "n_gate_open", 0)),
        gate_open_ticks=[[L["phase"], int(L["tick"])] for L in logs if L.get("gate_opened")],
        n_cert_violation=int(sum(1 for L in logs if L.get("cert_violation"))),
        path_min_z_mm=float(1e3 * min(L["cert"]["pen"] for L in logs if L.get("cert"))),
        path_min_selfdist_mm=float(1e3 * min(L["cert"]["selfd"] for L in logs if L.get("cert"))),
        path_min_jl_margin_rad=float(min(L["cert"]["jl"] for L in logs if L.get("cert"))),
        step_scale_hist={str(s_): int(sum(1 for L in logs if L.get("step_scale") == s_)) for s_ in (1.0, 0.5, 0.25)},
        overall_peak_rho=float(np.nanmax(arr["rho"])) if nT else None,
        fail=fail,
        q_final_sha=float(np.round(keyframes[-1]["q"], 10).sum()),
    )
    with open(os.path.join(ROOT, f"run_{tag}.json"), "w") as f:
        json.dump(jl(summ), f, indent=1)
    print("VERDICT", verdict, "wall", round(t_total, 1), "s", "peak_rho", summ["overall_peak_rho"], flush=True)
    if fail:
        print("FAIL", json.dumps(jl({k: v for k, v in fail.items() if k != "cert"}), indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a")
