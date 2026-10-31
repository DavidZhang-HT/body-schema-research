#!/usr/bin/env python
"""E2 Step 1: clean re-run of scheme A's get-up oracle (SUPINE->SIDE R/L, SUPINE->SIT).

HARNESS-LOCAL deviations from committed sim/ semantics (all documented, sim/ untouched):
  D1. Local mirrored mode SIDE_L registered into getup_modes.MODES / MODE and
      getup_oracle.ORIENT_TARGET (left-roll test; the committed vocabulary is right-only).
  D2. P-F3 prune bypassed for SUPINE->SIT only: ctx.rules[("SUPINE","SIT")] set to candidate.
  D3. getup_oracle.LADDER_TICKS monkeypatched 15 -> 60 (>=60 ticks per relaxation rung).
  D4. getup_oracle.run_phase wrapped to capture the per-tick logs (q trajectory, selfd trace).

Usage: python run_step1.py   (writes edge_*.json, traj_*.npz, results.json incrementally)
"""
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..", "sim")
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)

import numpy as np

import getup_modes as gm
import getup_oracle as go
from getup_modes import Mode, EdgeRule, mode_check, LYING_SIDE
from getup_mpc import certify
from getup_keyframes import KeyframeImaginer
from model import JOINTS
from mirror import mirror_q

SELF_OK = -0.001  # [m] clean-start / per-tick self-collision bound


# ------------------------------------------------------------------ D1: SIDE_L mode
def _f(*r):
    return frozenset(r)


SIDE_L = Mode(
    "SIDE_L", "side-lying {left side}",
    req=(_f("upper_trunk", "lower_back", "pelvis", "L_upper_arm"),), allowed=gm.ALL,
    orient_txt="left_z <= -0.7 (body +y down, i.e. left side down)",
    orient=lambda R: R[2, 1] <= -0.7,
    oclass=LYING_SIDE, level=0,
    meaning="harness-local exact mirror of SIDE (E2 Step 1 only); tests left roll")
gm.MODES.append(SIDE_L)
gm.MODE["SIDE_L"] = SIDE_L
go.ORIENT_TARGET["SIDE_L"] = (1, -90.0)   # body +y axis down

# ------------------------------------------------------------------ D3: ladder ticks
go.LADDER_TICKS = 60
assert go.RELAX_LADDER.get("no_hold_ramp") is not None, "no_hold_ramp missing from ladder"
assert all(v >= 60 for v in go.SUBGOAL_TICKS.values()), go.SUBGOAL_TICKS
assert go.TRACK_TICKS >= 60, go.TRACK_TICKS

# ------------------------------------------------------------------ D4: capture per-tick logs
_cap = {}
_orig_run_phase = go.run_phase


def _wrap_run_phase(S, mpc, q, phase, mu, logs, stall_ticks=60, stall_tol=0.01, verbose=True):
    if "logs" not in _cap:
        _cap["logs"] = logs
    return _orig_run_phase(S, mpc, q, phase, mu, logs, stall_ticks, stall_tol, verbose)


go.run_phase = _wrap_run_phase


# ------------------------------------------------------------------ serialisation
def to_jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if is_dataclass(o):
        return to_jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    return o


# ------------------------------------------------------------------ context
def make_ctx():
    ctx = go.OracleContext()
    # D2: P-F3 bypass for this test only (SUPINE->SIT evaluated despite the committed prune)
    assert ctx.rules[("SUPINE", "SIT")].rule == "P-F3", ctx.rules[("SUPINE", "SIT")]
    ctx.rules[("SUPINE", "SIT")] = EdgeRule("SUPINE", "SIT", "candidate")
    r_sl = ctx.rules[("SUPINE", "SIDE_L")]
    assert r_sl.status == "candidate", (r_sl.status, r_sl.rule, r_sl.reason)
    print(f"[harness] SUPINE->SIDE_L rule: {r_sl.status} ({r_sl.rule or 'no prune'})", flush=True)
    print(f"[harness] SUPINE->SIT rule after bypass: {ctx.rules[('SUPINE','SIT')].status}", flush=True)
    print(f"[harness] LADDER_TICKS={go.LADDER_TICKS} SUBGOAL_TICKS={go.SUBGOAL_TICKS} "
          f"TRACK_TICKS={go.TRACK_TICKS}", flush=True)
    return ctx


def load_starts():
    from getup_schema import Schema
    S = Schema()
    q_clean = np.load(os.path.join(HERE, "..", "..", "results", "getup_search", "roll_probe_q_0.npy"))
    q_mirr = mirror_q(q_clean, JOINTS, S.r.qadr)
    return q_clean, q_mirr


def check_start(S, q, mu, key):
    c = certify(S, q, mu)
    k = S.kin(q, with_G=False, with_pairs=False)
    ok, why = mode_check(S, k, c, key)
    assert ok, f"start not in mode {key}: {why}"
    assert bool(c["feasible"]) and c["selfd"] >= SELF_OK, (c["feasible"], c["selfd"])
    return c, k


def run_edge(ctx, tag, src, dst, q_src):
    print(f"\n===== edge {tag}: {src} -> {dst} =====", flush=True)
    S, mu = ctx.S, ctx.p.mu
    c0, k0 = check_start(S, q_src, mu, src)
    print(f"[harness] start ok: mode {src}, rho {c0['rho']:.4f}, "
          f"selfd {1e3 * c0['selfd']:.1f} mm", flush=True)
    _cap.clear()
    t0 = time.time()
    res = go.evaluate_edge(ctx, q_src, src, dst, verbose=True, ladder=True)
    wall = time.time() - t0
    logs = _cap.get("logs", [])
    print(f"[harness] edge done: status={res.status} ticks={res.ticks} "
          f"wall={wall:.1f}s n_log={len(logs)}", flush=True)

    # ---- trajectory + per-tick selfd assertion
    traj = dict(tick=[], phase=[], q=[], selfd=[], pen=[], rho=[], cost=[])
    last_q = q_src.copy()
    for L in logs:
        qq = L.get("q")
        if qq is None:
            qq = last_q
        last_q = qq
        traj["tick"].append(L.get("tick", -1))
        traj["phase"].append(L.get("phase", "?"))
        traj["q"].append(np.asarray(qq, float))
        cc = L.get("cert", {})
        traj["selfd"].append(float(cc.get("selfd", np.nan)))
        traj["pen"].append(float(cc.get("pen", np.nan)))
        r = cc.get("rho", np.nan)
        traj["rho"].append(float(r) if r is not None and np.isfinite(r) else np.nan)
        traj["cost"].append(float(L.get("task_cost_after", np.nan)))
    for k_ in ("tick", "selfd", "pen", "rho", "cost"):
        traj[k_] = np.array(traj[k_])
    traj["q"] = np.array(traj["q"]) if traj["q"] else np.zeros((0, len(q_src)))
    selfd_min = float(np.nanmin(traj["selfd"])) if len(traj["selfd"]) else float("nan")
    viol = [int(i) for i, s in enumerate(traj["selfd"]) if s < SELF_OK]
    aborted = bool(viol)
    if aborted:
        print(f"[harness] ABORT: selfd < -1mm at {len(viol)} ticks, first tick={viol[0]} "
              f"phase={traj['phase'][viol[0]]} selfd={traj['selfd'][viol[0]] * 1e3:.2f} mm", flush=True)

    # ---- trunk rotation achieved (from trajectory)
    if len(traj["q"]):
        R0 = S.kin(traj["q"][0], with_G=False, with_pairs=False).R
        R1 = S.kin(traj["q"][-1], with_G=False, with_pairs=False).R
    else:
        R0 = R1 = k0.R
    # ---- imagined keyframe chain q's (deterministic re-run of the imagine stage)
    col, elev = go.ORIENT_TARGET[dst]
    chain = KeyframeImaginer(S, mu).chain(q_src, dst, col, elev, verbose=False)
    chain_q = np.array([kf["q"] for kf in chain["keyframes"]]) if chain["keyframes"] else np.zeros((0, len(q_src)))
    chain_stage = [kf["stage"] for kf in chain["keyframes"]]

    np.savez(os.path.join(HERE, f"traj_{tag}.npz"),
             q_traj=traj["q"], tick=traj["tick"], phase=np.array(traj["phase"]),
             selfd_m=traj["selfd"], pen_m=traj["pen"], rho=traj["rho"],
             task_cost=traj["cost"], chain_q=chain_q,
             chain_stage=np.array(chain_stage), q_src=q_src, q_end=res.q_end)

    edge_doc = to_jsonable(res)
    edge_doc["harness"] = dict(tag=tag, ladder_ticks=go.LADDER_TICKS,
                               deviations=["D1 SIDE_L local mode", "D2 P-F3 bypass (SIT only)",
                                           "D3 LADDER_TICKS=60", "D4 run_phase log capture"],
                               selfd_assert_passed=not aborted,
                               selfd_min_traj_mm=round(selfd_min * 1e3, 3) if selfd_min == selfd_min else None,
                               selfd_violation_ticks=viol[:20],
                               trunk_R2_start=[round(float(v), 4) for v in R0[2]],
                               trunk_R2_end=[round(float(v), 4) for v in R1[2]],
                               wall_s_edge=round(wall, 1))
    with open(os.path.join(HERE, f"edge_{tag}.json"), "w") as f:
        json.dump(edge_doc, f, indent=1)

    summ = dict(tag=tag, src=src, dst=dst, status=res.status,
                reason=res.reason, rule=res.rule, ticks=res.ticks,
                wall_s=round(res.wall_s, 1), wall_s_edge=round(wall, 1),
                n_subgoals=len(res.subgoals),
                subgoal_outcomes=[(s.get("name"), s.get("outcome")) for s in res.subgoals],
                n_keyframes=len(res.keyframes),
                cost_total=(res.cost or {}).get("total"), peak_rho=(res.cost or {}).get("peak_rho"),
                selfd_min_traj_mm=(round(selfd_min * 1e3, 3) if selfd_min == selfd_min else None),
                selfd_violations=len(viol),
                trunk_R2_start=[round(float(v), 4) for v in R0[2]],
                trunk_R2_end=[round(float(v), 4) for v in R1[2]],
                diagnosis_kind=(res.diagnosis or {}).get("kind"),
                diagnosis_subgoal=(res.diagnosis or {}).get("subgoal"),
                diagnosis_stage=(res.diagnosis or {}).get("stage"))
    print(f"[harness] summary: {json.dumps(summ, default=str)}", flush=True)
    return summ


def main():
    ctx = make_ctx()
    q_clean, q_mirr = load_starts()
    # verify starts once more under the real ctx
    for q, key, nm in ((q_clean, "SUPINE", "clean"), (q_mirr, "SUPINE", "mirrored")):
        c, k = check_start(ctx.S, q, ctx.p.mu, key)
        print(f"[harness] start[{nm}]: mode {key} ok, rho {c['rho']:.4f}, "
              f"selfd {1e3 * c['selfd']:.1f} mm", flush=True)
    np.save(os.path.join(HERE, "q_supine_clean.npy"), q_clean)
    np.save(os.path.join(HERE, "q_supine_clean_mirrored.npy"), q_mirr)

    edges = [
        ("supine_to_side_R", "SUPINE", "SIDE", q_clean),
        ("supine_to_side_L", "SUPINE", "SIDE_L", q_mirr),
        ("supine_to_sit", "SUPINE", "SIT", q_clean),
    ]
    if len(sys.argv) > 1:          # smoke / resume: run only named edges
        edges = [e for e in edges if e[0] in sys.argv[1:]]
        assert edges, f"no edges match {sys.argv[1:]}"
        print(f"[harness] edge filter: {[e[0] for e in edges]}", flush=True)
    results = []
    for tag, src, dst, q in edges:
        try:
            s = run_edge(ctx, tag, src, dst, q)
        except Exception as e:  # never fake results: record the failure
            import traceback
            traceback.print_exc()
            s = dict(tag=tag, src=src, dst=dst, status="HARNESS_ERROR", reason=f"{type(e).__name__}: {e}")
        results.append(s)
        with open(os.path.join(HERE, "results.json"), "w") as f:
            json.dump(dict(edges=results,
                           harness_deviations=["D1 SIDE_L local mode", "D2 P-F3 bypass (SIT only)",
                                               "D3 LADDER_TICKS=60", "D4 run_phase log capture"]),
                      f, indent=1)
    print("\n[harness] ALL EDGES DONE", flush=True)


if __name__ == "__main__":
    main()
