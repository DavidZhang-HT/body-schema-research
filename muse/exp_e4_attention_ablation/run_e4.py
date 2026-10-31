#!/usr/bin/env python
"""E4 (attention ablation): SUPINE->SIDE_R with three attention arms on the clean Step-1 setup.

HARNESS-LOCAL pieces only (sim/ untouched):
  E4a. import run_step1 as R  -> applies the committed Step-1 deviations
       D1 (SIDE_L local mode), D3 (LADDER_TICKS=60), D4 (run_phase log capture).
       D2 (P-F3 bypass) is inside R.make_ctx and only affects SUPINE->SIT;
       the SIDE edge is unaffected by it.
  E4b. ctx.A replaced with one of three arms BEFORE R/evaluate_edge:
       - attn     : fresh SchemaAttention(ctx.S) with default AttnParams  (control)
       - uniform  : UniformAttention  — support a_cur/a_des flat over regions,
                    motion a_joint flat over 20 joints, effort() -> None (MPC alpha=ones)
       - permuted : PermutedAttention(seed=7) — same weight magnitudes, region vector
                    permuted by prm_r, joint vector permuted by prm_j; effort head
                    inherited from SchemaAttention (real alpha).  (Matches the
                    SETUP_FOR_E4.md sketch exactly; the uniform/permuted effort
                    asymmetry is documented in the report.)
  E4c. run_edge adapted from R.run_edge; identical oracle behaviour, writes go to
       this workdir instead of step1/.

Usage: python run_e4.py <arm>       arm in {attn, uniform, permuted}
Writes: edge_e4_<arm>_R.json, traj_e4_<arm>_R.npz, results_e4.json (incremental)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = "/home/hatch/workspace/research/body-schema-gait/muse/sim"
STEP1 = "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/step1"
sys.path.insert(0, SIM)
sys.path.insert(0, STEP1)

import numpy as np

import run_step1 as R          # noqa: E402  (applies D1/D3/D4 at import)
import getup_oracle as go      # noqa: E402
from getup_attention import SchemaAttention, AttnParams  # noqa: E402
from getup_schema import REGIONS  # noqa: E402
from getup_keyframes import KeyframeImaginer  # noqa: E402

SELF_OK = R.SELF_OK
PERM_SEED = 7


# --------------------------------------------------------------- attention arms
class UniformAttention(SchemaAttention):
    """Flat prices; geometry (touching, low, d) inherited from the real heads."""

    def support(self, k, JT, eT, c_star):
        out = super().support(k, JT, eT, c_star)
        n = len(REGIONS)
        out["a_cur"] = np.full(n, 1.0 / n)
        out["a_des"] = np.full(n, 1.0 / n)
        return out

    def motion(self, k, JT, eT, support_regions):
        out = super().motion(k, JT, eT, support_regions)
        out["a_joint"] = np.full(20, 1.0 / 20)
        return out

    def effort(self, tau_bar):
        return None          # MPC falls back to alpha = ones((N, 20))


class PermutedAttention(SchemaAttention):
    """Real weight magnitudes, schema grounding destroyed by a fixed permutation."""

    def __init__(self, S, seed=PERM_SEED):
        super().__init__(S, AttnParams())
        rng = np.random.default_rng(seed)
        self.prm_r = rng.permutation(len(REGIONS))
        self.prm_j = rng.permutation(20)
        self.seed = seed

    def support(self, k, JT, eT, c_star):
        out = super().support(k, JT, eT, c_star)
        out["a_cur"] = out["a_cur"][self.prm_r]
        out["a_des"] = out["a_des"][self.prm_r]
        return out

    def motion(self, k, JT, eT, support_regions):
        out = super().motion(k, JT, eT, support_regions)
        out["a_joint"] = out["a_joint"][self.prm_j]
        return out


ARMS = {
    "attn":     ("SchemaAttention default AttnParams (control arm)",
                 lambda S: SchemaAttention(S, AttnParams())),
    "uniform":  ("UniformAttention (flat prices, effort=None)",
                 lambda S: UniformAttention(S)),
    "permuted": (f"PermutedAttention(seed={PERM_SEED}) (magnitudes kept, grounding destroyed)",
                 lambda S: PermutedAttention(S, seed=PERM_SEED)),
}


# --------------------------------------------------------------- edge runner (adapted from R.run_edge)
def run_edge(ctx, tag, src, dst, q_src, outdir=HERE):
    print(f"\n===== E4 edge {tag}: {src} -> {dst} =====", flush=True)
    S, mu = ctx.S, ctx.p.mu
    c0, k0 = R.check_start(S, q_src, mu, src)
    print(f"[harness] start ok: mode {src}, rho {c0['rho']:.4f}, "
          f"selfd {1e3 * c0['selfd']:.1f} mm", flush=True)
    R._cap.clear()
    t0 = time.time()
    res = go.evaluate_edge(ctx, q_src, src, dst, verbose=True, ladder=True)
    wall = time.time() - t0
    logs = R._cap.get("logs", [])
    print(f"[harness] edge done: status={res.status} ticks={res.ticks} "
          f"wall={wall:.1f}s n_log={len(logs)}", flush=True)

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
        print(f"[harness] ABORT: selfd < -1mm at {len(viol)} ticks", flush=True)

    if len(traj["q"]):
        R0 = S.kin(traj["q"][0], with_G=False, with_pairs=False).R
        R1 = S.kin(traj["q"][-1], with_G=False, with_pairs=False).R
    else:
        R0 = R1 = k0.R
    col, elev = go.ORIENT_TARGET[dst]
    chain = KeyframeImaginer(S, mu).chain(q_src, dst, col, elev, verbose=False)
    chain_q = np.array([kf["q"] for kf in chain["keyframes"]]) if chain["keyframes"] else np.zeros((0, len(q_src)))
    chain_stage = [kf["stage"] for kf in chain["keyframes"]]

    np.savez(os.path.join(outdir, f"traj_{tag}.npz"),
             q_traj=traj["q"], tick=traj["tick"], phase=np.array(traj["phase"]),
             selfd_m=traj["selfd"], pen_m=traj["pen"], rho=traj["rho"],
             task_cost=traj["cost"], chain_q=chain_q,
             chain_stage=np.array(chain_stage), q_src=q_src, q_end=res.q_end)

    edge_doc = R.to_jsonable(res)
    edge_doc["harness"] = dict(tag=tag, ladder_ticks=go.LADDER_TICKS,
                               deviations=["D1 SIDE_L local mode (unused by this edge)",
                                           "D2 P-F3 bypass (SIT only, unused by this edge)",
                                           "D3 LADDER_TICKS=60", "D4 run_phase log capture",
                                           "E4 ctx.A replaced per arm"],
                               attention_arm=tag,
                               selfd_assert_passed=not aborted,
                               selfd_min_traj_mm=round(selfd_min * 1e3, 3) if selfd_min == selfd_min else None,
                               selfd_violation_ticks=viol[:20],
                               trunk_R2_start=[round(float(v), 4) for v in R0[2]],
                               trunk_R2_end=[round(float(v), 4) for v in R1[2]],
                               wall_s_edge=round(wall, 1))
    with open(os.path.join(outdir, f"edge_{tag}.json"), "w") as f:
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
                diagnosis_stage=(res.diagnosis or {}).get("stage"),
                ladder=(res.diagnosis or {}).get("ladder"))
    print(f"[harness] summary: {json.dumps(summ, default=str)}", flush=True)
    return summ


def main(arm):
    assert arm in ARMS, f"arm must be one of {sorted(ARMS)}"
    desc, make_A = ARMS[arm]
    print(f"[E4] arm={arm}: {desc}", flush=True)
    ctx = R.make_ctx()
    q_clean, _ = R.load_starts()
    ctx.A = make_A(ctx.S)     # <<< the only behavioural change vs Step 1
    tag = f"e4_{arm}_R"
    try:
        s = run_edge(ctx, tag, "SUPINE", "SIDE", q_clean, outdir=HERE)
        s["arm"] = arm
        s["arm_desc"] = desc
        s["perm_seed"] = PERM_SEED if arm == "permuted" else None
    except Exception as e:
        import traceback
        traceback.print_exc()
        s = dict(tag=tag, arm=arm, status="HARNESS_ERROR", reason=f"{type(e).__name__}: {e}")
    res_path = os.path.join(HERE, "results_e4.json")
    prev = json.load(open(res_path))["edges"] if os.path.exists(res_path) else []
    prev = [p for p in prev if p.get("arm") != arm] + [s]
    with open(res_path, "w") as f:
        json.dump(dict(edges=prev,
                       harness_deviations=["D1 SIDE_L local mode", "D2 P-F3 bypass (SIT only)",
                                           "D3 LADDER_TICKS=60", "D4 run_phase log capture",
                                           "E4 ctx.A replaced per arm"]),
                  f, indent=1)
    print("\n[E4] arm done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "attn")
