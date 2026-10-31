# SETUP_FOR_E4 — re-running a Step-1 edge with modified attention prices

E4 (attention ablation: attention vs uniform vs permuted) reuses the exact Step-1
harness; only the attention object handed to the oracle changes. No edits to
`muse/sim/` are needed — everything is injected through `OracleContext`.

## 1. Exact Step-1 call path (what E4 must reproduce)

```
workdir = muse/exp_e2_momentum_test/step1/
venv    = /home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python
sys.path: sim/  and  step1/            (run_step1.py inserts both)
```

```python
import sys, os
sys.path.insert(0, os.path.abspath("muse/sim"))
sys.path.insert(0, os.path.abspath("muse/exp_e2_momentum_test/step1"))
import run_step1 as R          # module import applies D1 (SIDE_L mode) + D3 (LADDER_TICKS=60)
import getup_oracle as go

ctx = R.make_ctx()             # OracleContext + D2 P-F3 bypass for SUPINE->SIT
q_clean, q_mirr = R.load_starts()
# starts: muse/results/getup_search/roll_probe_q_0.npy (clean SUPINE),
#         sagittal-mirror of it via mirror.mirror_q  (also clean SUPINE)

# >>> E4 modification point: replace ctx.A BEFORE evaluate_edge <<<
# ctx.A = <your SchemaAttention / wrapper>

s = R.run_edge(ctx, tag, src, dst, q_src)
#   tags/src/dst/q used in Step 1:
#   ("supine_to_side_R", "SUPINE", "SIDE",   q_clean)
#   ("supine_to_side_L", "SUPINE", "SIDE_L", q_mirr)
#   ("supine_to_sit",    "SUPINE", "SIT",    q_clean)   # P-F3 bypass already applied in make_ctx
```

Why replacing `ctx.A` is sufficient: `OracleContext.__init__` builds
`self.A = SchemaAttention(self.S)`, and every MPC instance in the pipeline is
constructed from `ctx.A` — `evaluate_edge` does `QSMPC(S, ctx.A, ctx.p)` per
edge, the cold-restart loop does `QSMPC(S, ctx.A, ctx.p)` again, and
`relaxation_ladder(ctx, q, ph)` does `QSMPC(S, ctx.A, pp)` per ladder family.
Nothing caches an attention object before your replacement.

`run_edge` streams: prints per-edge progress, writes `traj_<tag>.npz`
(q trajectory, per-tick selfd/pen/rho/cost, imagined keyframe chain) and
`edge_<tag>.json` (full EdgeResult + harness block: deviations, selfd
assertion, trunk R[2] start/end, wall time) into the step1/ directory, and
appends to `results.json` after every edge.

## 2. The attention object and its parameters

`SchemaAttention` lives in `muse/sim/getup_attention.py`. Constructor:
`SchemaAttention(S, p=AttnParams())`. The three heads and the exact call
signature the MPC expects:

| head | method | returns (keys the MPC reads) |
|---|---|---|
| SUPPORT | `support(k, JT, eT, c_star)` | `a_cur` (nR softmax: where load CAN+SHOULD go), `a_des`, `touching` (bool per region), `s_cur`, `s_des`, `kappa`, `conflict`, `util`, ... |
| MOTION | `motion(k, JT, eT, support_regions)` | `a_joint` (20-vector, sums to 1: which joints realise the sub-goal) |
| EFFORT | `effort(tau_bar)` | `(N,20)` alpha, mean 1; `None` → MPC uses ones |

`AttnParams` (defaults = the Step-1 baseline; change these for the "modified
prices" arm of E4):
```python
@dataclass
class AttnParams:
    w_kappa: float = 2.0    # statics price weight (support head)
    w_conf:  float = 4.0    # kinematic task-conflict weight (support head)
    w_util:  float = 2.0    # support utility weight (support head)
    sigma_u: float = 0.12   # [m] utility length scale
    lambda_d: float = 0.02  # [m] contact-proximity length scale
    T:       float = 0.5    # support softmax temperature
    T_e:     float = 0.1    # effort softmax temperature (torque-ratio units)
```

Example — modified prices (no sim/ edits):
```python
from getup_attention import SchemaAttention, AttnParams
ctx.A = SchemaAttention(ctx.S, AttnParams(w_kappa=8.0, T=0.25))  # sharper, statics-heavy
s = R.run_edge(ctx, "e4_sharp_side_R", "SUPINE", "SIDE", q_clean)
```

## 3. Uniform-attention arm

Keep the interface, return flat weights. Minimal wrapper (drop-in for `ctx.A`):

```python
import numpy as np
from getup_attention import SchemaAttention
from getup_schema import REGIONS

class UniformAttention(SchemaAttention):
    def support(self, k, JT, eT, c_star):
        out = super().support(k, JT, eT, c_star)   # reuse geometry: touching, low, d, ...
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

ctx.A = UniformAttention(ctx.S)
```

Notes: with uniform `a_cur`, the MPC's contact gate `a_cur >= p.a_gate`
(`MPCParams.a_gate`, default in getup_mpc.py) admits every near point, and the
force price `1/(a+a0)` becomes flat — the QP then prices force purely by its
own objective. `support_regions` (motion head input) is still derived from
`touching`, so kinematics stay schema-grounded; only the prices are flat.

## 4. Permuted-attention arm

Permute the *weights*, not the geometry (a control that keeps magnitudes but
destroys the schema grounding):

```python
class PermutedAttention(SchemaAttention):
    def __init__(self, S, p=None, seed=0):
        super().__init__(S, p or AttnParams())
        rng = np.random.default_rng(seed)
        self.prm_r = rng.permutation(len(REGIONS))
        self.prm_j = rng.permutation(20)
    def support(self, k, JT, eT, c_star):
        out = super().support(k, JT, eT, c_star)
        out["a_cur"] = out["a_cur"][self.prm_r]
        out["a_des"] = out["a_des"][self.prm_r]
        return out
    def motion(self, k, JT, eT, support_regions):
        out = super().motion(k, JT, eT, support_regions)
        out["a_joint"] = out["a_joint"][self.prm_j]
        return out
```

Fix the seed for reproducibility; run ≥3 seeds if you want the variance.

## 5. Interpreting E4 against the Step-1 baseline

Step-1 result you are ablating (see STEP1_NOTES.md): SUPINE→SIDE stalls at the
first tracking sub-goal with binding family `no_hold_ramp` (contact-state
machine), dominant residual `com` (10.04), `trunk_left` 2.06 / `trunk_head`
2.00 / `posture` 3.33 / `upper_trunk_land` 3.70, zero roll achieved
(R[2,1] ≈ 0 throughout), selfd clean (+3.3…+5.0 mm).

The falsification question for E4: does any attention arm change (a) the
stall verdict, (b) which ladder family binds, or (c) the residual-term
profile? Because the stall is constraint-bound (state machine), not
price-bound, the prior is that attention variants should NOT unstick it —
any arm that does is the interesting result, not the null.

## 6. Cost / determinism notes

- Deterministic given the attention object (fixed seeds in permuted arm; the
  oracle itself is deterministic). Same (tag-independent) trajectory for the
  same `ctx.A`.
- Wall time ≈ 14 min/edge on this VM (821 s for SUPINE→SIDE_R; qp ~0.7–1.5 s
  per MPC tick, STALL_TICKS=60, up to 4 cold restarts, ladder 6×60 ticks).
  Budget 90 min/edge; kill and mark TIMEOUT beyond that.
- Keep one process at a time: the MPC is single-threaded and the VM has 2
  CPUs; parallel edges thrash the QP timings.
