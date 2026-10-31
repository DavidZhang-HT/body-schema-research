# E4 — attention-head ablation (SUPINE→SIDE edge + F3 scenario)

**Date:** 2026-11-01 · **Workdir:** `muse/exp_e4_attention_ablation/`
**Question:** are scheme A's three attention heads (support / motion / effort, zero learned
parameters, closed-form QP prices) *functional components* of the get-up planner, or
*descriptive read-outs* that don't change planning behaviour?

**Fixed falsifier (applied honestly):** uniform and permuted prices reaching the SAME
progress, cost, and stall point as attention prices ⇒ heads are descriptive read-outs
(useful for explanation, not credited with planning behaviour). Attention prices doing
strictly better ⇒ report the margin.

**Bottom line:** the falsifier does **not** trigger in its strict form — attention prices
beat uniform prices by a wide, measurable margin on both scenarios (functional, not
read-only). But the *planning verdict* is attention-invariant: all three arms stall at the
same point with the same binding constraint, and the E2 prior ("re-tuning prices will not
unstick T1") holds. Honest verdict: **the heads are functional steering components that
shape the MPC's trajectory within the feasible region, but they are not planning-critical —
the stall is constraint-bound, not price-bound.**

---

## 1. Methods

### 1.1 Setup reuse (clean E2-Step-1 harness, `sim/` untouched)

- venv: `/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python`
- `run_e4.py` imports `run_step1` as a module, which applies the committed Step-1
  deviations at import: **D1** (local `SIDE_L` mode), **D3** (`LADDER_TICKS=60`),
  **D4** (`run_phase` log capture). D2 (P-F3 bypass) lives inside `make_ctx()` but only
  affects the SUPINE→SIT edge, which E4 does not run.
- The **only** behavioural change vs Step 1: `ctx.A` is replaced *before*
  `evaluate_edge`. Verified in `getup_oracle.py` that every MPC construction reads
  `ctx.A` at call time (lines 334, 384, 412 — ladder, connectivity, cold-restart), so
  the replacement propagates everywhere. Nothing caches an attention object earlier.
- Clean start: `roll_probe_q_0.npy` (clean SUPINE), `selfd >= -1 mm` asserted at start
  and every tick; trunk rotation measured from trajectory.
- `run_edge` is adapted from `R.run_edge` (identical oracle behaviour; writes go to
  this workdir instead of `step1/`). Imagine stage (`KeyframeImaginer.chain`) is
  attention-free and therefore identical across arms by construction.

### 1.2 The three arms

| arm | construction | seed |
|---|---|---|
| `attn` (control) | fresh `SchemaAttention(ctx.S)` with default `AttnParams` | — |
| `uniform` | `UniformAttention`: support `a_cur`/`a_des` flat over 18 regions; motion `a_joint` flat over 20 joints; `effort()` → `None` (MPC falls back to `alpha = ones`) | — |
| `permuted` | `PermutedAttention`: real weight magnitudes, region vector permuted by `prm_r`, joint vector permuted by `prm_j`; effort head inherited (real alpha) — exactly the `SETUP_FOR_E4.md` sketch | **7** |

`seed=7` permutations (`numpy.random.default_rng(7)`):
- `prm_r` (18 regions) = `[3, 14, 6, 8, 1, 10, 0, 7, 4, 16, 15, 17, 13, 2, 12, 5, 9, 11]`
- `prm_j` (20 joints)  = `[0, 19, 9, 4, 17, 6, 10, 15, 12, 18, 7, 14, 3, 8, 11, 1, 13, 16, 2, 5]`

### 1.3 Exact commands

```bash
cd /home/hatch/workspace/research/body-schema-gait/muse/exp_e4_attention_ablation
V=/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python
# (i) SUPINE->SIDE_R oracle edge, one arm at a time (2-CPU VM; parallel runs thrash QP timing)
$V run_e4.py attn        # -> edge_e4_attn_R.json, traj_e4_attn_R.npz      (~8.5 min)
$V run_e4.py uniform     # -> edge_e4_uniform_R.json, traj_e4_uniform_R.npz (~21 min)
$V run_e4.py permuted    # -> edge_e4_permuted_R.json, traj_e4_permuted_R.npz (~7 min)
# (ii) F3 scenario = getup_f3_run pipeline (supine->standing keyframe search),
#      attention swapped by monkeypatching the module's SchemaAttention reference + ROOT
cp ../results/f3/q_supine_settled.npy .     # read-only input copy (script reads ROOT/q_supine_settled.npy)
$V f3_e4.py attn        # -> run_f3_e4_attn.json / .npz
$V f3_e4.py uniform     # -> run_f3_e4_uniform.json / .npz
$V f3_e4.py permuted    # -> run_f3_e4_permuted.json / .npz
```

Harness-local deviations (all documented, `sim/` semantics untouched):
E4a ctx.A replaced per arm · E4b outputs redirected to this workdir ·
E4c F3: `getup_f3_run.main` imported as module, `SchemaAttention` → arm factory,
`ROOT` → workdir (the script reads its start config from `ROOT`, hence the input copy).

---

## 2. Results (i): SUPINE→SIDE_R oracle edge

### 2.1 Per-arm table

| metric | attn (control) | uniform | permuted (seed 7) |
|---|---|---|---|
| status | INFEASIBLE | INFEASIBLE | INFEASIBLE |
| stall point | T1_track[elev +15°] | T1_track[elev +15°] | T1_track[elev +15°] |
| tracking ticks | 202 | 309 | 152 |
| stall task cost (ladder cost0) | **21.14** | **33.89** | **24.28** |
| residual `com` | **10.04** | **25.97** | **11.47** |
| residual `trunk_head(+z)` | 2.00 | 0.01 | 1.73 |
| residual `upper_trunk_land` | 3.70 | 0.00 | 3.20 |
| residual `posture` | 3.33 | 5.84 | 5.89 |
| binding families | `[no_hold_ramp]` | `[no_hold_ramp]` | `[]` (see §2.2) |
| ladder: no_hold_ramp drop | 21.14 → 13.07 (QP_INFEASIBLE) | 33.89 → **0.28** | 24.28 → 19.64 (QP_INFEASIBLE) |
| peak rho | 0.096 | 0.0969 | 0.0964 |
| roll achieved | 0° | 0° | 0° |
| trunk R[2] end | [0.852, 0.0001, 0.5236] (pitched up) | [0.9947, −0.0, 0.1032] (unmoved) | [0.8685, 0.0049, 0.4956] (pitched up) |
| selfd min (traj) | +3.28 mm | +3.76 mm | +2.78 mm |
| selfd violations | 0 | 0 | 0 |
| path cost_total | 2.6046 | 3.3621 | 2.2008 |
| wall time | 504 s | 1248 s | 414 s |

Control-arm reproduction check: attn arm reproduces Step-1 byte-identically —
status, reason (`T1_track[elev +15deg] STALLED: constraint-bound: no_hold_ramp`),
ticks (202), cost dict (`{switch:1.0, effort:0.4746, margin:0.12, length:1.01,
peak_rho:0.096, total:2.6046}`), residual terms to 3 decimals. The E4 harness is
the Step-1 harness.

### 2.2 What the arms agree on (the planning verdict is attention-invariant)

- **Same stall point, same verdict:** all three arms are INFEASIBLE at
  `T1_track[elev +15°]` — the first 15° of roll. Existence stage OK for all three
  (identical imagined chain: `elev +15/+30/+45°`, attention-free by construction).
- **Same binding constraint:** `no_hold_ramp` (the quasi-static contact-state machine)
  is the binding family for attn and uniform, and the *only* family with a substantive
  drop for permuted. The permuted arm's `binding_families=[]` is a **threshold
  artifact**, not a different physics: its no_hold_ramp drop is
  4.64 = 19.1% of cost0, just under the oracle's `UNLOCK_FRAC = 0.25` bar
  (4.64 < 0.25×24.28 = 6.07); all other families drop ≤ 0.39 there.
  Verdict strings: attn/uniform `"constraint-bound: no_hold_ramp"`,
  permuted `"local: ... no single constraint-family relaxation unlocks it"` —
  same ladder table, different label, caused by the 25% cutoff.
- **Same qualitative behaviour:** 0° roll in all arms (R[2,1] ≈ 0 throughout),
  com-dominant residual, clean selfd (+2.8…+5.0 mm, 0 violations), never
  torque-bound (peak rho ≤ 0.097 ≪ rho_cap 0.8), unmet predicate `left_z >= 0.7`
  in all arms.

### 2.3 Where the arms differ (the heads steer the trajectory)

- **attn vs uniform — large, strict margin.** Uniform stall cost 33.89 vs 21.14
  (−38% with attention); com residual 25.97 vs 10.04; 309 vs 202 tracking ticks
  (~5 cold-restart attempts, each making ≥2% gain without escaping — thrashing);
  wall 1248 s vs 504 s (flat `a_cur` admits every near point through the contact
  gate `a_cur >= a_gate`, enlarging each QP); and qualitatively, the uniform MPC
  never even pitches the trunk up (R[2] end ≈ start), while attention's pitches
  chest_z 0.99→0.85 / head_z 0.10→0.52. Flat prices are strictly worse.
- **attn vs permuted (seed 7) — small margin, similar behaviour.** Stall cost
  21.14 vs 24.28 (−13% with attention); com 10.04 vs 11.47; trunk pitches up in
  both; end load distributions similar (lower_back-dominant: 0.691 vs 0.726).
  Permuted used *fewer* ticks (152 vs 202). With n=1 seed this reads as: the stall
  basin is broad — one random-permuted price set lands nearly where the
  schema-grounded prices land.
- **Uniform's ladder is the most damning for prices-as-cause:** relaxing
  no_hold_ramp collapses its stall cost 33.89 → 0.28 — i.e. with uniform prices
  the stall is *even more* explained by the contact-state machine than with
  attention prices. The worse the prices, the more the failure is about contact
  switching, not about prices.

---

## 3. Results (ii): F3 scenario (supine→standing keyframe search, current tree)

| metric | attn (control) | uniform | permuted (seed 7) |
|---|---|---|---|
| verdict | NOT_FOUND | NOT_FOUND | NOT_FOUND |
| P1_head_up | **DONE (97 ticks)** | STALLED (94) | STALLED (87) |
| failure point | P2_tuck MAX_TICKS (400) | P1_head_up STALLED | P1_head_up STALLED |
| fail task cost | 23.58 | 26.72 | 26.96 |
| peak rho | 0.192 | 0.686 (near rho_cap 0.8) | 0.094 |
| tracking ticks | 497 | 94 | 87 |
| wall | 511 s | 85 s | 82 s |

Here attention prices **strictly** beat both ablations: only the attention arm
completes P1 (head-up to `tilt < 15°`); uniform and permuted stall in P1 with
higher residual cost, and uniform additionally drives peak rho to 0.686
(torque-pressure the attention arm never sees). But the overall F3 verdict is
still NOT_FOUND in all three arms — attention moves the failure from P1 to P2,
it does not produce a standing keyframe.

**Baseline note (honest):** the committed F3 baseline `results/f3/run_a.json`
(NOT_FOUND, P1 STALLED at 122 ticks) was written at 11:26 while
`getup_f3_run.py` was last edited at 11:29 the same day — it predates the
current script version and is **not** a valid control for this comparison.
The valid control is my own attn run on the current tree (above). Determinism
check: re-running the attn F3 arm reproduced P1 bit-identically
(DONE, 97 ticks, peak_rho 0.19237360048575208 identical to 16 decimals), so the
arm comparison is internally consistent. (The rerun process was killed by the
runtime during P2 — 7 GB VM, memory pressure — after P1's bit-identity was
already established.)

---

## 4. Falsification verdict

The strict falsifier — uniform **and** permuted reaching the SAME progress,
cost, and stall point as attention — **does not trigger**:

1. **Same stall point, same binding constraint, same verdict** across all arms
   (edge: T1[elev+15°] INFEASIBLE, 0° roll, no_hold_ramp binding/near-binding;
   F3: NOT_FOUND). The E2 prior is confirmed: **attention re-tuning does not
   unstick T1 and does not change the planning verdict.**
2. **But costs and progress are not the same.** Attention prices strictly beat
   uniform: edge stall cost 21.14 vs 33.89, com residual 10.04 vs 25.97, fewer
   ticks 202 vs 309; F3 P1 DONE vs STALLED. The heads therefore **are functional
   steering components** — they change the QP's prices, the path taken, and the
   quality of the stall — not pure read-outs. The "descriptive read-outs" label
   is rejected.
3. **The permuted arm (n=1) weakens any stronger claim:** a single fixed
   permutation reaches a stall comparable to attention's (cost 24.28 vs 21.14,
   same qualitative behaviour). The schema-grounding of the prices is
   load-bearing for *trajectory quality* (uniform shows the floor), but the
   planning-relevant outcome — where and why the planner stalls — is
   price-insensitive.

**One-line verdict for the falsification program:** the attention heads earn
credit for shaping behaviour inside the feasible region, but **zero credit for
the planning outcome** — the get-up stall is constraint-bound (contact-state
machine), not price-bound. Falsifying the planner's stall verdict will have to
come from defeating `no_hold_ramp` (E2 Steps 2–3: momentum/dynamic strategies),
not from re-tuning attention.

---

## 5. Honest caveats

1. **Single permuted seed (7).** Permutation variance is untested; the near-parity
   of permuted vs attention could be seed luck. Recommended follow-up: 2–3 more
   seeds on the SUPINE→SIDE edge only (~15 min each).
2. **Effort-head asymmetry between the ablation arms** (per the SETUP_FOR_E4.md
   sketch): uniform ablates all three heads (`effort()` → None ⇒ MPC alpha=ones);
   permuted ablates only support+motion and inherits the real effort head. The
   uniform-vs-attn margin therefore bundles the effort ablation; the
   permuted-vs-attn margin does not.
3. **Permuted binding-label difference is a threshold artifact** (§2.2): 19.1%
   drop vs the oracle's 25% `UNLOCK_FRAC` cutoff. Do not read it as a different
   binding constraint.
4. **F3 baseline `run_a.json` is stale** (script edited 3 min after the baseline
   was written); the valid F3 control is the attn arm run here. Attn F3 P1 is
   bit-deterministic across reruns.
5. **E4 ran the right-roll edge only** (Step-1 showed L=R to 3 decimals, so the
   left edge adds no information for the ablation question). SUPINE→SIT was not
   re-run: it fails at the attention-free existence stage.
6. The attn F3 determinism rerun was killed mid-P2 by runtime memory pressure;
   P1 bit-identity was established before the kill. No results were fabricated:
   every number above comes from the JSON/NPZ artefacts listed below.
7. Wall times are VM-specific (2 CPUs, contended BLAS); tick counts and costs
   are the portable quantities.

---

## 6. Artefacts (all in this workdir)

- `run_e4.py` — edge driver (arms + adapted `run_edge`); `f3_e4.py` — F3 wrapper
- `edge_e4_attn_R.json`, `traj_e4_attn_R.npz`, `e4_attn.log` — control arm
- `edge_e4_uniform_R.json`, `traj_e4_uniform_R.npz`, `e4_uniform.log` — uniform arm
- `edge_e4_permuted_R.json`, `traj_e4_permuted_R.npz`, `e4_permuted.log` — permuted arm (seed 7)
- `results_e4.json` — incremental per-arm summaries (edge)
- `run_f3_e4_attn.json/.npz`, `run_f3_e4_uniform.json/.npz`, `run_f3_e4_permuted.json/.npz` — F3 arms
- `q_supine_settled.npy` — read-only input copy from `muse/results/f3/` (F3 script reads its start from ROOT)

Nothing was committed (coordinator handles git). `muse/exp_e3_statics_gap/`,
`codex/`, and `sim/` semantics were not touched — all E4 pieces are harness-local
wrappers, documented in §1.
