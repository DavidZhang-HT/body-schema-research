# E2 Step 1 — clean re-run of scheme A's get-up oracle: METHODS

## What is being tested

The committed `QUASI_STATIC_EXHAUSTION` verdict
(`muse/results/getup_search/verdict.md`) was proven byte-identical to the OLD
oracle's `run1_restart4/verdict.md` — i.e. the verdict on record was produced
by the old oracle, not the current one. Step 1 re-runs three edges with the
**current** oracle (`evaluate_edge` = `KeyframeImaginer.chain` existence stage
+ region-to-target tracking phases + relaxation ladder including
`no_hold_ramp`), from a **clean** SUPINE start:

- clean = `mode_check(SUPINE)` True, statics LP feasible, `selfd >= -1 mm`
  (the old run's stuck state self-penetrated 35 mm — this run asserts
  `selfd >= -1 mm` at the start AND flags every per-tick violation in the
  trajectory; the run is aborted/recorded as violated if any occurs)
- edges: SUPINE→SIDE (right roll), SUPINE→SIDE_L (left roll), SUPINE→SIT
- falsifier: SUPINE→SIDE completing ⇒ committed verdict was an artefact.
  Stalling ⇒ record exactly where (EXISTENCE_GAP vs tracking stall, tick,
  residual terms, selfd trace, trunk rotation achieved) as the baseline for
  Steps 2–3.

## Exact commands

```bash
VENV=/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python
cd /home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/step1
$VENV check_start.py     # clean-start pre-check (fast)
$VENV check_mirror.py    # mirror exactness pre-check (fast)
$VENV run_step1.py supine_to_side_R                 # smoke: single edge
$VENV run_step1.py supine_to_side_L supine_to_sit   # remaining edges
# (no args = all three edges; argv filters the edge list — harness-local addition)
```

Outputs stream into the same directory: `edge_<tag>.json` (full EdgeResult +
harness block), `traj_<tag>.npz` (q trajectory, per-tick selfd/pen/rho/cost,
imagined keyframe chain), `results.json` (rewritten after every edge),
`q_supine_clean.npy` / `q_supine_clean_mirrored.npy`.

Wall time ≈ 14 min/edge on this VM (qp 0.7–1.5 s per MPC tick,
STALL_TICKS=60, up to 4 cold restarts, ladder 6 families × 60 ticks).
90-min-per-edge kill budget was never approached.

## Clean-start construction

Start q: `muse/results/getup_search/roll_probe_q_0.npy` (the committed roll
probe's 0° config). Verified by `check_start.py` under the real `Schema`:
`mode_check(SUPINE)` True, `classify` = ['SUPINE'], feasible, rho 0.0952,
selfd +4.5 mm, pen -0.0 mm, touching {upper_trunk, lower_back, L_foot,
R_foot}, trunk R[2] = [0.9949, 0.0000, 0.101].

## Left-roll handling (SIDE is right-only in the committed vocabulary)

Committed `SIDE` = right side down (`R[2,1] >= 0.7`, body +y up). The left
roll is tested via two harness-local pieces (sim/ untouched):

1. `mirror.py::mirror_q` — exact sagittal-plane (x−z) mirror of the 20-DoF
   pi+ config: base `(M R M, M p)` with `M = diag(1,-1,1)` (quaternion via
   `mju_quat2Mat`/`mju_mat2Quat`, canonicalised w≥0); joints L↔R swapped with
   per-axis sign (roll/yaw −1, pitch +1; axes verified from
   `pi_plus_20dof.xml`). `check_mirror.py` verified: rho/feasible/selfd
   identical, region heights exactly L/R-swapped, trunk R[2,1] sign flip
   (+6e-6 → −6e-6), still classifies SUPINE, selfd +4.5 mm ≥ −1 mm.
   Caveat (honest): the statics LP returns a different *degenerate* load
   distribution for the mirrored config (orig L_foot 0.111 / R_foot 0.075 →
   mirr identical, not swapped) while rho is identical to 1e-9 — the LP
   optimum is degenerate and the solver picks a different vertex on mirrored
   numerics. Geometry is exactly mirrored; the load split is a solver
   artefact, not a physics asymmetry. (Because of this, `check_mirror.py`
   reports MIRROR EXACT: False and does NOT save the file; `run_step1.py`
   re-derives and saves it anyway after its own `check_start` asserts pass.)
2. `SIDE_L` mode registered locally in `run_step1.py` (D1): exact mirror of
   `SIDE` (`orient = R[2,1] <= -0.7`, `go.ORIENT_TARGET["SIDE_L"] = (1,
   -90.0)`), appended to `getup_modes.MODES` / `MODE` before `OracleContext`
   is built, so `edge_rules()` sees `("SUPINE","SIDE_L")` as a candidate and
   `chain` / `compile_tracking` / `mode_check` all resolve it.

The imagined keyframes for SIDE_L came out as the exact mirror of SIDE_R
(elev −15/−30/−45, loads L/R-swapped, same rhos) — end-to-end confirmation
the mirror is correct.

## P-F3 bypass (SUPINE→SIT only)

Committed `edge_rules()`: `("SUPINE","SIT")` is pruned by P-F3 (LYING_UP →
UPRIGHT, the F3 direct-trunk-lift evidence). Step 1 evaluates it anyway: in
`make_ctx`, after asserting the committed rule really is P-F3, the rule is
replaced by `EdgeRule("SUPINE","SIT","candidate")` on the *context-local*
rules dict (D2). Committed `sim/` files untouched. The F3 evidence itself is
not re-litigated here; the edge is run to see what the *current* oracle says.

## Ladder config

D3: `go.LADDER_TICKS = 60` (committed 15). Asserted at import:
`no_hold_ramp ∈ RELAX_LADDER`, all `SUBGOAL_TICKS ≥ 60` (300/400/300),
`TRACK_TICKS = 250 ≥ 60`. The ladder runs each constraint family relaxed for
60 MPC ticks from the failure config and reports which family unlocks
progress (`binding`). Families: no_selfcol, no_jlim, big_trust, rho_cap_inf,
mu_x2, no_hold_ramp (`F_hold=F_s=1e9` — contact state machine off: loaded
points may lift/slide, load may jump between points).

## Log capture (D4)

`go.run_phase` is wrapped (module-attribute replacement, so the oracle's
internal calls hit the wrapper) to capture the live `logs` list; `run_edge`
then extracts per-tick q / cert (selfd, pen, rho) / task cost, re-runs the
imagine stage deterministically to store the keyframe chain, and records
trunk R[2] start→end. No behavioural change to the oracle.

## NUMBERS — per-edge results

All three edges INFEASIBLE. None completed. Wall times below; the "≈2
min/edge" estimate from old attempts was wrong for this oracle on this VM
(qp 0.7–1.5 s/tick): budget 90 min/edge was never approached.

| edge | status | where it stopped | ticks | wall |
|---|---|---|---|---|
| SUPINE→SIDE (R) | INFEASIBLE | CONNECTIVITY: T1_track[elev +15deg] STALLED (tick 60 of last attempt; 202 ticks, 2 cold restarts, 362.5 s tracking) → ladder binding = **no_hold_ramp** | 202 | 821.3 s |
| SUPINE→SIDE_L (L) | INFEASIBLE | CONNECTIVITY: T1_track[elev −15deg] STALLED, same signature (202 ticks, 2 restarts, 213.9 s) → ladder binding = **no_hold_ramp** | 202 | 668.3 s |
| SUPINE→SIT | INFEASIBLE | **EXISTENCE_GAP** at [elev +13deg], frac 0.083: no certified quasi-static keyframe (all 6 CoM candidates: cert ok, rho 0.095, but hard IK residual lsq ≈ 0.28 > HARD_TOL) | 0 | 83.3 s |

### SUPINE→SIDE (both directions — identical to 3 decimals)

- **Existence: OK.** `KeyframeImaginer.chain` certified 3 keyframes:
  R: elev +15/+30/+45 deg, rho 0.093/0.083/0.142, loads e.g.
  {L_foot 0.223, R_foot 0.271, upper_trunk 0.444, lower_back 0.061};
  L: exact mirror (elev −15/−30/−45, loads L/R-swapped, same rhos).
- **Tracking: stalled at the FIRST sub-goal.** T1 cost 38.65 → 21.1, then
  flat for 60 ticks × 3 attempts (STALL_TICKS=60, TOL=0.01; cold-restart
  rule fired twice, third attempt gain < 2% → stop).
- **Residual task terms at stall** (R): com **10.04** (dominant),
  upper_trunk_land 3.70, posture 3.33, trunk_left(+y) 2.06, trunk_head(+z)
  2.00, lower_back/L_foot/R_foot_land 0.0. (L: 10.02 / 3.70 / 3.33 / 2.06 /
  2.01 — same.)
- **Relaxation ladder** (60 ticks/family from the stall config; R):
  full 21.14→20.79 ok; no_selfcol / no_jlim / big_trust / rho_cap_inf /
  mu_x2: no unlock; **no_hold_ramp 21.14→13.07 (QP_INFEASIBLE at the end,
  but far the largest drop)** → verdict "constraint-bound: no_hold_ramp".
  (L: no_hold_ramp 21.13→13.06, same.)
- **Trunk rotation achieved: 0°.** R[2,1] stayed 0.0001 (R) / −0.0001 (L)
  for all 202 ticks — max |roll| 0.49°. What the MPC did instead: pitched
  the trunk up (chest_z 0.99→0.85, head_z 0.10→0.52, head elevation
  7°→32°) — the F3 trunk-lift motion, not a roll. Unmet predicate:
  `left_z >= 0.7` (actual left_z +0.00).
- **Cleanliness:** 0 cert violations during tracking; end cert feasible,
  rho 0.090, loads {L_foot 0.219, R_foot 0.089, lower_back 0.691};
  selfd trace +3.28…+4.96 mm (mean 3.71), **0 ticks below −1 mm** —
  the clean-start assertion held for the whole trajectory. peak_rho 0.096
  (never torque-bound).

### SUPINE→SIT (P-F3 bypassed)

- Failed before any tracking tick: imagine stage could not certify even the
  first keyframe toward SIT. At frac 0.083 (elev +13° toward head-up), all
  six CoM candidates return cert-feasible (rho 0.095, `why: 'ok'`) but the
  hard IK residual stays at lsq ≈ 0.28 (tolerance HARD_TOL not met): the
  trunk cannot be pitched 13° toward vertical quasi-statically while the
  back is the support, with feet kept planted. Trunk R[2] unchanged
  ([0.9949, 0.0, 0.101]).
- This is the current oracle independently reproducing the *shape* of the
  F3 diagnosis (trunk-lift structural block), now at the keyframe-imagination
  level rather than the MPC-tracking level. It is still labelled NOT a proof
  of non-existence (a different keyframe parametrisation could in principle
  succeed; the oracle only reports its own search).

## Step-1 falsification verdict

**The fixed falsifier did NOT trigger** — SUPINE→SIDE did not complete, so
the committed `QUASI_STATIC_EXHAUSTION` verdict is not overturned as an
artefact by Step 1.

But the re-run **moved the failure** and sharpened it, which is the honest
baseline for Steps 2–3:

1. **Existence ≠ the problem for roll.** The current oracle *can* imagine a
   certified quasi-static keyframe chain from clean supine to side-lying
   (3 keyframes, rho ≤ 0.15). The old oracle's exhaustion verdict conflated
   stages; the new split is: existence OK / connectivity fails.
2. **The connectivity failure is precisely located**: T1 (first 15° of
   roll), STALLED with the CoM task term dominant (10.0 of 21.1 residual)
   and the *only* binding ladder family `no_hold_ramp` — i.e. the
   quasi-static contact-state machine (loaded points must unload gradually,
   load may not jump) is what blocks the roll, not torque (peak rho 0.096),
   not self-collision, joint limits, trust region, or friction. The MPC
   substitutes a trunk pitch-up (0° roll achieved) and stalls.
3. **SIT reproduces F3's shape one level earlier**: EXISTENCE_GAP at
   +13° head-up — no certified keyframe, IK residual stuck at 0.28 while
   statics stays feasible.
4. **Mirror symmetry confirmed end-to-end**: L edge reproduces R to 3
   decimals in residuals, binding family, tick count, and rotation (0°) —
   the left-roll test is valid and equally stuck.

Step 2–3 implication: a momentum/dynamic strategy must specifically defeat
the contact-state-machine binding (load transfer without quasi-static
unloading) to roll; static re-tuning of attention prices is not expected to
unstick T1 (recorded as the E4 prior in SETUP_FOR_E4.md).

## Honest caveats

- The quasi-static MPC is deterministic but slow; STALL_TICKS=60 and the
  cold-restart rule (≤4 restarts, ≥2% gain to continue) bound each sub-goal.
- `no_hold_ramp` hitting QP_INFEASIBLE while dropping cost means the relaxed
  problem is itself ill-posed at the failure config — "binding" is the
  oracle's operational label, not a physical proof that contact switching is
  *the* missing capability.
- Failure is labelled "NOT a proof of non-existence" by the oracle itself
  (see `evaluate_edge` ladder verdict string).
- SIT edge: P-F3 bypass is a deliberate deviation; the EXISTENCE_GAP there
  reports the current oracle's behaviour, it does not re-litigate F3's evidence.
