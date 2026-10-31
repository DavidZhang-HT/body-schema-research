# E2 — Momentum falsification for SUPINE→SIDE: full report (Steps 1–3)

## Bottom line

- **Step 1** (quasi-static oracle, clean re-run): all three edges INFEASIBLE —
  SUPINE→SIDE_R/L stall at the first 15° sub-goal (binding ladder family:
  `no_hold_ramp`), SUPINE→SIT hits EXISTENCE_GAP at +13° head-up.
- **Step 2** (dynamic positive control, MuJoCo forward dynamics): **6/6 SUCCESS**.
  A momentum-based roll (wind-up counter-swing → ballistic swing over the top,
  peak KE 1.8–4.9 J) reaches SIDE and **settles there** — the *full*
  `mode_check` (including the quasi-static statics LP) passes with
  rho 0.062–0.088 (cap 0.8).
- **Step 3** (momentum ablation, KE cap ≤0.005 J enforced at trajectory end):
  **6/6 FAIL**. The capped CEM cannot find any slow motion that rolls;
  best post-settle roll progress ≈ 0.000 on all starts.
- **Falsification verdicts:**
  - "Momentum required" is **SUPPORTED** — Step 2 succeeds AND Step 3 fails,
    across 6 ≥ 5 starts.
  - The "Step-1 failure is the solver's" falsifier does **NOT** trigger —
    Step 3 did not succeed, so the quasi-static failure is not shown to be a
    solver artefact. The evidence instead says the edge *requires* momentum.

Workdir: `muse/exp_e2_momentum_test/steps23/` (this report: `EXP_E2_REPORT.md`).
Step-1 artefacts: `../step1/` (`STEP1_NOTES.md`, `edge_*.json`, `traj_*.npz`).
Per-start JSONs: `step2/step2_<tag>.json`, `step3/step3_<tag>.json`;
trajectories: `step2/traj_step2_<tag>.npz`, `step3/traj_step3_<tag>.npz`
(qpos/qvel at 50 Hz, phase1 + phase2). No git commits (coordinator handles it).

---

## Step 1 — numbers folded in (from `step1/STEP1_NOTES.md`, not re-run)

Clean SUPINE start: `mode_check(SUPINE)` True, statics feasible, rho 0.0952,
selfd +4.5 mm. All asserted at start and per-tick (0 violations).

| edge | status | where it stopped |
|---|---|---|
| SUPINE→SIDE (R) | INFEASIBLE | CONNECTIVITY: T1_track[elev+15°] STALLED after 202 ticks / 2 cold restarts → ladder binding = **no_hold_ramp**; trunk roll achieved 0° (MPC pitched trunk up instead); selfd +3.3…+5.0 mm, 0 violations, peak rho 0.096 |
| SUPINE→SIDE_L (L) | INFEASIBLE | same signature to 3 decimals (202 ticks, binding no_hold_ramp, 0° roll) |
| SUPINE→SIT | INFEASIBLE | EXISTENCE_GAP at elev+13°: hard IK residual lsq ≈ 0.28 > HARD_TOL (statics feasible, rho 0.095) |

Step-1 falsifier (SUPINE→SIDE completes ⇒ committed verdict was an artefact)
did NOT trigger. Sharpened baseline for Steps 2–3: existence OK for roll,
connectivity fails at the contact-state machine (`no_hold_ramp` binds; only
that ladder family unlocks cost 21.1→13.1).

---

## Methods

### Plant (verified identical to scheme A's)

`common.py::Plant`: `model.PiPlus(NOMINAL, timestep=0.001, shell="full")` with
our own `mujoco.mj_step` loop (model.py's builder never steps). Clean nominal
physics: no sensor noise, no actuation delay, no q0 offsets (unlike
`RealRobot`). Verified byte-level against `getup_schema.Schema` (the object
Step 1's oracle reasons over): total mass 11.0103 kg both, timestep 0.001
both, 29 geoms both, body masses identical, 27 qpos / 20 actuators with
matching names. The ONLY difference from Step 1 is forward dynamics vs
kinematics+statics-LP. Torque limits: 20 Nm legs / 10 Nm arms (TAU_MAX);
leg Coulomb friction 0.2 Nm (nominal); floor mu 1.0; implicitfast integrator,
pyramidal cone.

Mode evaluation reuses the committed vocabulary: `Schema.kin` +
`statics_lp` + `mode_check` from `sim/getup_modes.py`, with `SIDE_L`
registered harness-locally (exact mirror of SIDE, as in step 1 D1).
Per the task, **success = the geometric mode predicate** (req-touch +
allowed/forbidden + orientation, via `mode_check` with a trivially-feasible
cert), and the real statics LP is additionally computed and reported for
every success.

### Step 2 — controller (task choice (b): sampling MPC / CEM)

CEM over PD-tracked joint-waypoint trajectories ("sampling MPC with
trajectory-parameterized actions"):

- **Phase 1 (scripted, quasi-static, 2.0 s):** move the down-side arm forward
  across the chest (shoulder_pitch → −1.5, elbow → 0.8; right arm for right
  roll, mirrored for left). Peak KE 0.0072 J. *Why scripted:* see "stability
  finding" below — this is a necessary preparatory motion, not the roll.
- **Phase 2 (searched, 4.0 s, 6 waypoints):** CEM (N=96, elite=12, iters=8,
  σ0=0.6, Δ=1.0 rad) over waypoint deltas, executed by joint PD
  (Kp 50 legs / 15 arms, Kd 2.5 / 0.8, τ clipped to TAU_MAX). The down-side
  arm is HELD at its Phase-1 pose (masked out of the search) so the search
  cannot undo the stabilizer. **Cost includes a 1.0-s zero-torque settle
  simulated inside every sample:**
  `cost = −50·p_settle −10·p_end −5·p_max + 0.01·mean|τ| + 500·selfpen`,
  where `p = sign·R[2,1]`. The true objective (end up settled in SIDE) is
  optimized directly. Elitism (best kept in population).
- **Success test:** 1.5-s zero-torque settle from trajectory end; predicate
  must hold afterwards; full `mode_check` (real statics LP) reported.

### The stability finding (why Phase 1 exists)

A 90°-roll placement with the clean-start limb pose (arms alongside) is
**unstable**: gentle placement at 90° topples to PRONE within 0.6 s
(`diag_side_stable2.py`); a drop test from 80° falls back to supine, from
90°+ goes prone. Cause: the down-side arm is trapped under the trunk and
acts as a rocker. With the down-side arm brought forward (`R_arm_forward`),
the same 90° placement **rests stably** (3-s settle: predicate True,
R21=+0.906). Hence SIDE is a physical attractor *only* with the arm clear —
the quasi-static oracle's vocabulary ("limbs anywhere") admits it, but the
MPC never moved the arm there. Phase 1 puts it there quasi-statically; the
*roll itself* (Phase 2) is discovered by CEM.

### Step 3 — momentum ablation (KE cap ENFORCED)

Same two-phase structure, Phase 2 replaced by the capped search:
- Slower: T=6.0 s, 8 waypoints, Δ=0.6 rad (vs 1.0), σ0=0.12 (vs 0.6).
- Soft: `W_KE=500` on peak KE in the cost (rebalanced so the settle reward
  still dominates for genuinely slow successes).
- **Hard (enforced, not just logged):**
  1. in-CEM rejection: any sample with peak KE > 0.03 J gets cost = INF;
  2. after CEM, the best trajectory is executed and `ke_end ≤ 0.005 J` is
     asserted — on violation the waypoint deltas are shrunk ×0.5 and
     re-executed (up to 3 tries); persistent violation → start marked
     `FAIL_KE_CAP`.
- Success criterion identical to Step 2 (settle + predicate + full check).

### Starts (6, all verified SUPINE with real cert before each step)

| tag | q | dst | CEM seed |
|---|---|---|---|
| clean_R_a | `q_supine_clean.npy` (Step-1 clean start) | SIDE | 11 |
| clean_R_b | same | SIDE | 22 |
| mirr_L_a | sagittal mirror (`mirror_q`) | SIDE_L | 11 |
| mirr_L_b | same | SIDE_L | 22 |
| noisy_R | clean + N(0,0.04) joint noise (rng 101) | SIDE | 33 |
| mirr_yaw_L | mirrored + 0.15 rad base yaw | SIDE_L | 33 |

All six: `mode_check(SUPINE)` True, rho 0.0952, selfd +4.5 mm at t=0
(`step2/starts.json`, `step3/starts.json`).

Exact commands:
```bash
VENV=/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python
cd muse/exp_e2_momentum_test/steps23
$VENV run_direct.py --step 2 --all --workers 1   # ~31 min (291–312 s/start)
$VENV run_direct.py --step 3 --all --workers 1   # ~40 min (396–409 s/start)
```
(`--workers 2` OOM-killed on this 8 GB VM — two forked workers × ~1.1 GB;
sequential is the documented configuration.)

---

## Numbers

### Step 2 — per-start (dynamic, no cap): 6/6 SUCCESS

| start | status | p_settle | predicate | full (statics LP) | rho | peak KE [J] | ke_end [J] | peak base-speed [m/s] | peak joint-speed [rad/s] | max self-pen [mm] |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_R_a | SUCCESS | +0.948 | True | True | 0.076 | 3.78 | 1.672 | 0.56 | 3.42 | 1.6 |
| clean_R_b | SUCCESS | +0.893 | True | True | 0.062 | 2.20 | 0.205 | 0.51 | 2.17 | 1.9 |
| mirr_L_a | SUCCESS | +0.940 | True | True | 0.079 | 2.89 | 0.062 | 0.53 | 2.86 | 1.1 |
| mirr_L_b | SUCCESS | +0.948 | True | True | 0.074 | 1.78 | 0.166 | 0.39 | 3.32 | 1.5 |
| noisy_R | SUCCESS | +0.930 | True | True | 0.069 | 1.83 | 0.037 | 0.39 | 3.61 | 1.7 |
| mirr_yaw_L | SUCCESS | +0.943 | True | True | 0.088 | 4.88 | 0.150 | 0.62 | 4.46 | 1.7 |

Phase-1 (all starts): peak KE 0.0072 J (noisy_R: 0.20 J — larger PD errors
from the perturbed start; still quasi-static, no cap in Step 2).
Per-iteration CEM traces are in `step2/step2_<tag>.json` (`cem_best` per
iteration: p_settle, peak KE); full 50-Hz qpos/qvel in
`step2/traj_step2_<tag>.npz`.

**What the successful motion does** (clean_R_a trajectory analysis,
`analyze_traj.py`): a **wind-up counter-swing** — R21 first rocks LEFT to
−0.40 (t≈1 s), returns through 0 (t≈2 s), then swings hard RIGHT, crossing
0.7 at t≈3.4 s and settling at +0.96. Prime movers: legs (hip_pitch/calf ROM
1.0–1.6 rad, peak speeds 3–4.7 rad/s). It is *not* a quasi-static lean: the
body goes ballistic past ~60° (limbs lose ground contact; the 10.6 J
quasi-static work to lift the CoM from 0.094 m to 0.192 m at 90° is done
impulsively, peak KE 1.8–4.9 J).

### Step 3 — per-start (KE cap ≤0.005 J enforced): 6/6 FAIL_STALL

| start | status | p_settle | predicate | peak KE [J] | ke_end [J] | cap ok | shrinks |
|---|---|---|---|---|---|---|---|
| clean_R_a | FAIL_STALL | −0.000 | False | 0.0034 | 0.0031 | True | 0 |
| clean_R_b | FAIL_STALL | −0.000 | False | 0.0039 | 0.0026 | True | 0 |
| mirr_L_a | FAIL_STALL | +0.000 | False | 0.0040 | 0.0037 | True | 0 |
| mirr_L_b | FAIL_STALL | −0.001 | False | 0.0037 | 0.0026 | True | 0 |
| noisy_R | FAIL_STALL | −0.000 | False | 0.0043 | 0.0027 | True | 0 |
| mirr_yaw_L | FAIL_STALL | −0.000 | False | 0.0037 | 0.0033 | True | 0 |

The cap was never even *triggered* (shrinks=0): the in-CEM peak rejection
(0.03 J) + W_KE shaping kept every executed trajectory at 0.003–0.004 J,
comfortably under the 0.005 J cap — and none of them rolled (best post-settle
progress ≈ 0.000; the CEM converged to near-stationary motions). Details in
`step3/step3_<tag>.json`; trajectories in `step3/traj_step3_<tag>.npz`.

**Sanity check that slow rolling is geometrically hard** (not just CEM
weakness): a hand-designed quasi-static weight-shift (Phase 1, then 8 s
moving all movable limb mass to the right: `diag_weightshift.py`) shifts the
CoM only 0.028 m toward the support edge (needs ≈0.09 m) at peak KE 0.0038 J;
R21 moves 0.000→0.019. The limbs are too light and joint-limited to tip the
body by CoM shifting alone.

---

## Falsification verdicts

**Falsifier 1** — *"If Step 3 succeeds where Step 1 failed → the Step-1
failure is the QUASI-STATIC SOLVER's, not physics."*
**Does NOT trigger.** Step 3 failed on all 6 starts. The quasi-static
oracle's failure is not shown to be a solver artefact.

**Falsifier 2** — *"'Momentum required' is supported ONLY if Step 2 succeeds
AND Step 3 fails, across ≥5 starts."*
**SUPPORTED.** Step 2: 6/6 SUCCESS (dynamic, peak KE 1.8–4.9 J, full
mode_check passes). Step 3: 6/6 FAIL (capped at 0.005 J, peak KE
0.003–0.004 J, zero roll progress). 6 ≥ 5. The SUPINE→SIDE edge on this plant
requires momentum: no slow (KE ≤ 0.005 J) strategy found by the same
optimizer that reliably finds dynamic ones, and the obvious quasi-static
route (CoM weight-shift) is geometrically infeasible (0.028 m achievable vs
≈0.09 m needed).

**What this says about Step 1.** The oracle was not wrong that *its*
quasi-static search fails — but the failure is deeper than the contact-state
machine. Even with the contact machine removed (real MuJoCo contact) and the
arm-placement problem solved (Phase 1), the edge still cannot be traversed
slowly: the 10.6 J lift over the roll hump cannot be delivered quasi-statically
by this body's limbs (weight-shift infeasible; no slow push discovered in
6×8×96 capped samples). Step 1's `no_hold_ramp` binding is real, but it is
not the whole story — the edge genuinely needs a ballistic phase.

---

## Honest caveats

1. **Phase 1 is hand-designed.** The CEM discovers the roll, but the
   down-side-arm-forward preparatory motion comes from my stability analysis,
   not from search. A pure tabula-rasa CEM (no Phase 1) reaches 84° rock but
   cannot settle (the trapped arm topples it to prone). The positive control
   is therefore "a dynamic strategy exists", not "an unbiased search found
   it unaided".
2. **Step-3 non-exhaustiveness.** The capped CEM is a genuine but bounded
   search (8 iters × 96 samples × 6 starts, Δ=0.6, 6-s horizon). A cleverer
   slow strategy (e.g. a multi-phase "bridge-and-roll" using sustained leg
   extension to do the 10.6 J of work quasi-statically) might exist and was
   not exhaustively ruled out. The weight-shift infeasibility + uniform
   6/6 CEM failure make "momentum required" the best-supported reading, not
   a proof.
3. **n = 6 starts.** Two optimizer seeds per direction + two perturbation
   variants. Covers L/R symmetry and mild robustness, not a distribution.
4. **Self-penetration 1.1–1.9 mm** during Step-2 dynamic contact (MuJoCo
   contact distance; Step-1's schema bound was −1 mm). Minor soft-contact
   interpenetration; did not affect the mode predicates. Reported, not hidden.
5. **Plant vs scheme B.** Step 2/3 run on scheme A's 20-DoF primitive-shell
   plant (verified identical to the Schema object). Any comparison to scheme
   B's plant must account for plant differences; this experiment does not
   address scheme B.
6. **Compute.** ~31 min (Step 2) + ~40 min (Step 3) wall on 2 CPUs, MuJoCo
   3.17, dt=0.001, sequential (the 2-worker pool OOMs on 8 GB).
7. **The settle is part of the method.** Success requires the 1.5-s
   zero-torque settle to hold the predicate — a transient fly-through does
   not count. All six Step-2 successes additionally pass the *full*
   mode_check including the statics LP (rho 0.062–0.088 ≤ 0.8), i.e. the
   reached SIDE configurations are quasi-statically certifiable *as
   configurations* — they just cannot be *reached* quasi-statically.

## Files

- `steps23/EXP_E2_REPORT.md` — this report
- `steps23/common.py` — plant, mode eval, starts, rollout
- `steps23/step2.py` — CEM block controller (dev history; superseded by run_direct.py)
- `steps23/run_direct.py` — Steps 2+3 direct trajectory optimization (`--step 2/3`)
- `steps23/step2/` — `starts.json`, `results.json`, `step2_<tag>.json`,
  `traj_step2_<tag>.npz` (×6)
- `steps23/step3/` — same for Step 3 (×6)
- `steps23/diag_*.py`, `probe_push.py`, `smoke2.py`, `analyze_traj.py` —
  diagnostics (reach, stability, energy landscape, weight-shift, trajectory)
- `steps23/*.log` — run logs
