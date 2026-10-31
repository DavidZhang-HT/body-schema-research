# Body-schema self-exploration gait study: plan as implemented

Status as of 2026-10-30 (run 2). Everything runs headless on CPU with `sim/.venv`.
Phase 1 is complete. Phases 2 and 4 are implemented in a simplified form. Phase 3 only has its per-tick feasibility projection; the outer gait-parameter optimisation is **not finished**.

## Platform
- **Robot**: the real HighTorque **pi+** model from the official `HT_Robot_URDF` release (`assets/pi_plus_24dof`, URDF has 24 DoF; MJCF has 20 actuated DoF). No proxy model is used.
  - 12 leg joints at 20 Nm and 8 arm joints at 10 Nm (URDF value; the MJCF says 20).
  - Wrist and head joints are frozen.
  - Meshes are replaced by primitive geoms. Sole boxes come from the STL bounding box (172×72 mm).
  - Bug fixed in the official MJCF: the free joint inherited motor friction, damping and armature (`sim/model.py`).
- **Sensors** (`RealRobot.observe`):
  - joint encoders (2e-4 rad noise; ankle backlash hysteresis in the plant)
  - motor-current torque (2% + 0.05 Nm)
  - IMU quaternion, gyro and specific force
  - base state estimator (2 mm, 0.02 m/s)
  - **no foot F/T sensors**, as on the real pi+. Contact and GRF are estimated from joint torques.
- **Control**: 500 Hz QP whole-body controller (`sim/wbc.py`) on a 1 kHz MuJoCo plant.

## Phase 1: reference (done)
- Source: LAFAN1 `walk1_subject1.bvh` → pi+ joint space via scaled-keypoint IK (`sim/phase1_retarget.py`).
- Uses the 5 s window (2 strides, about 20° heading change). The 6.5 s window curved 56° and was rejected.
- Validation: sagittal mean correlation 0.96 against HighTorque's GMR reference. IK residual is 5.5 mm mean; no self-contact.
- Output: `data/ref_walk.npz` (`q_cycle`, 100×20).

## θ: body-schema parameters (`sim/scenarios.py`)

**"Real" plant θ\*, seed 0:**
- +0.8 kg on the trunk
- trunk CoM offset (+2.5, −1.2, +3.0) cm
- 12 ms actuation delay
- 0.6 Nm leg friction (3× nominal)
- contact time constant 35 ms (vs 20 ms)
- μ = 0.55
- 0.8° ankle backlash

Seeds 1 and 2 are random draws around these values.

**Identified vs. kept as prior:**

| Group | Parameters | How / why |
|---|---|---|
| Identified | `trunk_dm`, `trunk_dx`, `trunk_dy` | Quasi-static whole-body moment balance. Foot wrenches are rebuilt from leg torques using leg links only, so they do not depend on the uncertain trunk parameters. |
| Identified | `delay` | Lag between commanded and measured torque. This needs a rich torque signal, i.e. active excitation. |
| Prior | `trunk_dz` | Not observable in double-support statics. |
| Prior | `mu`, `contact_tc`, `backlash` | Would need slip or impact experiments, which are unsafe during calibration. |
| Prior (intended to identify, not done) | `frictionloss` | Its effect is confounded with the reconstructed contact wrench. |

## Phase 2: safe active postural exploration (`sim/experiment.py`)
- **Excitation** (double support only, `exploration_targets`):
  - multi-sine CoM sway in x at 0.3, 0.7 and 1.3 Hz
  - 0.25 Hz lateral load transfer of ±4.5 cm (about 80/20 single-foot light unloading)
  - squat of up to 2.4 cm at 0.4 Hz
  - trunk pitch ±3.4°
  - 1 s ramps in and out
- **Safety**: all motion is a CoM/DCM target inside the support polygon, and the QP still enforces the friction cone and torque limits.
- **Nominal vs. real**: `experiment.py phase2` runs the same excitation on both plants and compares CoM model error, leg torques, GRF and joint angles.
- **Passive ID baseline**: the same estimator, fed with quiet-standing data and then ungated online data (impacts included).

## Phase 3: dynamic feasibility projection (partial)
- **Done, every tick**: the QP solves for q̈ and 4-corner contact forces. Its cost is task tracking (CoM, trunk, swing foot, posture toward the retargeted reference) plus torque/control cost. Its constraints are:
  - floating-base rigid-body dynamics
  - pyramid friction cones (μ_ctrl = 0.6)
  - unilateral contact
  - torque limits
  - joint position limits over a 40 ms horizon
- **Self-collision**: handled only through a minimum foot separation (`w_min` = 0.102 m, the ankle-separation bound from Phase 1). The capsules are defined but not used as QP constraints.
- **Step level**: DCM planning plus capture-point footstep adjustment.
- **Not done**: the outer optimisation over `GaitParams` (trunk pitch, step width, cadence, double-support fraction, DCM gain) with a cost of reference deviation + control + uncertainty risk. Only a grid sweep stub exists (`sim/_sweep.py`), and it was not run to completion. These parameters are decision variables; nothing is hard-coded in the controller.
- **Not fixed**: the lateral fall at 0.10/0.14 m stride. Results use 0.06 m stride.

## Phase 4: surprise-driven three-layer loop
1. **Fast**: DCM feedback, capture-point step adjustment and the QP, every tick.
2. **Estimation**: contact-state estimate from torque-reconstructed GRF plus foot height, with hysteresis.
3. **Slow model adaptation** (`SurpriseGate`):
   - samples count only if they are in planned double support and at least 0.15 s after the last touchdown, so impact transients never enter the model
   - an update is considered only after at least 150 admissible samples
   - it is committed only if the CoM-offset change exceeds 2σ

   Passive ID commits every batch, ungated.

## Curriculum (each episode)
1. stand 1 s
2. double-support calibration 6 s (same budget for all conditions)
3. step in place, 6 steps
4. reference-guided forward walk, 10 steps at 0.06 m stride
   - floor μ drops ×0.75 at the start of the forward walk
   - lateral 18 N × 0.12 s push at step 5

## 3-way experiment
- Conditions: fixed nominal vs. passive ID vs. active ID, on seeds 0–2.
- Metrics:
  - fall rate
  - leg-joint RMS error to the reference
  - constraint-violation rate (torque saturation, QP fallback, stance-foot slip)
  - recovery time (DCM error < 3 cm held for 0.1 s)
  - θ-estimation error

Results are in `results.md`.


## Phase 2b: joint-zero active identification (added 2026-10-31)

Goal: identify per-joint **encoder zero offsets** `q0` of the 12 leg joints from proprioception + IMU only, using the same safe double-support exploration as Phase 2, with the same gating discipline as Phase 4. Results are in `results.md` → "Phase 2b".

### Truth model (`sim/scenarios.py`, `sim/model.py`)
- `BodyParams.q0` (20-vector, rad) is now part of θ. `scenarios.true_params_q0(seed)` = the base mismatched plant of that seed + leg zero offsets. Seed 0 uses fixed values (±1–3°); seeds 1–4 are drawn from a **separate RNG stream** (`3000 + seed`), so `true_params(seed)` is unchanged. Prior: `q0 ~ N(0, 2°²)` per joint (`Q0_PRIOR_STD_DEG`).
- Injection: `RealRobot.observe` reports `q_meas = q_true + q0`. The controller model subtracts its own estimate (`BodyModel.load_state`: `qpos = q_meas − q0_hat`), so `q0_hat = 0` reproduces the old controller exactly. Regression: the old conditions reproduce the published numbers bit-for-bit.
- The scenario is selected with `experiment.py <cond> <seed> 0.06 q0` → `results/q0/`.

### Estimator (`sim/zero_id.py`, `ZeroEstimator`)
- **Measurement model**: loop closure through the floor during double support. With both soles flat on a flat floor and no slip, the kinematic model evaluated at `q_meas − δ` must satisfy, per sample:
  - (G) **IMU gravity anchor**: each sole is level, `(R_bf^T g_b)_{x,y} = 0` (2 per foot, σ = 3e-3 rad). Only IMU roll/pitch is used, not yaw.
  - (R) relative foot rotation is a pure yaw, `R_LR = Rz(ψ)` (3, σ = 1e-3 rad).
  - (P) relative foot position lies in the floor plane, `R_bL^T (p_R − p_L) = (x, y, 0)` (3, σ = 1 mm).
  - `(x, y, ψ)` = unknown but constant foot placement, treated as nuisance parameters. The base position estimate is **deliberately not used**, since on a real robot it comes from leg kinematics (circular). A "jig" variant fixes `(x, y, ψ)` to the known stance as an upper-bound reference.
- **Excitation**: the Phase-2 double-support sway (multi-sine CoM x, lateral load transfer, ≤2.4 cm squat, trunk pitch). This changes the leg configuration, so the loop-closure Jacobian varies between samples. The baseline is quiet standing with the same 6 s budget.
- **Solver**: MAP Gauss–Newton over `[δ(12), x, y, ψ]` with prior `N(0, 2°²)`. One robust outlier pass, then a Schur complement of the nuisances → marginal information on δ → eigen-decomposition.

### Three gates (same discipline as the Phase-4 `SurpriseGate`)
1. **Surprise / sample gate + commit gate**:
   - A sample is admissible only in planned double support, ≥ 0.15 s after the last touchdown, with both feet estimated in contact and each carrying > 10 % of body weight, and quasi-static (|q̇_leg| < 1 rad/s, |ω| < 0.4 rad/s, ||a| − g| < 0.6). Samples are decimated ×5.
   - Commit: ≥ 150 samples are required. Only eigen-directions whose data information is ≥ κ = 16 × the prior information are written; everything else stays on the prior (0). The projected update must also pass an evidence test: χ² drop > 2 × (number of committed directions).
2. **τ_int inflation**: the information is divided per residual channel by the integrated autocorrelation time of the post-fit residuals (typically 5–200 samples), because 100 Hz samples of a slow sway are far from independent. A misfit scale `s² = max(1, χ²/dof)` is applied on top.
3. **Loop-consistency gate** (needs no offset knowledge): at δ = 0, the gravity, relative-roll/pitch and relative-height residuals of a flat, non-slipping stance vary only slowly with posture. A sample deviating from the robust median by > 1° or > 5 mm means a sole rolled onto an edge or slipped, and it is rejected.

### Identifiability verdict (noise-free rank analysis, `sim/zero_observability.py`)
The 12×12 marginal information has **4 strong directions**, the full-chain sums anchored by the IMU:
- L−R antisymmetric pitch-chain sum (rel. sv 1.0) and roll-chain sum (0.82);
- common-mode pitch-chain sum (0.23) and roll-chain sum (0.20). These are observable because the IMU fixes the absolute sole tilt.

The remaining 8 directions are **intra-chain redistributions** (e.g. calf vs. ankle pitch, hip vs. knee + ankle pitch, hip roll vs. ankle roll, thigh):
- **Differential (L−R antisymmetric) redistributions** (calf/ankle-pitch difference, hip/ankle-roll difference, thigh difference) are observable through the relative-foot-pose loop (rel. sv 2.5e-2 to 4.5e-2 with active sway).
- **Common-mode (symmetric) redistributions** of pitch and roll are in the near-nullspace (rel. sv 8e-4 to 4e-3 with active sway, 5e-5 to 2e-4 quiet). Changing them moves the base relative to both feet without changing sole tilt or relative foot pose, so the unmeasured base pose absorbs them. The estimator leaves them on the prior.
- Note: "common-mode" here means the symmetric *redistribution within the chain*. The common-mode *chain sum* is anchored by the IMU and is observable.
- Active sway raises the weak-direction singular values by about 3–15× over quiet standing. It does not remove the nullspace: with the base pose free, the 2–4 weakest common-mode directions stay below the commit threshold.

### Phase 2b follow-up: safe-motion observability scan (added 2026-10-31, `sim/zero_motion_scan.py`)
Question: can any **safe double-support** motion lift the common-mode intra-leg allocation directions above the 2° prior?
- **Tool** (`zero_motion_scan.py`, pure kinematics, deterministic): a candidate = base-pose trajectory (x, y, z, roll, pitch, yaw)(t) over two planted flat soles at a chosen stance → 6-DoF leg IK on the controller model → true q(t) and exact IMU gravity → `ZeroEstimator.residuals` Jacobian → marginal information over a 6 s, 100 Hz trajectory (nuisances Schur-complemented). Metric: marginal std along the **4 weakest directions of the real seed-0 Phase-2 calibration**, times a per-direction factor K = 5.7–8.9. K = deployed std / noiseless std for that baseline, and it covers τ_int and misfit inflation.
- **Safety envelope per sample**: IK exact, joint limits ≥ 2° margin, knee ≥ 10° (no singular straight leg), quasi-static torque < 20 Nm, each CoP ≥ 5 mm inside its sole, CoM ≥ 2 cm inside the support polygon, base tilt ≤ 6°, each foot ≥ 5 % of body weight.
- **Candidates**: squat depth (static −2…−10 cm, slow 0.2 Hz squat cycles), base fore/aft over the feet, stance width ±, operator-placed stagger 4–12 cm, toe-out, **trunk yaw twist about the planted feet** (±5…±20°, new), base roll, and combinations, each on top of the existing multi-sine sway.
- **Validation**: the winning class was added as an **option** (default off, so Phase-2 and every published number are unchanged).
  - `experiment.exploration_targets(..., yaw_twist, yaw_f, squat_cyc)`.
  - The WBC trunk-orientation target honours `stand["yaw"]`.
  - Modes `yaw15`, `yaw15sq`, `yaw25` in `zero_experiment.py`.
  - Full-simulation check: `zero_motion_validate.py`.
- **External-reference sizing** (`zero_external_ref.py`): the deployed Phase-2 information plus a few static measurements of the sole positions in the base frame (σ per axis, 1–3 postures) gives the posterior std along the weak directions.
- **Verdict**: (b) NO. The details and the minimal-reference proposal are in `results.md` → "Safe-motion observability scan".

### Phase 2b torque channel (added 2026-10-31, `sim/zero_torque.py`, `ZeroEstimator(torque=True)`)
Question: the loop-closure estimator reduces the 12 leg joint torques to nothing (and `SchemaEstimator` reduces them to 3 equations: weight + CoM x/y). Does the **full** torque vector, used as a residual, make the common-mode intra-leg directions W1–W4 observable?

**Mechanism.** In double support with both soles planted and no F/T sensors:

    M(q) q̈ + h(q, q̇; θ_trunk) = Sᵀ τ_j + J_Lᵀ F_L + J_Rᵀ F_R
    τ_j = τ_meas − b q̇ − a q̈ − μ tanh(q̇ / v0)          (MJCF damping b, armature a; Coulomb friction μ, v0 co-identified)
    q = q_meas − δ

- The 12 leg rows are solved **exactly** for the two unknown 6-D sole wrenches, `F_i = J_leg,iᵀ⁻¹ (h_leg,i − τ_j,i)` (J_leg,i is 6×6 and well conditioned at the ≥ 10° knee bend of the safe envelope). The other 6 equations, the floating-base rows, are the residual:

      r = h_base(q; θ) − J_L,baseᵀ F_L − J_R,baseᵀ F_R      (Fx Fy Fz [N], Mx My Mz [N m])

- Why δ enters: with a wrong calf-vs-ankle split, the knee sits at the wrong place relative to the hip and ankle. The 6 leg torques are then inverted with wrong lever arms into a wrong sole wrench. The dominant effect is on the **vertical force** (≈ 0.8–2.2 N per 1° along W1, depending on knee bend), plus Fx (≈ 0.3–0.5 N/°). It is *not* mainly a CoM-like moment (≈ 0.02 N m/°).
- Base velocity/acceleration (dynamic term only) come from the left-sole contact constraint with joint q̇ and a Savitzky–Golay q̈ from the full-rate q̇ stream. No base-state estimator is used, IMU only for roll/pitch. Noise model per joint: σ² = 0.05² + (0.02 τ)² + μ²(1 − tanh²)/3. The last term is the indeterminate static friction of a stalled joint. It is propagated exactly to the 6 residuals, `A = ∂r/∂τ_j = J_baseᵀ J_legᵀ⁻¹`, and each sample is Cholesky-whitened.
- Unknowns: δ (12), trunk (dm, dx, dy, dz), μ, v0, plus the loop-closure nuisances. Trunk/μ/v0 have Phase-1 priors (0.6 kg, 2/1.5/3 cm; μ 0.2 ± 0.3 N m; v0 0.03 ± 0.02 rad/s) and are Schur-complemented out like the nuisances. δ is data-only in the information.

**Friction (co-identified, not averaged).** The plant applies MuJoCo `frictionloss` μ to every leg joint. μ is one scalar, with a 12-per-joint variant scanned as a robustness check. The friction law's transition velocity v0 is **also co-identified**: a fixed v0 = 0.02 rad/s biases μ̂ low by 4–14 % (fitted on the true friction torques, 13–26 %). MuJoCo's soft friction constraint saturates only above ≈ 0.05 rad/s.

**Trunk-vs-offset disambiguation (`zero_torque_disamb.py`).**
- The trunk parameters have **posture-invariant** signatures: dm → Fz = 0.98 N / 0.1 kg (std over the trajectory 0.005), dx/dy → My/Mx = 0.42 N m/cm.
- The W1 signature is **posture-dependent**: Fz −1.6 N/° at the straightest knees vs −0.8 N/° at the most bent ones, plus Fx.
- In one static pose a +0.8 kg trunk mass leaks −0.43…+0.55° into W1. Over the Phase-2 trajectory the leak drops to ≤ 0.02°.

**Gates** (the published three, unchanged, plus two):
1. sample gate + commit gate (κ = 16, evidence test summed over both channels, each scaled by its own s² and mean τ_int);
2. τ_int + misfit inflation **per channel**;
3. loop-consistency gate;
4. **torque-consistency gate** (new, needs no offset knowledge): each torque residual at δ = 0 is detrended with a ±15-sample running median. A sample > 6 robust σ off in any channel is rejected.
   - A noise-normalised test against the global median was tried first. It rejected ≈ 50 % of normal samples, because the offset-induced posture dependence is itself many σ.
5. **channel-agreement veto** (new): on every direction the loop closure *alone* commits, the combined commit must agree with the LC-only commit within 3 LC-σ. If not, the torque channel is vetoed and the LC-only result is kept. This guarantees no regression on the LC-observable subspace when the torque model is wrong.

**Wiring**: `experiment.py` condition suffix `_zt` (e.g. `active_nodc_zt <seed> 0.06 q0`). Default `_z` and all published numbers are unchanged; the LC-only path is bit-identical (checked).

**Tools**: `zero_torque_record.py` (full-rate re-recording of the calibrations, bit-identical to the published runs), `zero_torque_scan.py` (the observability arbiter), `zero_torque_validate.py` (offline re-solve on the recordings, plus torque-gain robustness variants), `zero_torque_disamb.py`, `zero_torque_summary.py`. Results are in `results.md` → "Torque channel".

### Get-up proposal gap analysis (added 2026-10-31, analysis only)
- Documents: `analysis-getup.md` (English) and `report-cn-getup.md` (Chinese summary). They analyse the proposal "get up from lying via explore → learn contact/CoM mapping → imagine-and-invert → compose with minimum effort".
- **Verdict**: the logic matches philosophically. Structurally, the pipeline lacks the switching-contact layer and the compose/sequencing layer. Rough split: about 35–45 % reuse, about 20 % extension, about 40 % new.
- **Probe facts**:
  1. Limb capsules (contype/conaffinity 4/4) never collide with the floor (3/3), and there are no thigh or hand geoms. Arm, knee and thigh support therefore cannot form in the current plant.
  2. Contact forces are observable from τ as a net wrench per link for supine, sit, quadruped and squat contact sets. Two cases are rank-deficient: knee + flat foot on the same leg (by 2), and the distribution within a link (only the net wrench is observable).
- **Next step if pursued (first test, F3)**: check quasi-static get-up feasibility under the 10/20 Nm limits on the nominal model, with peak torque ratio ≤ 0.8.

### F3: attention-based explicit MPC get-up test (added 2026-10-31, implemented; verdict NOT_FOUND)
Implements the F3 falsification test from the get-up gap analysis: search for a quasi-statically feasible keyframe sequence supine→standing on the nominal model, peak |tau|/tau_max ≤ 0.8 per keyframe (20 N·m legs / 10 N·m arms). Uses an interpretable attention mechanism + explicit MPC; no black-box policy.

**G0 prerequisites** (`sim/model.py` edits; probe `sim/getup_g0_probe.py` → `results/f3/g0_probe.json`): simplified collision shells for thighs/hands/knees (visual meshes untouched; mesh-vs-model gap kept as "contact schema error" per convention), supine spawn/reset pose, get-up-appropriate success/failure criterion. Probe verified: supine contacts = base_link + both ankles (12 contacts, torque-channel readout max err 2.3% mg); long_sit = ankles + elbows + hip_roll_links (new limb geoms collide as intended); prone hands-and-feet = ankles + elbows.

**Attention** (`sim/getup_attention.py`): weights are functions of schema quantities, not learned scalars — support attention over 18 candidate contact regions (which links bear load) and motion attention over 20 joints per region-to-target task. Logged per MPC tick (`results/f3/run_a.npz`: a_cur, a_des, a_joint, alpha, loads, rho); visualized by `sim/getup_f3_plot.py` → `results/f3/attention_a.png` + `attention_a_table.md`. The weights track the certified static load share (phase entry: attention back 0.96 vs load 0.81; means: pelvis 0.21 vs 0.35, thighs 0.21 vs 0.29).

**Explicit MPC** (`sim/getup_mpc.py`, schema/statics in `sim/getup_schema.py`): min-effort keyframe-sequence QP subject to quasi-static equilibrium, unilateral + friction (mu 0.6) constraints on the attended contact set, joint/torque limits, self-collision avoidance. "Imagine" = solve for the next target; attention selects participating contacts/joints. Runner `sim/getup_f3_run.py`, diagnosis `sim/getup_f3_diag.py`.

**F3 verdict: NOT_FOUND** (`results/f3/run_a.json`, tag a, deterministic). 122 ticks, 45.8 s wall. Stalled at P1_head_up (tick 121, STALLED); only 2 keyframes certified; overall peak rho 0.181 — far below the 0.8 cap, so NOT torque-bound. Diagnosis ablations from the stall point (15 ticks each): no_selfcol 1.409/tilt 11.4°, no_jlim 1.739/12.5°, big_trust 1.466/11.6°, rho_cap→∞ 1.379/11.1°, mu×2 1.154/10.6° vs full 1.379/11.1° — relaxing every constraint class fails to unstick it. The stall is structural, not constraint-tuning: trunk_upright task cost stuck at 2.84, trunk tilt ~11–17°. Attention at stall is physically sensible (support: pelvis 0.645 + thighs; motion: shoulders 0.49, a pushing attempt) but cannot invent a feasible motion.

**Reading**: no quasi-static path found via the supine→head-lift→tuck→… strategy; the evidence points to momentum or a qualitatively different contact strategy (e.g. rolling to side/prone first) being necessary. The searched strategy is one strategy, not all strategies — the verdict falsifies "quasi-static imagine suffices via direct trunk lift", not get-up in general.

**Architecture correction (user, 2026-10-31 — ground truth, supersedes the head-first framing above):**
- "Head up first" was a misunderstanding: it was never a requirement. The F3 sequence (P1_head_up → …) was a human-imposed strategy choice, so the NOT_FOUND verdict is scoped to that choice, not to the user's intent.
- Body-part movements (head/hand/waist target reaching) are NOT for get-up; they are general motion-testing capabilities for learning the body schema (region-to-target relationships).
- Get-up is a SEPARATE task/requirement. Its role for region-to-target learning is decomposition: a complex task like get-up decomposes into region-to-target subproblems.
- The get-up strategy should be found AUTONOMOUSLY, and the process logic must be INTERPRETABLE — not end-to-end implicit/black-box. Rationale (user): interpretability's value is that the process is analyzable and optimizable. This is consistent with the existing design rules (interpretable attention, explicit MPC, falsification-first); it rules out replacing the strategy search with an end-to-end policy.
- Consequence for next steps: the "alternative contact strategy search" (e.g. roll-to-side first) must itself be an autonomous, interpretable search over contact-mode sequences (contact-mode graph + search, G4/G5), not a hand-supplied sequence. F3's stall diagnosis (binding constraint = trunk-flexion work unavailable) is the first input to that search, not a verdict on get-up.

### Autonomous interpretable get-up strategy search (added 2026-10-31, implemented; verdict QUASI_STATIC_EXHAUSTION)
Implements the user's corrected architecture: the robot finds its own contact-mode sequence from supine to standing, with every decision human-readable — no hand-supplied sequences, no end-to-end black-box policy. Region-to-target ("imagine") stays the separate low-level layer (`getup_mpc.py`, unchanged workhorse).

**Stack** (all in `sim/`):
- `getup_modes.py` — explicit contact-mode graph (layer 1, no solver): 8 modes (SUPINE/SIDE/PRONE/SIT/QUAD/HF/SQUAT/STAND) as human-readable predicates over body-schema regions (req/allowed/orient/extra/statics; touch threshold 5 mm; every mode additionally requires statics-LP feasibility with peak |tau|/tau_max ≤ 0.8). Candidate edges from explicit rules; every non-candidate ordered pair carries a stated pruning reason: P-TOPO (orientation classes not adjacent), P-SKIP (level 0→2 skips level 1 = the F3 trunk-lift sub-problem), P-DESC (level-monotone scope; documented incompleteness), P-F3 (F3 diagnosis as explicit evidence, never silent), P-START/P-GOAL (trivial pairs).
- `getup_oracle.py` — feasibility oracle per transition: compiles edge (mode A → mode B) into region-to-target sub-goals (G1_make: establish new contacts; G2_orient: reorient trunk), executed by `getup_mpc.py` with the interpretable attention. Returns EdgeResult records (status/cost/ticks/attention/diagnosis) — the only thing the sequencer sees.
- `getup_keyframes.py` — keyframe imagination layer: body-schema IK + exact statics certificate (no learning, deterministic).
- `getup_search.py` — A* sequencer over the graph. Documented edge cost: c(A→B) = 1.0 (per switch) + Σ mean_j(tau_j/tau_max_j)² (effort) + peak_rho/0.8 (margin) + 0.5·ticks/100 (length); heuristic h(A) = hops to STAND over candidate edges (admissible). Logs expanded nodes, per-edge cost breakdowns, pruned branches with reasons.
- `getup_roll_probe.py` — roll-angle probe (0–90°) used during development.

**Development findings** (all on disk, `results/getup_search/`): certified rolled equilibria exist 0–30° (ρ ≤ 0.095); a 45° keyframe exists with dynamic CoM placement (ρ = 0.078); SUPINE→SIDE has a certified 3-keyframe chain (15°/30°/45°, ρ ≤ 0.16) — keyframe EXISTENCE. Three bugs found and fixed along the way, all documented: pivot picked from arm/foot points instead of trunk edge (→ pivot restricted to trunk regions, 1-D CoM shift along roll); IK left base-xy free causing 2.2 m sideways drift (→ kept sticking contacts pinned + weak base-xy regulariser); acceptance threshold counted regulariser terms, rejecting valid keyframes (→ only hard residuals count). Warm-start/contact-state stall artefact → documented deterministic cold-restart rule (restart while ≥2% progress, max 4).

**Verdict: QUASI_STATIC_EXHAUSTION** (`results/getup_search/verdict.md`, `summary.json`, `trace.jsonl`; deterministic; 124.3 s wall). A* expanded only SUPINE; reachable modes = [SUPINE]. The single evaluated edge SUPINE→SIDE: INFEASIBLE — G2_orient STALLED after 344 ticks (102.6 s, 4 restarts); trunk never rotated even 1° (axes z: [0.995,0,0.101] → [0.983,0.156,0.097]); residual trunk_left(+y) = 59.99; cost 4.0847 = switch 1.0 + effort 1.0918 + margin 0.2729 + length 1.72. All other SUPINE edges pruned with stated reasons (P-TOPO ×3, P-F3 ×3). So: keyframes for rolling EXIST, but the oracle cannot EXECUTE the transition quasi-statically — existence ≠ executability, and the search certifies the gap.

**Reading**: within the defined graph, rules, and quasi-static scope, the autonomous search certifies that no strategy exists from supine to standing — this answers the momentum question in the affirmative direction: quasi-static imagination is exhausted, so getting up (from supine, on this model) requires momentum or a contact strategy outside the defined vocabulary. The verdict is scoped by the documented incompleteness (P-DESC level-monotone; single seed; nominal model).

**Honest limitations**: core built by agy across two legs (first cut by 429 quota after `getup_modes.py`; second cut by a runtime restart after the search completed — verdict files intact). This plan.md section, the results.md section, and report-cn.md "补充 5" were written by the coordinating subagent from on-disk data (`verdict.md`, `summary.json`, `trace.jsonl`, `graph.md`, `roll_probe.json`, `run.log`); all numbers quoted verbatim from those files.
