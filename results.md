# Results: 3-way comparison (run 2, 2026-10-30)

How to reproduce:

```
cd sim
.venv/bin/python experiment.py {fixed|passive|active|active_nodc} <seed>
.venv/bin/python experiment.py phase2
```

Raw JSON is in `results/`, logs are `results_*.log`.

Every condition gets the same budget: 6 s of double-support calibration, then the curriculum (6 steps in place, then 10 forward steps at 0.06 m stride). Perturbations: floor μ drops ×0.75 when forward walking starts, and an 18 N × 0.12 s lateral push is applied at forward step 5. Plants are seeds 0–2 of `scenarios.true_params`.

## Headline
**Every condition falls in every seed on the mismatched "real" plant.** The earlier run's baseline result (it holds 0.06 m stride) was measured on the *nominal* plant, and it does not carry over. So no recovery time could be measured: no episode survived to and through the push. The table below is mostly a negative result. What it does show is the **identification quality** and **where each condition fails**.

## Locomotion metrics (n = 3 seeds; "fwd" = reached the forward-walking stage)

| Condition | Falls | Furthest stage reached | Ref. fidelity, leg RMS (deg) | Constraint-violation rate | Recovery after push |
|---|---|---|---|---|---|
| (a) Fixed nominal | 3/3 | fwd 3/3 | 20.1 (19.2/19.0/22.1) | 0.091 | n/a (all fell) |
| (b) Passive online ID | 3/3 | fwd 1/3, sip 2/3 | 21.1 (seed 0 only) | 0.106 | n/a |
| (c) Active ID + gated adaptation | 3/3 | **sip 3/3** (falls earlier) | — | **0.060** | n/a |
| (c′) Active ID, no delay compensation (ablation) | 3/3 | fwd 2/3, sip 1/3 | 24.8 (23.0/26.7) | 0.043 | n/a |

- **Violation rate** is the fraction of control ticks with torque saturation, a QP fallback, or a stance-foot slip faster than 0.15 m/s.
- **Fidelity** only counts the forward-walking stage. Its values (around 20°) show that the leg-posture task has almost no weight (0.01): stability tasks dominate.

## Identification quality (final θ̂ − θ\*; prior row = URDF nominal error)

| | Seed | Δtrunk mass (kg) | Δx CoM (cm) | Δy CoM (cm) | Δdelay (ms) |
|---|---|---|---|---|---|
| Prior (URDF nominal) | 0/1/2 | −0.80 / −1.08 / −0.83 | −2.5 / −1.8 / −3.3 | +1.2 / −1.5 / −0.1 | −12 / −11 / −16 |
| Passive | 0/1/2 | +1.20 / −0.51 / −1.27 | +0.3 / **+4.2** / +2.7 | **−4.8** / −3.4 / −1.7 | +2 / +1 / +0.5 |
| Active | 0/1/2 | +0.61 / +0.51 / +0.43 | +0.8 / +0.6 / +1.9 | −1.3 / −0.4 / −1.1 | +2 / +1 / +0.5 |

- **Active ID** brings the CoM-offset error down to 0.6–1.9 cm in x, from 1.8–3.3 cm with the prior, and to ≤1.3 cm in y. Errors are consistent across seeds.
- **Delay** is identified within 2 ms by both ID conditions (motor-current torque lag).
- **Trunk mass** is overestimated by a consistent **+0.4–0.6 kg** in the active condition. That points to a systematic bias, most likely unmodelled 0.6 Nm leg friction entering the torque-based wrench reconstruction. Passive ID is unbiased in sign but scattered (±1.3 kg).
- **Passive online ID**, ungated with impacts included, made 7–17 commits and drifted to CoM errors as large as 4.8 cm. This is the failure mode the Phase-4 gate is meant to prevent.
- **Active ID** committed only 2 online updates per episode and rejected none. In other words the gate admits few samples during stepping, and those that pass show changes above 2σ.

## Phase 2: nominal vs. "real" under the same active exploration (seed 0)
- **Both plants stayed upright** through the exploration (6 s; foot load was unloaded down to 21–25% of body weight). The exploration is safe as designed.
- **CoM model error** (controller CoM vs. true CoM): (1.6, 1.6, 1.6) mm on the nominal plant vs. **(9.2, 4.4, 20.9) mm** on the real plant. The z error is the unobservable `trunk_dz`, kept as prior.
- **Leg torques**: the largest mean differences are knee (calf) +1.1/+0.6 Nm and ankle pitch +1.0/+1.3 Nm. That matches the extra trunk mass and the forward CoM offset. Hip roll and ankle roll differ by less than 0.1 Nm.
- **Joint angles** differ by up to 8° RMS at the knees (the friction plus delay response).
- **GRF sum is identical on both plants (107.6 N)**. The controller's own GRF estimator (`wbc.estimate_grf`) uses the model's leg dynamics and does not reflect the extra 0.8 kg. That is a gap in the contact-estimation layer.

**Task-relevant deviations, ranked:** sagittal trunk CoM offset and delay (these drive knee/ankle torque and the DCM error) > mass > lateral offset. `contact_tc` and backlash showed no clear signature in this experiment.

## Honest limitations
1. **No condition walks on the mismatched plant**, so the central hypothesis (active ID improves walking) is **not demonstrated**. Only better parameter estimates are shown.
2. **The active condition fails earlier.** The ablation shows **delay-compensation state extrapolation (12–14 ms) destabilises stepping in place**: without it, active reaches forward walking in 2 of 3 seeds (fixed baseline: 3 of 3). The identified delay is right, but the way it is used in the controller is wrong.
3. **Phase 3 is incomplete.** There is no outer gait-parameter optimisation and no uncertainty-risk term. The lateral fall at 0.10/0.14 m stride is **not fixed**. Self-collision is handled only as a minimum-foot-separation bound.
4. **Not identified:** friction, μ, contact stiffness, backlash, CoM z. The trunk-mass bias is uncorrected.
5. **Statistics are thin:** 3 seeds, one push direction, one friction change. Recovery-time metrics are therefore empty.
6. The delay-compensation conclusion is based on only 3 seeds (2 of 3 reached forward walking without it, vs 0 of 3 with it).


---

# Phase 2b: joint-zero active identification (q0 plant, 2026-10-31)

How to reproduce (run from the repo root; deterministic, CPU):

```
sim/.venv/bin/python sim/zero_experiment.py calib_all          # results/q0/calib_*.json, samples_*.npz
sim/.venv/bin/python sim/zero_observability.py                 # results/q0/observability.json
sim/.venv/bin/python sim/experiment.py <cond> <seed> 0.06 q0   # results/q0/<cond>_<seed>_s0.06.json
```

Logs: `results_q0_calib.log`, `results_q0_observability.log`, `results_q0_runs.log`.

- Plant: `true_params_q0(seed)` = the base mismatched plant + leg zero offsets.
- Prior: q0_hat = 0, σ = 2°.
- RMS values below are over the 12 leg joints, in degrees.
- **free** = the deployed estimator (foot placement unknown, IMU only). **jig** = foot placement known (reference upper bound).

## Calibration: active sway vs. quiet standing (6 s double support, seeds 0–4)

**Deployed estimator ("free")**

| Seed | Prior RMS | Active: posterior RMS | Active: obs-subspace RMS | Active: dirs committed / outcome | Quiet: posterior RMS | Quiet: obs-subspace RMS | Quiet: dirs committed / outcome |
|---|---|---|---|---|---|---|---|
| 0 | 1.79 | **0.91** | **0.31** | 8/12, commit | 1.57 | 1.39 | 7/12, commit |
| 1 | 2.07 | **1.29** | **0.38** | 6/12, commit | 1.58 | 1.10 | 7/12, commit |
| 2 | 2.07 | 1.34 | **0.17** | 6/12, commit | **1.29** | 0.73 | 7/12, commit |
| 3 | 2.14 | **1.39** | **0.20** | 6/12, commit | 1.64 | 0.36 | 7/12, commit |
| 4 | 2.17 | 2.17 (prior kept) | — | **refused**: robot fell during calibration | **1.23** | 1.05 | 7/12, commit |
| Mean, seeds 0–3 | 2.02 | **1.23** | **0.27** | | 1.52 | 0.90 | |

**Known-stance reference ("jig")**

| Seed | Active: posterior / obs-subspace | Quiet: posterior / obs-subspace |
|---|---|---|
| 0 | 0.60 / 0.18 | 1.24 / 0.22 |
| 1 | 0.84 / 0.24 | 1.03 / 0.65 |
| 2 | 1.14 / 0.15 | 1.42 / 1.04 |
| 3 | 1.03 / 0.25 | 1.59 / 1.39 |
| 4 | refused (fell) | refused by evidence test (2.17) |

- **Obs-subspace RMS** is the error projected onto the committed eigen-directions. That error is what the estimator is accountable for. The remaining error lives in uncommitted directions, which stay on the prior.
- **Active**: committed-subspace error 0.17–0.38°, against 0.36–1.39° for quiet standing. Quiet standing commits 7 directions, but its "observable" subspace is wrong by up to 1.4°. Without excitation, the information estimate is overconfident: the commit gate passes directions that the data do not actually pin down.
- **Seed-4 active refusal**: the sway excitation tipped this plant over. The true sole tilt reached 16.5° and the slip 17.8 mm.
  - The surprise gate rejected 199 of 1216 ticks.
  - The loop-consistency gate rejected 110 of the remaining 203 decimated samples. That leaves 93, below the 150 minimum.
  - Result: **no commit**, and q0_hat stays at the prior. The gates worked as designed: no corrupted update was written. The fall itself is a safety failure of the excitation on this plant.
- **Posterior RMS is dominated by the uncommitted subspace** (null-space RMS 0.85–1.37° active). The total error falls only from about 2.0° to about 1.2°.
- Backlash: the ankle-backlash bias (≤ 0.4°) is part of the measured zero. Errors against "true + backlash" differ by ≤ 0.04° in the obs-subspace RMS.

### Residual collapse (seed 0, active, free; RMS before = δ fixed at 0 with nuisances fitted, after = committed δ)

| Channel | Before | After |
|---|---|---|
| Gravity (sole tilt vs. IMU) | 1.02° | **0.15°** |
| Relative foot rotation | 1.26° | **0.06°** |
| Relative foot position | 2.25 mm | **0.97 mm** |

> [!WARNING]
> Residual collapse is **not** evidence of correct offsets. Quiet standing on seed 0 collapses its residuals just as well (0.96° → 0.12°, 1.13° → 0.03°, 0.91 → 0.42 mm) with a 1.39° error in its committed subspace. The nullspace directions fit the residuals at any value.

## Observability (noise-free kinematic rank, `observability.json`)

Relative singular values of the marginal 12×12 information (nuisances Schur-complemented). Seed 0 is shown; seeds 1–3 have the same structure.

| Direction (eigenvector) | Rel. sv, active | Rel. sv, quiet | Std from 1 s of data, active | Verdict |
|---|---|---|---|---|
| L−R antisymmetric pitch-chain sum (hip+calf+ankle) | 1.0 | 1.0 | 0.002° | observable (IMU + loop) |
| L−R antisymmetric roll-chain sum (hip+ankle roll) | 0.82 | 0.82 | 0.003° | observable |
| Common-mode pitch-chain sum | 0.23 | 0.23 | 0.010° | observable (IMU anchor) |
| Common-mode roll-chain sum | 0.20 | 0.20 | 0.011° | observable (IMU anchor) |
| Differential thigh / calf−ankle pitch / hip−ankle roll | 2.5e-2 to 4.5e-2 | 4.6e-3 to 1.3e-2 | 0.05–0.09° | observable with sway |
| Common-mode thigh | 1.2e-2 | 1.9e-3 | 0.18° | weak |
| Mixed hip-pitch difference / roll redistribution | 3.7e-3 to 7.7e-3 | 2.4e-4 to 5.4e-4 | 0.3–0.6° | weak, usually uncommitted |
| **Common-mode calf vs. ankle pitch** | **7.9e-4** | **5.1e-5** | **2.9°** | **near-nullspace** |
| **Common-mode hip pitch vs. calf + ankle** | **1.1e-3** | **1.5e-4** | **2.1°** | **near-nullspace** |

The quiet column is matched to the active one by rank. Only the active eigenvectors are labelled in the log, and the quiet labels were not checked one by one.

**Verdict:**
- The full-chain sums are anchored by the IMU gravity reference and are observable, including the common-mode sums.
- Differential (L−R) intra-chain redistributions are observable through the relative-foot-pose loop, and active sway improves them about 3–5×.
- **Common-mode intra-chain redistributions of pitch (and the mixed roll ones) are in the near-nullspace.** They move the base relative to both feet without changing sole tilt or relative foot pose, so the unmeasured base pose absorbs them.
- Active sway lifts these directions about 7–15× over quiet standing, which is still far below the prior. In the deployed estimator (τ_int-inflated, misfit-scaled), their formal std is 6.5–10.5° (seed 0), so the commit gate correctly leaves 4 to 6 directions on the prior.

## Downstream: 7 conditions × 3 seeds on the q0 plant (stride 0.06 m, same curriculum as above)

- `_z` = the zero-ID commit at the end of calibration, after which `SchemaEstimator` is replayed with q0_hat.
- passive_z uses quiet-standing calibration data. active_z and active_nodc_z use the active sway.
- Fidelity is the mean over seeds that reached forward walking. The violation rate is the mean over the 3 seeds (per-seed values in parentheses).

| Condition | Falls | Furthest stage (seeds 0/1/2) | Mean t_end (s) | Fidelity (deg) | Violation rate | q0 error RMS (deg), seeds 0/1/2 |
|---|---|---|---|---|---|---|
| fixed | 3/3 | fwd / sip / sip | 9.90 | 19.0 (s0) | 0.160 (0.088/0.190/0.201) | 1.79 / 2.07 / 2.07 (prior) |
| passive | 3/3 | sip / sip / sip | 9.25 | — | 0.158 (0.260/0.122/0.093) | 1.79 / 2.07 / 2.07 (prior) |
| active | 3/3 | sip / sip / fwd | 9.62 | 25.1 (s2) | 0.147 (0.228/0.136/0.077) | 1.79 / 2.07 / 2.07 (prior) |
| active_nodc | 3/3 | sip / sip / fwd | 9.83 | 25.1 (s2) | 0.108 (0.166/0.065/0.092) | 1.79 / 2.07 / 2.07 (prior) |
| passive_z | 3/3 | fwd / sip / fwd | 11.76 | 23.0 (23.4/22.6) | 0.136 (0.104/0.159/0.144) | 1.38 / **2.37** / 1.29 |
| active_z | 3/3 | sip / sip / sip | 8.71 | — | 0.116 (0.026/0.223/0.099) | **0.45 / 1.14 / 1.34** |
| active_nodc_z | 3/3 | fwd / sip / fwd | 11.04 | 22.7 (22.8/22.6) | **0.072** (0.093/0.067/0.055) | **0.45 / 1.14 / 1.34** |

- **All 21 episodes fall.** Zero-ID does not change the locomotion outcome on this plant, so the failure is dominated by factors other than q0. On the q0 plant, the delay-compensation effect from run 2 is less clear-cut: active falls in sip in 2/3 seeds, active_nodc also in 2/3, and active_z in 3/3 vs active_nodc_z in 1/3.
- active_z / active_nodc_z have the lowest q0 error. Their episode calibration is a separate run from the `calib_*` runs, so seed 0 gives 0.45° here vs 0.91° in calibration. passive_z made seed 1 **worse** than the prior (2.37 vs 2.07°): the quiet-standing overconfidence seen in calibration carried into the episode.
- active_nodc_z has the lowest violation rate of all 7 conditions (0.072), and both `_z` conditions without delay compensation reach forward walking in 2 of 3 seeds, against 1 of 3 for their non-`_z` counterparts. With n = 3 this is a **trend only**. Note that passive_z also reaches fwd in 2 of 3 despite a worse q0 estimate, so the gain may not come from q0 accuracy.
- Seed-2 `_z` runs (active_z, passive_z, active_nodc_z) were added on 2026-10-31 with the same entry point. Wall time was 15–22 s each.

## Safe-motion observability scan (2026-10-31, follow-up to Phase 2b)

How to reproduce:

```
sim/.venv/bin/python sim/zero_motion_scan.py                     # results/q0/motion_scan.json, results_q0_motion_scan.log
sim/.venv/bin/python sim/zero_experiment.py calib <seed> yaw15   # also yaw15sq, yaw25 -> calib_<mode>_<seed>.json
sim/.venv/bin/python sim/zero_motion_validate.py                 # results/q0/motion_validate.json, results_q0_motion_validate.log
sim/.venv/bin/python sim/zero_external_ref.py                    # results/q0/external_ref.json, results_q0_external_ref.log
```

**Weak directions**: the 4 weakest eigenvectors of the real seed-0 Phase-2 calibration, with the deployed formal std of each.

| | Direction | Deployed formal std, seed 0 |
|---|---|---|
| W1 | Common-mode calf vs. ankle pitch | 10.4° |
| W2 | Common-mode hip pitch vs. calf + ankle | 6.5° |
| W3 | Common-mode hip roll vs. ankle roll (+ r hip pitch) | 1.4° |
| W4 | Mixed L/R hip-pitch / ankle-pitch / roll | 0.75° |

**"Dep-eq"** = noiseless 6 s marginal std × K, K = (5.7–8.9) per direction, calibrated on the same seed-0 run. Thresholds: **2°** = prior; **0.5°** = commit gate (κ = 16).

### Why symmetric motions cannot work (exact result)
A common-mode allocation δ_L = δ_R with zero chain sum leaves both sole orientations unchanged. It moves both soles by the same vector J(q)·δ in the base frame whenever the two legs have the same sagittal configuration. The IMU sees nothing, and neither does the relative foot pose. **Any L/R-symmetric motion, however large, keeps W1 in the exact nullspace.** In the kinematic replica, the rel. sv of W1 is 1e-8 for sway, squat, base fore/aft, and their combinations. The 1e-3 seen in simulation comes only from incidental asymmetries in tracking. Only motions that make the two legs' sagittal Jacobians **differ** can help: stance width, stagger, toe-out, trunk yaw twist, and base roll.

### Scan (noiseless kinematics, 6 s at 100 Hz; full list in `results_q0_motion_scan.log`)

| Candidate (on top of the Phase-2 sway, base −2 cm) | Safe | W1 dep-eq | W2 | W3 | W4 | # dirs < 0.5° | < 2° prior? |
|---|---|---|---|---|---|---|---|
| Baseline sway (replica) | ✓ | **∞ (exact null)** | 18.0 | 9.1 | 0.99 | 8 | no |
| Squat −2 / −4 cm static | ✓ | ∞ | 18.0 | 9.1 | 0.84–0.94 | 8 | no |
| Squat −6 / −8 cm static | ✗ (ankle-pitch limit) | ∞ | 18.0 | 9.1 | 0.63–0.73 | 8 | no |
| Squat cycle 0…−6 cm (0.2 Hz) | ✓ | ∞ | 18.0 | 9.0 | 0.44 | 9 | no |
| Base ±2 cm fore/aft over the feet | ✓ | ∞ | 18.0 | 9.1 | 0.88–1.10 | 8–9 | no |
| Width +3 / +6 cm (operator-placed) | ✓ | 8.7 / 4.5 | 1.4 / 1.0 | 1.35 / 1.09 | 0.65 / 0.39 | 9 | no |
| Width −2 cm | ✓ | 13.8 | 1.8 | 1.1 | 0.95 | 9 | no |
| Stagger 4 / 8 cm (operator-placed) | ✓ | 8.4 / 4.9 | 1.6 / 1.5 | 0.95 / 0.62 | 0.67 / 0.45 | 9 | no |
| Stagger 12 cm | ✗ (joint limit, CoP 12 mm) | 3.6 | 1.4 | 0.47 | 0.37 | 9 | no |
| Toe-out 10° | ✓ | 11.1 | 1.4 | 1.0 | 1.0 | 9 | no |
| Base roll ±3° / ±5° | ✓ / ✗ (tilt 6.1°) | 6.4 / 3.9 | 1.6 / 1.5 | 0.61 / 0.38 | 0.42 / 0.29 | 9 | no |
| **Trunk yaw twist ±5 / ±10 / ±15°** (0.35 Hz) | ✓ | 7.5 / 3.8 / **2.5** | 0.92 / 0.52 / **0.36** | 0.45 / 0.24 / 0.17 | 0.32 / 0.19 / 0.14 | 9 / 10 / 11 | W2–W4 yes, W1 no |
| Yaw twist ±20° | ✗ (l hip-roll limit 1.1°) | 1.9 | 0.27 | 0.13 | 0.12 | 11 | borderline, unsafe |
| **Yaw ±15° + squat cycle 0…−6 cm** | ✓ | **2.2** | **0.35** | 0.16 | 0.12 | 11 | W1 no (closest safe) |
| Yaw ±10° + roll ±3° | ✓ | 3.8 | 0.52 | 0.20 | 0.17 | 10 | no |
| Yaw ±10° / ±15°, no sway | ✓ | 4.4 / 2.9 | 1.8 / 1.0 | 0.32 / 0.21 | 0.34 / 0.23 | 8 / 9 | no |

- Known L/R foot placement ("jig") makes **no difference**: the jig std equals the free std for every candidate, because the jig fixes L↔R, not base↔feet.
- **Scan verdict**: the trunk yaw twist about the planted feet is the only motion class with real leverage. It needs no stepping and no re-placement, since the hip yaw ("thigh") joints absorb the twist.
  - The best safe candidate predicts W1 ≈ 2.2° (still above the prior) and W2 ≈ 0.35° (below the commit gate).
  - Squat depth and fore/aft shifts do nothing for W1, as the symmetry argument predicts.

### Full-simulation validation (deployed estimator, free stance, seeds 0–3)

Mode `yaw15` = ±15° commanded at 0.35 Hz; `yaw15sq` = the same + a 0…−6 cm squat cycle. Weak-direction std is the deployed formal marginal std. The last column is the error along W2 against its true value.

| Mode | Seed | Realised hip-yaw range | n | Dirs committed | W1 std | W2 std | W3 std | W4 std | Total RMS (prior) | Obs-subspace err | Slip | W2 err (true) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| active | 0 | 1.5° | 545 | 8 | 10.4 | 6.5 | 1.43 | 0.75 | 0.91 (1.79) | 0.31 | 1.4 mm | −0.58 (+0.54) |
| active | 1 | 3° | 548 | 6 | 19.7 | 11.7 | 4.2 | 2.6 | 1.29 (2.07) | 0.38 | 2.7 | +1.49 (−1.53) |
| active | 2 | 1.2° | 555 | 6 | 16.8 | 5.7 | 3.7 | 2.4 | 1.34 (2.07) | 0.17 | 1.2 | +0.95 (−0.93) |
| active | 3 | 1.5° | 561 | 6 | 30.7 | 14.9 | 6.0 | 3.1 | 1.39 (2.14) | 0.20 | 1.4 | −1.94 (+1.93) |
| yaw15 | 0 | 17.4° | 455 | 9 | 7.0 | 1.04 | 0.41 | 0.33 | 1.04 | 0.28 | 2.0 | −0.44 |
| yaw15 | 1 | 18.7° | 414 | 7 | 21.0 | 2.5 | 1.5 | 1.2 | 1.37 | **0.79** | **9.1** | +1.39 |
| yaw15 | 2 | 11.4° | 511 | 4 | 53.6 | 9.2 | 7.0 | 5.4 | 1.54 | 0.44 | 4.6 | +0.78 |
| yaw15 | 3 | 11.0° | 526 | 7 | 13.5 | 2.1 | 1.3 | 1.2 | 0.80 | 0.22 | 1.3 | −1.95 |
| yaw15sq | 0 | 15.2° | 382 | 10 | 6.1 | **0.31** | 0.46 | 0.31 | **0.78** | **0.75** | 3.1 | **−0.12** |
| yaw15sq | 1 | 17.3° | 413 | 7 | 9.3 | 2.3 | 1.1 | 0.65 | 1.03 | 0.20 | **7.1** | +1.69 |
| yaw15sq | 2 | 14.3° | 407 | 4 | 49.3 | 9.7 | 5.3 | 5.3 | 1.63 | 0.34 | 5.6 | +0.89 |
| yaw15sq | 3 | 9.0° | 431 | 8 | 11.5 | 0.87 | 0.84 | 0.72 | 0.99 | **0.73** | 1.6 | −1.70 |
| yaw25 | 0 | 35.6° | 188 | 7 | 14.0 | 1.2 | 1.05 | 0.94 | 1.29 | 0.22 | 4.4 | −0.56 |

- **W1 (calf vs. ankle, common mode)**: 6.1–53.6° with the twist, against 10.4–30.7° without it. It is never below 2°, and the estimate stays on the prior (error ≈ −true) in every run.
- **W2 (hip pitch vs. calf + ankle)**: below 2° in 3 of 8 twist runs (0.31, 0.87, 1.04°). Only seed 0 of `yaw15sq` actually corrected it (error −0.12 against +0.54 true). Seed 3 (formal std 0.87°) stayed near the prior.
- **Why the prediction was not met** (decomposed on the cached seed-0 samples, true q):
  - The realised trajectories *are* geometrically 2–3.6× more informative on W1: noiseless 0.34–0.60° against 1.22°.
  - But the realised hip-yaw ranges were only 9–19°, i.e. about ±4.5…±9.5° of trunk yaw out of ±15° commanded: the WBC trunk-orientation task competes with the CoM / posture tasks.
  - The twist adds torsional sole slip and pad compliance that the loop-closure model does not include. The τ_int of the relative-position channels rises to 130–190 samples and the misfit scale to 1.3–2.3. The inflation factor goes from ≈ 9× to ≈ 11–12×.
  - `yaw25` realises ±18°, but the loop-consistency gate rejects 114 samples (n = 188), so the run ends worse.
- **Side effects**: slip grows from 1.2–2.7 mm to 1.3–9.1 mm. The committed-subspace error rises in 3 of 8 runs (0.73–0.79° against 0.17–0.38°): a 3.4σ bias on W3 in yaw15sq seed 0. So the twist adds unmodelled systematics as well as information.

**Verdict: (b) NO.**
- No safe double-support motion brings the common-mode calf-vs-ankle-pitch allocation (W1) below the 2° prior.
- Best achieved: predicted 2.2° (yaw ±15° + squat cycle, noiseless × K); realised in simulation 6.1° (seed 0) to 49° (seed 2).
- The hip-pitch allocation (W2) is reachable only intermittently.
- The yaw-twist option stays in the code, default **off**, and is **not** integrated into the Phase-2 `exploration_targets` defaults. Its gain is not robust across seeds and it raises slip and committed-subspace bias.

### Minimal external reference (sizing, `zero_external_ref.py`; deployed Phase-2 information of seeds 0–3 + prior + external measurements)

**What is missing**: the pose of the base relative to the (common) feet. In the stand posture, 1° along W1–W4 moves both soles by only about **1.2–1.9 mm** in x, 0.05–0.5 mm in z, and up to 1.9 mm in y (W3/W4), as seen in the base frame.

| External measurement of both sole positions in the base frame | W1 | W2 | W3 | W4 | Posterior RMS std (12 joints) |
|---|---|---|---|---|---|
| None (Phase-2 sway only, prior incl.) | 1.98 | 1.93 | 1.54 | 1.08 | 1.05 |
| Base height only (z), 1 mm, 3 postures | 0.96 | 1.59 | 1.47 | 0.99 | 0.81 |
| Fore-aft only (x), 1 mm, 3 postures | 1.70 | 1.02 | 1.34 | 0.85 | 0.81 |
| x+z, 1 mm, 3 postures | 0.49 | 0.36 | 1.32 | 0.85 | 0.58 |
| **x+y+z, 2 mm, 3 postures** | **0.88** | **0.61** | **0.40** | **0.37** | **0.45** |
| **x+y+z, 1 mm, 2 postures** | **0.55** | **0.38** | **0.28** | **0.27** | **0.33** |
| x+y+z, 1 mm, 3 postures | 0.49 | 0.33 | 0.23 | 0.22 | 0.30 |
| x+y+z, 5 mm, 3 postures | 1.47 | 1.11 | 0.80 | 0.67 | 0.70 |

Postures: stand (−2 cm), squat (−6 cm), base 2 cm forward, all static double support. Values are means over seeds 0–3; the max across seeds is at most 0.07° higher.

**Minimal reference**: a **3-D measurement of the base (pelvis) pose relative to both soles, accurate to ≤ 1–2 mm, in 2–3 static double-support postures.**
- A base-height measurement alone, or a known foot fixture alone, does not suffice.
- A single-foot contact probe does not address the common mode, and it adds single-support risk.

**Concrete safe proposal for the real pi+** (not built; user decision):
1. **Markers**: one rigid AprilTag/ArUco board on the pelvis, at a known pose relative to `base_link` from CAD, and one flat foot plate with a printed tag grid and heel/side stops for both soles. Alternatively, small tags on both feet.
2. **Postures**: the robot stands on the plate in double support under the WBC, on the safety gantry with a slack rope. Hold the 3 static postures (stand, −6 cm squat, base +2 cm forward) for 2 s each. The robot already performs all of these in Phase 2, so no new risk is added.
3. **Capture**: a calibrated camera at 0.6–0.8 m takes a few frames per posture, ideally 2 views to tighten depth. Expected relative-pose accuracy is about 0.5–1 mm. This should be checked first against a caliper on a dummy board.
4. **Fusion**: the measured sole positions in the base frame enter `ZeroEstimator` as an extra residual block, σ = 1–2 mm, with q at the capture instant. This closes W1–W4 to 0.3–0.9°, after which the commit gate can write them.
5. **Alternative without vision**: a mechanical pelvis-datum fixture (gauge blocks / dial indicators from the foot plate to two pelvis datum points). This gives the same information, but it is more hardware.

## Torque channel: full joint-torque residual for W1–W4 (2026-10-31, user's proposal)

**Question**: the loop-closure estimator reduces the 12 leg joint torques to nothing (and `SchemaEstimator` reduces them to 3 equations: weight + CoM xy). Does the **full** torque vector, used as an overdetermined residual, make the common-mode intra-leg directions W1–W4 observable? The user's physical intuition: with the leg truly straight, pitch gravity torques vanish — a first-order signal (~1.5 N·m at 2° bias under body weight, vs 0.05 N·m torque noise).

**Method** (`sim/zero_torque.py`, `ZeroEstimator(torque=True)`; design in `plan.md` → "Phase 2b torque channel"): in double support with no F/T sensors, the 12 leg rows are solved exactly for the two 6-D sole wrenches (`F_i = J_leg,i⁻ᵀ(h_leg,i − τ_j,i)`); the 6 floating-base rows are the residual `r = h_base − J_LᵀF_L − J_RᵀF_R`. Base velocity/acceleration come from the left-sole contact constraint (no base-state estimator); IMU roll/pitch only. Noise model σ² = 0.05² + (0.02τ)² + stick term, propagated exactly and Cholesky-whitened per sample. Unknowns: δ (12) + trunk (dm, dx, dy, dz) + friction μ and transition velocity v0 co-identified. Two new gates: torque-consistency gate + channel-agreement veto (torque channel is vetoed wherever it disagrees with the LC-only commit → no regression on the LC-observable subspace by construction).

**Scan verdict: YES** (`sim/zero_torque_scan.py` → `results/q0/torque_scan.json`; W1–W4 formal std, deployed convention):

| seed / mode | LC-only W1–W4 (°) | + torque channel W1–W4 (°) |
|---|---|---|
| 0 active | 10.3, 6.7, 1.6, 0.8 | 0.75, 0.41, 0.43, 0.36 |
| 1 active | 28.0, 12.1, 3.8, 1.8 | 0.60, 0.44, 0.37, 0.55 |
| 2 active | 17.4, 5.5, 4.6, 0.9 | 0.83, 0.59, 0.55, 0.57 |
| 3 active | 32.2, 17.0, 7.2, 3.7 | 1.18, 0.84, 1.43, 1.21 |
| 0–3 quiet | 17–140, 6–13, 5–8, 1.7–2.7 | 0.9–1.4, 0.4–1.2, 0.4–0.6, 0.2–1.0 |

W1 drops from 10–140° to **0.6–1.4°**, below the 2° bar on every seed and mode. Per-degree signatures (seed 0, active): W1 → Fz 1.02 N/° (std 0.44 over the trajectory — posture-dependent); trunk dm → Fz 0.98 N/0.1 kg (std 0.005); dx → My 0.416 N·m/cm (std 0.002). The posture dependence is what separates W1 from the trunk parameters.

**Validation** (`sim/zero_torque_validate.py` → `results/q0/torque_summary.json`; offline re-solve on the recorded calibrations):

| seed | LC-only: RMS / W1 err | + torque: RMS / W1 err | weakest std LC → tq | μ̂ (true 0.60) |
|---|---|---|---|---|
| active 0 | 0.91° / −0.52° | 0.42° / −0.32° | 10.5° → 0.73° | 0.608 |
| active 1 | 1.29° / +0.93° | 0.81° / +0.10° | 20.9° → 0.81° | 0.615 |
| active 2 | 1.34° / +1.09° | 1.04° / +2.75° | 17.1° → 0.85° | 0.775 |
| active 3 | 1.39° / +0.41° | 1.39° / +0.41° (vetoed) | 32.3° → 1.54° | 0.864 |
| active 4 (fell) | rejected | rejected (same) | — | — |
| quiet 0–4 | 1.2–1.6° | unchanged (LC kept) | — | — |

- Friction co-identification works on seeds 0–1 (μ̂ ≈ 0.61); on seeds 2–3 it drifts (0.78/0.86), and seed 2's W1 error degrades (+1.09° → +2.75°) — the friction↔W1 entanglement warned about in the brief is real on less informative trajectories.
- Seed 3: the channel-agreement veto kept the LC-only result (no regression, no gain). Quiet standing: the torque channel changes nothing (no configuration variation → no disambiguation); LC result kept.
- Seed 4 (fell during calibration): still correctly rejected — too few gated samples (93 < 150).
- Observable-subspace RMS shows no regression (seed 0: 0.31° LC-only vs 0.27° with torque).

**Trunk-vs-offset disambiguation** (`sim/zero_torque_disamb.py` → `results/q0/torque_disamb_active_*.json`): over the sway trajectory, W1's Fz signature varies −1.63 → −0.81 N/° across knee-bend bins (21° → 51°), while trunk dm (Fz 0.98 N/0.1 kg) and dx (My −0.416 N·m/cm) are posture-invariant. In a single static pose a +0.8 kg trunk-mass error leaks −0.43…+0.55° into W1; over the trajectory the leak drops to ≤ 0.02°.

**Downstream**: one smoke-test episode with the torque channel wired in (`experiment.py` condition `active_nodc_zt`, seed 0): fell at t = 12.51 s in the fwd stage, viol_rate 0.089 — in line with the non-torque `_z` runs, no dramatic change from a single seed. A full `_zt` comparison was not run.

**Honest limitations**: validation gains are real but uneven — clear wins on seeds 0–1, W1 degradation on seed 2 tied to friction misidentification, veto on seed 3. The torque channel is an offline-validated estimator enhancement; the AprilTag external reference remains a valid fallback but is no longer the only path to W1.

## F3: attention-based explicit MPC get-up test (2026-10-31, verdict NOT_FOUND)

Falsification test from the get-up gap analysis: on the nominal model, search a quasi-statically feasible keyframe sequence supine→standing with per-keyframe peak |tau|/tau_max ≤ 0.8 (20 N·m legs / 10 N·m arms). Interpretable attention + explicit MPC; no black-box policy; quasi-static only.

**G0** (`sim/model.py`; probe `results/f3/g0_probe.json`): simplified thigh/hand/knee collision shells, supine spawn/reset, get-up failure criterion. Probe: supine → base_link + both ankles (12 contacts, torque-channel readout max err 2.3% mg); long_sit → ankles + elbows + hip_roll_links; prone → ankles + elbows. Multi-link contacts form and are torque-readable.

**Method** (`sim/getup_schema.py`, `sim/getup_attention.py`, `sim/getup_mpc.py`): attention weights are functions of schema quantities — support attention over 18 contact regions, motion attention over 20 joints per task; explicit min-effort keyframe QP with quasi-static equilibrium, unilateral + friction (mu 0.6) on the attended contact set, joint/torque/self-collision limits (horizon N=4). Runner `sim/getup_f3_run.py`; diagnosis `sim/getup_f3_diag.py`; plots `sim/getup_f3_plot.py`.

**Verdict: NOT_FOUND** (`results/f3/run_a.json`, deterministic; 122 ticks, 45.8 s wall). Stalled at **P1_head_up**, tick 121, outcome STALLED; 2 keyframes certified; overall peak rho **0.181** — far below the 0.8 cap, so not torque-bound.

| Ablation from stall (15 ticks) | task cost | trunk tilt |
|---|---|---|
| full (baseline) | 1.379 | 11.1° |
| no self-collision | 1.409 | 11.4° |
| no joint limits | 1.739 | 12.5° |
| big trust region | 1.466 | 11.6° |
| rho_cap → ∞ | 1.379 | 11.1° |
| friction ×2 | 1.154 | 10.6° |

Relaxing every constraint class fails to unstick it — the stall is structural, not constraint-tuning. trunk_upright task cost stuck at 2.84 with tilt ~11–17°: quasi-static trunk lift from supine has no feasible path through this strategy.

**Attention** (`results/f3/attention_a.png`, `attention_a_table.md`): weights track the certified static load share — phase entry back 0.96 vs load 0.81; means pelvis 0.21 vs 0.35, thighs 0.21 vs 0.29, knees/shanks 0.25 vs 0.02. At stall: support attention pelvis 0.645 + thighs 0.157/0.17 (correct — that is where the load is); motion attention shoulders 0.49 (a pushing attempt), effort alpha hip 0.34/shoulder 0.31. Interpretable, physical — but attention cannot invent a feasible motion where none exists.

**Reading**: falsifies "quasi-static imagine suffices via direct supine→trunk-lift→tuck→…". The evidence points to momentum or a qualitatively different contact strategy (e.g. rolling to side/prone first). The verdict covers the searched strategy, not get-up in general. Single seed (tag a).

**Honest limitations**: F3 docs in this section were written by the coordinating subagent from on-disk run data (agy's two legs were cut by print-timeout/SIGTERM before docs); all numbers are quoted verbatim from `results/f3/run_a.json`, `run_a.log`, `attention_a_table.md`. Attention figure generated from the run's own per-tick logs.

## Autonomous interpretable get-up strategy search (2026-10-31, verdict QUASI_STATIC_EXHAUSTION)

The robot searches its own contact-mode sequence supine→standing: no hand-supplied sequences, no black-box policy. Region-to-target ("imagine") stays the separate low-level layer (`getup_mpc.py` unchanged).

**Graph** (`sim/getup_modes.py`; full table in `results/getup_search/graph.md`): 8 modes — SUPINE (back, LYING_UP/0), SIDE (right flank, LYING_SIDE/0), PRONE (chest, LYING_DOWN/0), SIT (butt, UPRIGHT/1), QUAD (hands+knees, PITCHED/1), HF (hands+feet, PITCHED/1), SQUAT (feet, UPRIGHT/2, base z ≤ 0.26 m), STAND (feet, UPRIGHT/2, base z ≥ 0.32 m = goal). Edge rules: P-TOPO (orientation adjacency), P-SKIP (no 0→2 level skips), P-DESC (level-monotone scope, documented incompleteness), P-F3 (F3 diagnosis as explicit pruned-edge evidence), P-START/P-GOAL. Heuristic hops to STAND: STAND 0, HF/SIT/QUAD/SQUAT 1, SIDE/PRONE 2, SUPINE 3.

**Oracle** (`sim/getup_oracle.py`): edge → sub-goals G1_make (establish new contacts) + G2_orient (reorient trunk), executed by the MPC with interpretable attention; returns EdgeResult (status/cost/ticks/attention/diagnosis). Keyframe imagination (`sim/getup_keyframes.py`): body-schema IK + exact statics certificate, deterministic. **A*** (`sim/getup_search.py`): c(A→B) = 1.0 + Σmean_j(τ_j/τ_max_j)² + peak_rho/0.8 + 0.5·ticks/100; h(A) = hops-to-STAND (admissible). Deterministic, single-threaded.

**Development probes** (`results/getup_search/roll_probe.json`): certified rolled equilibria 0°→30° (ρ 0.095→0.082, binding hip pitch); 45° keyframe exists with dynamic CoM placement (ρ = 0.078); SUPINE→SIDE certified 3-keyframe chain (15°/30°/45°, ρ ≤ 0.16, 8 s). Three fixed bugs: pivot picked from arm/foot points (→ trunk-region pivots, 1-D CoM shift); IK base-xy drift 2.2 m (→ pin kept sticking contacts + weak base-xy regulariser); acceptance threshold counted regulariser terms (→ hard residuals only). Warm-start stall artefact → deterministic cold-restart rule (≥2% progress, max 4).

**Search trace** (`results/getup_search/trace.jsonl`, `summary.json`; 124.3 s wall): expanded nodes in order: SUPINE only. Reachable: [SUPINE]; never reached: SIDE, PRONE, SIT, QUAD, HF, SQUAT, STAND.

| SUPINE frontier edge | status | rule / failing sub-goal | diagnosis |
|---|---|---|---|
| → SIDE | INFEASIBLE | G2_orient STALLED (344 ticks, 102.6 s, 4 restarts) | structural: no constraint-family relaxation unlocks progress; trunk axes z [0.995,0,0.101]→[0.983,0.156,0.097] — never rotated even 1°; residual trunk_left(+y)=59.99; cost 4.0847 = 1.0 + 1.0918 + 0.2729 + 1.72; peak rho 0.218 |
| → PRONE / QUAD / HF | PRUNED | P-TOPO | orientation classes not adjacent (must pass through an intermediate class) |
| → SIT / SQUAT / STAND | PRUNED | P-F3 | F3 run_a evidence: direct supine trunk lift structurally stalled (peak rho 0.181, all relaxations failed) |

**Verdict: QUASI_STATIC_EXHAUSTION** (`results/getup_search/verdict.md`). Key distinction the run establishes: keyframe EXISTENCE (certified 15°/30°/45° chain) ≠ EXECUTABILITY (the oracle cannot drive the transition quasi-statically). Within the defined graph, rules, and quasi-static scope, the autonomous search certifies no strategy exists — i.e. the momentum question is answered affirmatively: getting up from supine on this model needs momentum or a contact strategy outside the defined vocabulary. Scoped by: P-DESC level-monotone incompleteness, single seed, nominal model.

**Attention at the stall** (from `summary.json`): support attention top = upper_trunk 0.75, lower_back 0.154 (correct — that is where the load is: cert loads upper_trunk 0.588, lower_back 0.334); motion attention top = l/r hip_roll 0.295/0.279 (the roll joints); effort bottleneck l_hip_pitch 2.895. The attention is physically sensible throughout — interpretability held, but there was no feasible motion to select.

**Honest limitations**: core built by agy in two legs (429-quota cut after `getup_modes.py`; runtime-restart cut after the search completed — verdict files intact). This section was written by the coordinating subagent from on-disk data; all numbers verbatim from `verdict.md`, `summary.json`, `trace.jsonl`, `graph.md`, `roll_probe.json`, `run.log`.
