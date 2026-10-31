# EXP E3: Is scheme B's supine plateau a statics gap or a search/action-basis failure?

Date: 2026-11-01. Plant: scheme B's 24-DoF pi+ (`codex/assets/scene.xml`, mesh collision, 11.00 kg).
Venv: `/home/hatch/workspace/research/body-schema-gait/.venv-e3` (mujoco 3.17.0, numpy 2.3.5, scipy 1.17.0).
All paths below are relative to this directory. The pre-registration ([PREREGISTRATION.md](PREREGISTRATION.md)) was written after step 1 and before steps 2 and 3.

## TL;DR

The pre-registered falsifier, applied literally, gives **neither (a) nor (b)**.
- **(b) is rejected.** Certified feet-only static equilibria exist "within reach" as pre-registered (feet kept at their plateau contacts): 9/9 IK solves certified, ρ = 0.067–0.095 against a cap of 0.8, CoM margin 48–66 mm. So the plateau is **not a statics gap in the existence sense**, and it is **not a torque-authority limit** (ρ ≤ 0.1 everywhere relevant).
- **(a) is not established.** Neither the augmented basis nor a (post-hoc) score variant produced a success. Count: 0 of 2 augmented-or-score runs, and 0 of 2 original-basis controls.

Supporting evidence for a narrower "support-polygon jump" story: across 9 local solves, I **could not find a lift-off configuration**. That is a configuration in which any non-foot support (the plateau hand pinned or re-placed, or the pelvis) still touches the floor while the feet-only certificate already holds. The best CoM margins reached were −52 to −63 mm (hand pinned, all constraints met), −19 mm (hand re-placed, with residual violations) and −16 mm (pelvis touching, with violations). The certified hand-supported region and the certified feet-only region therefore appear **not to overlap**. Releasing the last non-foot support would then require a non-static interval (momentum) of at least ~5 cm of CoM travel. This is a gap in **connectivity, not existence**, so it differs from A's 40–45° roll gap (an existence gap). It also rests on local least-squares searches and is not a proof.

## 0. Scope actually implemented

| Item | Status |
|---|---|
| Port of A's `statics_lp` + `certify` to B's plant | **Full port** for any B state ([e3_cert.py](e3_cert.py)). Same LP (HiGHS), same friction pyramid, ρ cap 0.8, pen ≥ −1.5 mm, selfd ≥ −1 mm, eps_cert 5 mm, μ = 0.6 (A) with 0.8 (B floor) also reported. Adaptations: mesh collision → convex-hull vertices as shell points (10,668 points); 796 self-collision pairs as MuJoCo's default filter; τ_max = B's ctrlrange (20/10/3 N·m); feet = `*_ankle_roll_link` only (as B's `metrics()`); self-contacts carry no force (A's convention, conservative). |
| Sanity checks | B's standing pose: feet-only certified, ρ = 0.057. B's passive supine: all-contacts certified, ρ = 0.022; feet-only infeasible (expected). |
| Step 1: plateau + all 10 block-end states of B's best path | done ([results/step1_plateau_cert.json](results/step1_plateau_cert.json)); replay error 0.0 |
| Step 2: IK-nearby feet-only search | done for tier A (9 solves) and partial tier B (3 of 6 solves; stopped for time, stdout kept in `results/step2_stdout_final.log`) |
| Step 2b: lift-off / connectivity | L1 (2×3 solves) + max-margin (3) done; L2 2 of 4 solves; L3 reduced to 2 solves. The chain test was **not run** because no lift-off configuration was found. |
| Step 3: augmented-basis search from the plateau | done for 1 seed of the augmented basis + 2 seeds of the original basis. 3 seeds per arm were pre-registered; the shortfall is due to CPU contention (another experiment was loading the 4-CPU VM to ~14). Plus 1 post-hoc score-variant run. |
| Step 3c: certificate-guided PD execution (post-hoc diagnostic) | done |

## 1. Step 1: the plateau, certified

Commands: `../../.venv-e3/bin/python e3_step1_plateau.py` (log `results_step1.log`).

| State | Support (loaded) | All-contacts cert (μ 0.6) | Feet-only cert |
|---|---|---|---|
| plateau snapshot (base 0.261, up 0.691, feet 104 N, nonfoot 53 N, **KE 0.126 J**) | L_foot 0.23, R_foot 0.30, **R_hand 0.47** mg | **OK**, ρ = 0.160 (r_hip_pitch) | **infeasible**: CoM x = 0.234 m lies behind both feet (contacts at x = 0.41/0.51 m); hand contact at x = −0.03 m |
| blocks 1–10 of the best path | feet + trunk/head/arm/hand | all OK, ρ 0.08–0.35 | all infeasible |
| plateau + zero-action holds 1..5 | — | OK, OK (ρ 0.21, 0.57), then infeasible while falling | falls back to supine by hold 3 (base 0.12, up −0.02) |

Facts:
- The plateau is a **certified tripod equilibrium (2 feet + right hand)** with low torque demand.
- The saved snapshot is **not at rest**. The beam kept a transient state whose score (3.884) no successor beats.
- B's own zero-action PD hold does not keep it. As a control (step 3c), B's PD hold also fails to keep **nominal standing** beyond ~1.8 s (handover criterion true for 3 blocks, then base 0.307 → 0.096). B relies on its QP `balance()` for standing.
- The plateau stalled from **level 14** (not 17) of `hybrid_supine_101.json` (identical metrics at levels 14–24). The plateau stall of B's run is reproduced here: from the plateau, B's 12-synergy physics search never beats score 3.884 in 3,824 evaluations, for either seed.

## 2. Step 2: does a feet-only certified equilibrium exist within reach?

Command: `../../.venv-e3/bin/python e3_step2_feet_ik.py` ([e3_step2_feet_ik.py](e3_step2_feet_ik.py)). The method is bounded least squares over base(6) + joints(24), A-keyframe style:
- flat soles on the floor;
- sole centre within R_pin of each foot's plateau contact;
- every non-foot geom ≥ 1 cm above the floor;
- no self-collision (≥ 3 mm);
- CoM over the support centroid;
- posture regulariser w_post ∈ {1, 0.3, 0.1};
- 3 initial guesses (plateau, standing joints, squat joints);
- a polish pass, then A's certificate.

**Tier A (pre-registered "within reach": feet stay put, R_pin = 6 cm): 9/9 certified.**
- ρ 0.067–0.095 (μ 0.6; identical at μ 0.8).
- CoM margin inside the foot polygon 48–66 mm.
- pen −0.3 mm, selfd 3–10 mm.
- Joint change ‖Δq‖∞ 0.81–1.74 rad.
- **Base displacement 0.28–0.31 m.**

Nearest certified solution (min ‖Δq‖₂ = 1.37 rad; saved as `results/feet_only_eq_tierA.npy`):
- qpos = `[0.443, -0.229, 0.358, 0.926, 0.055, 0.156, -0.341, 0.083, 0.013, 0.547, -0.392, 0.035, -0.217, -0.651, 0.007, -0.516, -0.014, 0.293, -0.157, 0.671, 1.434, 0.534, 0.547, -0.001, 0.514, -0.959, -0.514, -0.2, 0.028, -1.256, 0.174]`
- Certificate: feasible, **ρ = 0.069** (binding head_pitch), loads L_foot 0.44 / R_foot 0.56 mg, CoM margin 51 mm, up_z 0.945, both sole centres 6.0 cm from their plateau contacts (the pin is active).

Tier B (feet may be re-placed within 20 cm): 3/3 certified (ρ 0.066–0.070), as expected.

**Interpretation, stated honestly:** the solution is essentially *standing on the plateau footprint*. It exists because a ~0.5 m robot can stand with its feet where they already are. The pre-registered "reach" definition constrains only the feet, so it makes existence easy. The CoM must move ~22 cm forward (x 0.234 → 0.456). Whether that move can be made statically is the real question, and step 2b addresses it.

## 3. Step 2b: is there a quasi-static load transfer (lift-off configuration)?

Commands: `e3_step2b_liftoff.py` (stopped after L1, max-margin and 2 L2 solves, for time) and `e3_step2b_L3only.py`. Logs: `results/step2b_stdout_partial.log`, `results_step2b_L3.log`, `results/step2b_log.json`, `results/step2b_L3_log.json`.

A lift-off configuration has a non-foot support still touching the floor (≤ 5 mm) while the **feet-only** certificate already holds. Without one, the last non-foot support cannot be released at zero load.

| Variant | Solves | Certified lift-off | Best CoM margin to the feet polygon (+ = inside) |
|---|---|---|---|
| L1: R_hand kept at its plateau xy (±3 cm) | 6 | **0** | −51.6 mm (hand z 5.3 mm, pen −1.8 mm); max-margin solve with hand valid: −63 mm. Forcing the CoM inside (w_com = 100) lifts the hand 42 mm and penetrates 31 mm. |
| L2: R_hand re-placed anywhere | 2 (of 4 planned) | **0** | −19.3 mm (pen −2.8 mm) |
| L3: pelvis/trunk touching anywhere (sit-squat) | 2 | **0** | −16.0 mm (pelvis z 21 mm, pen −8 mm: constraints violated) |

Reading:
- **In every local solve, a hand or pelvis touching the floor and a CoM over the feet were mutually exclusive.** The arms are too short and the deep squat too limited for the hand-supported and feet-only certified sets to overlap.
- B's plateau stalls exactly at this transition (non-foot load 53 N, all on the right hand).
- Caveat: these are local least-squares searches (≤ 6 starts per variant), **not a proof of non-existence**. I did not search kneeling or both-hands supports.

## 4. Step 3: augmented basis from the plateau, same `score()`

Command: `e3_step3_search.py --arm {orig12|aug16} --seed S --depth 18` ([e3_step3_search.py](e3_step3_search.py)).
- **Protocol** = `codex/ground_search.search` loop: width 14, 16 branches, half symmetric, branch 0 = zero action, B's diversity cells.
- **Start** = plateau snapshot. rng = 2027 + S.
- **Budget**: 18 levels = **3,824 evaluations**, matched to the hybrid run's 3,864.
- **Augmented basis** (gs.B/BOUNDS swapped in-process, files untouched): differential hip roll (0.30), differential ankle roll (0.30), mirror ankle-roll spread (0.30), asymmetric knee (0.50).
- **Success** = B's handover criterion followed by a 1.2 s zero-action hold. This is my stand-in for `balance()`, which needs osqp/`experiment.Controller`. Nominal standing passes this hold.

| Arm | Evals | Success | End state (beam best) | Rollouts meeting the handover criterion (score; survived hold?) |
|---|---|---|---|---|
| orig12 s0 | 3,824 | **no** | unchanged plateau (score 3.884) | 0 |
| orig12 s1 | 3,824 | **no** | unchanged plateau (score 3.884) | 0 |
| **aug16 s0** | 3,824 | **no** | score 3.893, base 0.260, up 0.806, **feet 100 N, nonfoot 0 N**, KE 0.57 J, CoM speed 0.29 m/s (transient feet-only, not at rest; 7 blocks) | 2: base 0.165 / up 0.922 / nonfoot 0 (score 3.228; fell: base 0.12, nonfoot 65 N) and base 0.194 / up 0.950 (score 3.576; fell) |
| aug16 s0 + score v2 (**post-hoc**: score − 0.8·min(nonfoot/108, 1)) | 2,032 (10 levels, time-limited) | **no** | base 0.241, up 0.768, feet 109 N, nonfoot 12.5 N, KE 0.03 J | 2: up 0.908 and 0.987, nonfoot 0 (both fell back under the hold, nonfoot 80–84 N) |

Seeds 1 and 2 of aug16 and seed 2 of orig12 were pre-registered but **not run** (time). This is n = 1 for the augmented arm.

What differs between the arms:
- The augmented basis (with either score) **does reach transient feet-only states**: nonfoot 0 N, up_z up to 0.99. The original basis never did in 7,648 evaluations.
- With the original `score()`, those states rank **below** the moving plateau snapshot (3.23 / 3.58 < 3.884), so the beam never adopts them. This is a score pathology: a transient, high-scoring snapshot pins the beam.
- **None** of these states is a static equilibrium that survives a 1.2 s hold. All are dynamic pass-throughs that fall back.

Step 3c (post-hoc, `e3_step3c_guided.py`) tested direct PD tracking of the certified target. With a full 24-D joint basis, I drove from the plateau toward the tier-A equilibrium (or nominal standing) in K = 1/2/4/8 blocks. **0/8 succeeded**: the robot falls back to sitting or supine (best: K = 4, base 0.208, up 0.71, nonfoot 54 N). Open-loop PD toward a statically certified target is not a quasi-static path, which is consistent with §3.

## 5. Falsifier and verdict

Pre-registered rule, applied literally:
- (b) requires "no feet-only equilibrium within reach". **False**: 9/9 certified tier-A equilibria, ρ ≤ 0.095. → **(b) rejected.** In the existence sense, B's plateau is not the same kind of gap as A's 40–45° roll gap (where no equilibrium exists at those angles at all).
- (a) requires equilibrium exists **and** augmented basis or score succeeds. The equilibrium exists, but there were **0 successes** (aug16 1 seed; score-v2 1 post-hoc run). → **(a) not established.**

Outcome: **NEITHER (a) NOR (b)**, the case the pre-registration anticipated.

What the evidence does support (weaker, local, not pre-registered as decisive):
1. **Torque authority is not the limit.** ρ ≤ 0.16 at the plateau, ≤ 0.1 at the feet-only equilibria, and ≤ 0.57 along the entire best path.
2. **There is a likely connectivity gap at the last support release.** No lift-off configuration was found for hand-pinned, hand-re-placed or pelvis-touching supports (best margin −16 to −63 mm). So hand→feet transfer from this posture appears to need a non-static interval, i.e. a momentum transition: a "support-polygon jump" in the *connectivity* sense.
3. **The search is also part of the failure.** The original 12-synergy basis never produced a feet-only state from the plateau. The augmented basis did (transiently), and B's `score()` ranked those states below the moving plateau snapshot. A closed-loop stabiliser at the moment of release (B's QP `balance()`, not reproduced here) may be what is missing. That is untested.

**One-line verdict: NEITHER (a) NOR (b). Feet-only equilibria exist near the plateau (ρ ≈ 0.07, so it is not a statics-existence gap and not a torque limit), but no augmented-basis or score search succeeded (0/2), and no hand/pelvis lift-off configuration was found (a likely connectivity gap that needs momentum); the "common support-polygon jump" story survives only in the weaker connectivity sense.**

## 6. Limitations
- Lift-off searches are local least squares with few starts. A certified lift-off may exist outside them, for example with knees down or both hands down. That would turn point 2 above into a search failure.
- The hold test substitutes for B's QP `balance()`. The original hybrid's handover would have invoked `balance()` on states meeting the criterion. Here the handover states were never ranked first, so `balance()` would not have been called anyway.
- Step 3 used the physics-only beam protocol, not the network-screened hybrid, and was under-replicated: aug16 n = 1, orig12 n = 2.
- The plant adaptations in §0 (mesh hull shell, B's foot definition, self-contacts carrying no load) can change ρ. They cannot turn an infeasible LP feasible, except for self-contact, which is excluded conservatively.
- Literature: none cited. Contact-stance multi-contact planning (Hauser/Bretl/Latombe) is the relevant framework for "connectivity of support sets" (**UNVERIFIED**, from memory).

## 7. Files
- Code: `e3_plant.py` (B shim, unmodified B code), `e3_cert.py` (port), `e3_step1_plateau.py`, `e3_step2_feet_ik.py`, `e3_step2b_liftoff.py`, `e3_step2b_L3only.py`, `e3_step3_search.py`, `e3_step3c_guided.py`.
- Results: `results/` (step1/2/2b JSON, npy states), `results/step3/` (per-run JSON, best npz, handover-state npz, logs; `v1_nosave/` = first aug16 s0 run, identical trajectory, without saved handover states), `results_step*.log`.
