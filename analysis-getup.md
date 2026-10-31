# Get-up from lying via body-schema / contact-schema exploration: gap analysis against the current pipeline

Date: 2026-10-31. Analysis only. No pipeline code was changed. Two small probes were run to check factual claims. Their scripts and outputs are in `~/.gemini/antigravity-cli/brain/b8c4bf39-ea5c-45d6-b944-3d841d999d84/scratch/` (`probe_supine.py`, `probe_contact_obs.py` and their `.out` files). Both run in under 2 s with `sim/.venv/bin/python`. The results are quoted in §6.

---

## 0. Short answers

| Question | Verdict | The fact that decides it |
|---|---|---|
| (a) Does the current logic match? | **Philosophically yes, structurally no.** The loop "URDF prior → active ID → constrained optimisation → surprise-gated adaptation" is the same loop as the proposal's "explore → learn → invert → (re-)plan". The pipeline is missing the proposal's **outer layer**: switching contact modes plus composing a sequence of imagined sub-goals. | `WBC.control` hard-codes the contact set to the 8 sole corners plus two 6-D foot locks (`sim/wbc.py` contacts block). In the plant, **arms, thighs and knee capsules cannot touch the floor at all** (probe 1). |
| (b) How big is the difference? | **Large at the planning and contact layers, small at the estimation and dynamics layers.** | Four load-bearing differences (§3b). The biggest is fixed vs switching contact. |
| (c) How much must change? | Of the core modules, about **35–45 %** is reusable as-is, about **20 %** needs extension, and about **40 %** is new. The new part is mainly a contact-set estimator, a region-task inverse, a mode sequencer and a get-up curriculum. | The parts that carry the scientific claim (torque-channel residuals, whitening, gating, evidence tests, the URDF-prior ID) carry over. The control layer that *uses* them for get-up does not exist. |

A key reframing that follows from the analysis is: **most of "learning the joint-trend ↔ CoM mapping" is analytic.** While the contacts hold, the mapping is `ΔCoM = J_com N_c Δq`, computed from the URDF plus the masses the pipeline already identifies. What genuinely has to be learned or identified is narrower:
- which contact set is active, and where on the body;
- when the set changes (tipping, rolling, slip);
- the contact properties of the shell (geometry vs URDF, friction);
- actuator limits under sustained load.

---

## 1. The proposal as a research problem

### 1.1 State
- Proprioceptive state: joint positions and velocities, measured joint torques τ (motor current), IMU orientation, gyro and specific force. No F/T sensors and no tactile skin, as on the real pi+.
- Body schema θ_body:
  - URDF kinematics (prior; zero offsets δ are already identified);
  - link masses and CoMs (trunk dm/dx/dy are identified now; dz and limb masses are priors);
  - actuator delay and friction.
- Contact schema θ_contact:
  - collision geometry of each link, i.e. *where* the body can touch a flat plane;
  - friction μ per surface material (rubber sole ≠ plastic shell);
  - contact compliance.
- Contact mode σ ⊂ {trunk-back, trunk-front, head, L/R elbow/forearm, L/R hand, L/R thigh, L/R knee, L/R sole, L/R toe/heel edge, pelvis}. Under the flat-ground assumption, σ is *mostly predictable* from kinematics + IMU: whichever geometry is lowest touches the plane.

### 1.2 What is learned (and of what type)
| Object | Type | Analytic from URDF + identified θ? | Needs data? |
|---|---|---|---|
| Forward kinematic map q → region pose, CoM | forward model | **Yes** (FK, `mj_jacSubtreeCom`) | Only θ_body (masses, zeros), which the pipeline already does in part |
| Contact-preserving map Δq → ΔCoM, Δp_region | forward differential model | **Yes**, given σ: `J N_c`, where N_c projects onto motions that keep the σ contacts | No, except θ |
| Contact force distribution given σ, q, τ | contact model (inverse dynamics) | Yes for the **net wrench per contacting link** when `J_cᵀ` has full rank (probe 2) | Friction, compliance |
| Which σ results from a motion (tipping, rolling, slip) | **hybrid/discrete transition model** | Partially: static stability = CoM over the contact hull plus friction cones. Rolling on curved shell is geometry-dependent. | **Yes**: shell geometry, μ, compliance, impacts |
| Inverse: desired ΔCoM / Δp_region → Δq, τ | inverse model | Solved by optimisation, not learned (QP / NLP) | — |
| Mode sequence for get-up | plan / policy | Graph search over σ with keyframe feasibility | Edge costs and feasibility are validated by trials |

### 1.3 What "imagine" means operationally
"Imagine the head lifting" means **inverse inference on the forward model without executing**:
1. Specify a sub-goal in task space, e.g. Δp_head = +z, or ΔCoM toward the support hull of the next mode.
2. Solve for the motion that achieves it. The motion must respect:
   - the current contact set σ (contacts held, unilateral normal force ≥ 0, friction cones);
   - static or dynamic equilibrium;
   - torque and joint limits.
3. Score the motion by effort.
4. Check whether it is feasible in σ. If it is not, mode σ is insufficient for this sub-goal, and the imagined goal triggers a **contact switch**: "to lift the waist I must put my hands down".

One step of this is exactly what a WBC QP does each tick (task → q̈, τ, f). Over a horizon it is a sequence of quasi-static keyframes / trajectory optimisation. "Imagining" is a rollout of the internal model. The existing pipeline already *has* an internal model (`BodyModel`); it only never queries it counterfactually.

### 1.4 What "minimum effort" means (objective)
For slow get-up, effort is dominated by **holding torque against gravity**, not by kinetic energy. Candidate objectives:
- `J1 = ∫ Σ_i (τ_i / τ_max,i)² dt`: normalised torque-squared. It is a thermal proxy and accounts for the arms (10 Nm) being weaker than the legs (20 Nm).
- `J2 = max_t max_i |τ_i| / τ_max,i`: peak torque ratio, i.e. the **feasibility margin**. This is the one that actually decides success on pi+.
- `J3 = ∫ Σ |τ_i q̇_i| dt`: mechanical work, for comparison only.

Recommendation: minimise J1 subject to J2 ≤ 0.8, plus a stability margin (CoM distance to the support-hull edge ≥ m_s). Today's QP has only a tiny `wt = 2e-3 · ‖τ‖²` regulariser; it is not an effort objective.

### 1.5 Formal statement
Given θ̂ (identified) and current (q, σ):
- find a mode sequence σ_0 = σ_lying, …, σ_K = σ_stand;
- find keyframes q_k that are statically feasible in both σ_{k−1} and σ_k (transition postures);
- find trajectories within each mode,

minimising Σ J1 subject to J2 ≤ 0.8, stability margins, and contact-consistency. "Bring region X to target" (proposal item 6) is the same problem with the terminal condition replaced by `p_X(q_K) = p*`.

---

## 2. Point-by-point mapping (exists / partial / missing)

### 2.1 Start lying; find passively stable rest states
- **Exists**:
  - MuJoCo plant `RealRobot` (`sim/model.py`) with latency, friction, backlash and zero offsets, which applies equally to a lying robot;
  - trunk boxes and the head box collide with the floor (`build_xml`: `contype="2"`).
- **Partial**: `RealRobot.fallen()` (base z < 0.22 or tilt > 50°) defines lying as *terminal failure*. `PiPlus.reset` always places the robot standing on its soles.
- **Missing**:
  - Floor collision for limbs. The limb capsules use `contype=4 conaffinity=4` and the floor `3/3`, so **hip, shoulder and elbow capsules and the knee/calf capsules never touch the floor**. Probe 1 lists the geoms that can touch: 8 trunk/head boxes, 2 small 1×2×5.6 cm calf boxes from the official MJCF, and the 2 soles. There are no thigh or hand geoms at all.
  - A lying reset.
  - A rest-state finder: settle from varied postures under limp or low-gain PD, then classify the resulting stable σ.

### 2.2 Active exploration: joint moves → effect on CoM and body parts (flat ground)
- **Exists**:
  - The excitation philosophy: `exploration_targets` (`sim/experiment.py`) is task-relevant, safe excitation (multi-sine sway, load transfer, squat, trunk pitch and yaw) with safety enforced by the QP.
  - The safe-motion observability scan (`zero_motion_scan.py`).
  - Information/Fisher machinery to rank excitations (`ZeroEstimator.information`, `zero_torque_scan.py`).
- **Partial**: All of it is specific to double support. Excitation is expressed as CoM/DCM targets inside the foot support polygon.
- **Missing**:
  - Excitation primitives for lying (single-limb lifts, limb sweeps, pushing against the floor);
  - a safety notion for lying (avoid unplanned roll-over, joint-limit and self-collision safety);
  - an information-driven choice among primitives in a new σ.

### 2.3 Learn contact: contact forces vs joint torques, where the body touches, how forces redistribute
- **Exists (the strongest overlap)**:
  - Contact wrench reconstruction from joint torques: `WBC.estimate_grf` and `SchemaEstimator.add` (`F = J_legᵀ⁻¹(h − τ)`).
  - The full-torque residual `TorqueChannel` (`sim/zero_torque.py`): exact solve for the contact wrenches with the floating-base rows as residual, a noise model including stall friction, exact propagation, Cholesky whitening, and co-identified Coulomb friction (μ, v0).
  - Contact on/off hysteresis from GRF + foot height (`WBC.control`).
  - Evidence tests and gating (`ZeroEstimator` commit/evidence logic, `torque_consistency`).
- **Partial**: All of it assumes the contacting bodies are exactly the two feet and inverts the 6×6 leg Jacobian. The general form is "solve `J_c(σ)ᵀ F = M q̈ + h − Sᵀτ` for the stacked F of an arbitrary σ".
- **Missing**:
  - Contact-set hypothesis testing across candidate σ (this is the natural generalisation of the existing evidence test);
  - geometric contact prediction (lowest geometry under the flat-ground prior);
  - identification of shell friction and compliance;
  - a treatment of unobservable directions, which probe 2 shows are real:
    - **the force distribution *within* one link is not observable**, only the net wrench, i.e. normal force + CoP + tangential force + yaw moment;
    - **knee + flat foot on the same leg is rank-deficient by 2**.

### 2.4 Learn the joint-trend ↔ CoM mapping
- **Exists**:
  - `BodyModel.com_jac` (`sim/wbc.py`) gives the CoM Jacobian and its derivative from the identified model;
  - the CoM task in the QP uses it every tick;
  - the identified trunk_dm/dx/dy feed it.
- **Partial**: This is the free-floating Jacobian. The world-frame effect of Δq depends on σ, through `J_com N_c(σ)`. The pipeline only ever uses σ = both feet.
- **Not needed as a learned network (analytic)**:
  - While contacts hold, the mapping follows from FK + masses. The joint columns of the base-frame CoM Jacobian are `Σ_limbs m_i J_i / M_tot`.
  - Trunk mass error only rescales them by `M/(M+dm)`: 6.8 % for the +0.8 kg seed-0 truth (probe 1), and this is already corrected by the identified trunk_dm.
- **Needs data**:
  - The **non-smooth part**: what happens when a joint motion breaks static stability or makes the body roll on its shell. This depends on shell geometry and friction.
  - Limb mass errors. They are not identified today; lying poses with limbs in the air give new gravity directions to identify them.

### 2.5 Invert: imagined CoM / region change → movement; compose the get-up with minimum effort
- **Exists**:
  - A one-tick inverse: the WBC QP maps task accelerations (CoM, trunk orientation, swing-foot 6-D pose, posture) to q̈, contact forces and τ, under friction-cone, torque-limit and joint-limit constraints (`WBC.control`). The swing-foot task is already an operational-space "bring region X to pose" task.
  - A torque regulariser.
- **Partial**:
  - The tasks are tracking-reference (gait reference, DCM planner `GaitPlanner` with modes `stand` / `walk`), not "goal + effort".
  - Phase 3's outer optimisation is unfinished even for walking (plan.md).
- **Missing**:
  - multi-step / keyframe inverse (quasi-static trajectory optimisation);
  - an effort objective (J1/J2);
  - transition-posture search;
  - a mode graph and sequencer;
  - a counterfactual "imagine" API that queries `BodyModel` without executing.

### 2.6 Generalise: "bring region X to target" for head, hand, leg alike
- **Exists**: the generic machinery (`BodyModel.jac6`, `point`, `jacp_point`) for any body and any point. The swing-foot task code is body-agnostic apart from the hard-coded `self.feet[sw]`.
- **Partial**: The task list is fixed in code (CoM, trunk, swing foot, posture).
- **Missing**:
  - a task API `(body, local point/axis, target, weight/priority)`;
  - the same API for contact regions (so a support region is just a region with a "stay" target and a force variable);
  - an evaluation on held-out regions and targets.

---

## 3. The three questions

### (a) Does the current logic match the proposal's logic?
**At the epistemic level, yes.** Both say:
- start from a kinematic prior;
- explore actively to identify what the prior gets wrong, as far as it matters for the task;
- act by optimisation on the internal model;
- adapt when predictions fail.

The current "predictive body schema + active ID + constrained control" maps one-to-one onto the proposal's "explore → learn mapping → invert", *within one contact mode*.

**At the architectural level, no.** The proposal has an extra layer the pipeline lacks entirely: **compose**. Get-up is a *sequence* of imagined sub-goals, and each one may require a *different contact set*. The pipeline's control is single-mode reactive tracking. Its "planner" is a footstep/DCM generator, and its exploration only exists in double support.

There is also a difference in emphasis:
- The proposal says "learn the mapping".
- The pipeline says "identify a few parameters, compute the mapping".

The analysis (§2.4) sides largely with the pipeline: the mapping should be *computed*, and learning reserved for the residual hybrid and contact parts.

### (b) How big is the difference? The load-bearing differences
1. **Fixed vs switching contact sets (largest).**
   - Lying → sitting → hands+feet → squat → stand (or a prone/quadruped route) is multi-contact planning with discrete mode changes.
   - The current WBC hard-codes feet (FOOT_CORNERS, 6-D foot locks, `stance ⊂ {l, r}`).
   - The plant cannot even produce arm, knee or thigh contacts (probe 1).
   - The contact estimator only knows on/off per foot.
2. **Parametric ID vs a learned mapping (smaller than it looks).**
   - The contact-preserving joint ↔ CoM / region mapping is analytic given θ.
   - What is missing is (i) the discrete transition model (stability boundaries, rolling, slip) and (ii) contact-schema parameters (shell geometry vs URDF, μ_shell, compliance).
   - Contact *distribution* within a link is unobservable without tactile sensing (probe 2, set G). The proposal's "how forces distribute over changing contact regions" therefore has a hard limit: net wrench + CoP per link, not a pressure map.
3. **Task representation.** Today the tasks track a gait reference plus a DCM plan. The proposal needs "bring region X to target with minimum effort": operational-space goals plus an effort objective plus feasibility margins. The QP has the right *shape* (weighted tasks + constraints) but the wrong *content* and no horizon.
4. **Exploration curriculum and starting point.** Today: standing → double-support calibration → stepping. Proposal: passively stable rest poses as the start and as safe anchors, which the pipeline treats as failure (`fallen()`). There is an upside: lying and quadruped postures rotate gravity in the body frame. This makes **trunk_dz observable** (it is a prior today because it is unobservable in upright statics), and also limb masses (analytic argument; not probed).

Not a difference: torque-based contact sensing, whitening, gating and evidence-tested commits. These transfer directly and are the project's strongest asset for this proposal.

### (c) How much would need to change?
These are module-level estimates in relative size, not person-days.

| Module | Status for get-up | Notes |
|---|---|---|
| URDF prior, `BodyParams`, `PiPlus` index helper | **Reuse** | |
| `RealRobot` truth model (delay, friction, backlash, zeros, sensors) | **Reuse** | |
| `build_xml` collision model | **Small extension** | Change the limb capsule bits so they collide with the floor. Add thigh, forearm/hand and knee geoms. Ideally use the STL hulls for the shell. |
| `PiPlus.reset`, `RealRobot.fallen` | **Small extension** | Lying resets. "Fallen" becomes "unplanned mode change". |
| `BodyModel` (dyn, jac6, com_jac, point Jacobians) | **Reuse** | Already body-agnostic. |
| `TorqueChannel` noise model, whitening, SG derivatives, friction co-ID | **Reuse** | |
| `estimate_grf` / `TorqueChannel.eval` contact solve | **Extension** | Generalise from the 6×6 leg inverse to `J_c(σ)ᵀ`, with rank and conditioning checks and a prior on the null directions (probe 2). |
| `ZeroEstimator` / `SchemaEstimator` GN, Schur, evidence, gating | **Reuse (core) + extend** | New parameters: trunk_dz, limb masses, μ_shell. The gates must become σ-aware: no ID during mode transitions. |
| WBC QP skeleton (weighted tasks, friction cones, torque and joint limits, fallback) | **Reuse** | |
| WBC contact block (feet only, 6-D locks) | **Rewrite (moderate)** | Generic contact list `(body, points, normal, μ)`, point contacts, and add/remove with force ramps. |
| WBC task list (CoM / trunk / swing / posture) | **Extension** | Region-task API, an effort term (J1), and margin constraints. |
| `GaitPlanner` (DCM, footsteps) | **Not usable for get-up** | Keep it only for the final standing balance. |
| Experiment stage machine | **New** | Get-up curriculum. |
| Contact-set estimator (geometry prediction + torque-residual hypothesis test) | **New** | |
| Rest-state finder plus lying exploration primitives with information ranking | **New** | |
| "Imagine" API: counterfactual quasi-static inverse on `BodyModel` | **New** | |
| Transition-posture search plus a mode-graph sequencer with effort cost | **New** | |
| Evaluation harness (success, J1/J2, margins, region-target generalisation) | **New** | |

The rough split by function is about **35–45 % reuse as-is, about 20 % extension, about 40 % new.**
- The new 40 % is the *planning/compose* layer and the *contact-mode* layer. These are exactly the two places where the proposal goes beyond the current research question.
- The reused part is the estimation core, which carries the project's scientific claims so far.

There is a dependency caveat. All 9 runs fell, and the delay-compensation bug is the prime suspect. The falls happened in the stepping stages, while slow standing was mostly fine. The end state of a get-up is static standing, so get-up work is **not blocked** by the walking failure to first order. The final squat-to-stand rise is quasi-dynamic, though, and uses the same `delay_comp` path. Fixing or disabling `delay_comp` first is still advisable.

---

## 4. Minimal extension path (stays inside the current philosophy and sim)

**Phase G0: plant and resets (prerequisite, small).**
- Make the limb capsules collide with the floor.
- Add thigh, forearm/hand and knee geoms.
- Add supine and prone resets.
- Redefine failure as an unplanned σ change or a self-collision.
- Quantify how far the primitive shell is from the STL meshes. That gap *is* the contact-schema mismatch to identify later. Use the meshes in the truth plant and primitives in the internal model, the same pattern as θ* vs θ̂ today.

**Phase G1: supine exploration and rest states.**
- Settle from N randomised postures (limp or low-gain PD) and cluster the resulting stable (q, σ).
- These rest states are the safe anchors (the counterpart of quiet standing).
- Lying primitives: lift one limb, sweep, press.
- Rank them by Fisher information for the new unknowns (limb masses, trunk_dz, μ_shell), using the existing `information()` / scan pattern.

**Phase G2: contact-set estimator (reuse the torque channel).**
- For each candidate σ:
  - geometric plausibility from flat-ground lowest points (kinematics + IMU);
  - torque consistency: solve `J_c(σ)ᵀF = M q̈ + h − Sᵀτ`, then whiten the residual with the existing noise model (`TorqueChannel`, `whiten`), and require the unilateral constraints and friction cones on F.
- Posterior over σ; commit with the existing evidence-test logic.
- Report unobservable directions explicitly (same-link distribution; knee + flat foot on one leg) instead of hiding them.

**Phase G3: per-mode identification.**
- GN plus a Schur complement over the extended θ (trunk_dz, limb masses, μ_shell), gated to quasi-static samples within one committed σ.
- Analytic part, not learned: `J_com`, `J_region`, `N_c(σ)`, static equilibrium.
- Learned/identified part: shell contact parameters and a **transition classifier** (does motion Δq from (q, σ) keep σ, tip, roll or slip?).
  - Start analytic (CoM-in-hull, friction-cone margins).
  - Fit a learned correction only if the analytic predictor's error exceeds the falsification threshold (F2 below).

**Phase G4: the "imagine" inverse with region tasks.**
- A quasi-static QP over Δq (and F):
  - minimise J1 + w‖Δq‖²;
  - subject to: `J_X N_c Δq = Δp_X*` (soft or hard), static equilibrium with `J_c(σ)ᵀF`, friction cones, F_n ≥ 0, torque and joint limits, and a stability margin.
- Iterate it as sequential QP to reach keyframes.
- This is the generalisation in proposal item 6: head, hand, foot and CoM are all just `(body, point, target)`.
- Infeasibility in σ is a *signal*: it tells the sequencer a contact must be added.

**Phase G5: sequencing ("compose").**
- Start from a hand-specified mode graph, e.g. supine → long-sit (trunk + pelvis + heels) → hands + feet squat → squat → stand, and a prone variant via quadruped.
- Edges are transition keyframes found by G4 (feasible in both modes). Edge cost is J1, subject to J2 ≤ 0.8.
- Search the graph (Dijkstra).
- Execute with the generic-contact WBC, with surprise gating on the predicted vs estimated σ and the torque residual (the Phase 4 analogue): re-plan on surprise.
- Automate the graph discovery later. It is not needed for the first falsification.

### What the existing MuJoCo sim can validate vs what needs the real robot
| Can be validated in sim | Needs the real robot |
|---|---|
| Contact-set observability and estimator accuracy under the truth model's sensor noise | The real shell geometry, covers and cables vs URDF meshes, i.e. where it really touches |
| Rest-state discovery and mode-graph feasibility under torque limits | Shell friction and compliance, carpet vs hard floor |
| Quasi-static get-up with min-effort plans, and robustness to θ* vs θ̂ (same mismatch protocol as walking) | Motor thermal limits for sustained high-torque poses (sim has no thermal model); accuracy of current-based torque at stall |
| Region-target generalisation | Impact and wear risk to the shell during exploration, and safety procedures |
| Identifiability of trunk_dz and limb masses from lying poses | Validation that the identified θ transfers |

---

## 5. Risks and falsification criteria

- **F1: Contact-set observability.**
  - Claim: σ can be inferred from proprioception alone.
  - Test: in sim with the truth plant, classify σ over all quasi-static samples in full-rank configurations.
  - Falsified if accuracy is below 95 %, or if the estimator claims to separate directions probe 2 shows are unobservable (same-link distribution; knee + flat foot) with better-than-prior confidence. That would be over-fitting noise.
  - Consequence: tactile or contact switches become mandatory, a hardware change for pi+.
- **F2: "Learning the mapping" vs analytic.**
  - Before any learned joint ↔ CoM model, measure the analytic `J_com N_c(σ)` prediction error with identified θ̂ under contact-preserving motions.
  - If the error is within the CoM estimation noise (order mm per 10° joint move), a learned mapping is **unjustified** within modes.
  - Learning is justified only where the analytic transition predictor (CoM-in-hull, cone margin) mispredicts tip/roll/slip in more than 10 % of exploratory moves.
  - This criterion protects the project from building a network that reproduces FK.
- **F3: Quasi-static feasibility under pi+ torque limits.**
  - The arms are 10 Nm, and get-up routes typically load the arms (sit-up push, quadruped).
  - First cheap check: on the nominal model, does a statically feasible keyframe chain from supine (and from prone) to stand exist with J2 ≤ 0.8?
  - If not, get-up needs momentum (swing, roll), and the quasi-static "imagine" framework is insufficient. It would need dynamic trajectory optimisation, which changes the approach. This should be the **first** experiment, since it is cheap and potentially decisive.
- **F4: Task-relevance of identification (the project's core claim, carried over).**
  - On the mismatched plant θ*, the plan built on θ̂ must beat the plan built on the nominal θ: higher success rate or lower J2 at equal success.
  - Falsified if the two are indistinguishable across seeds. This is analogous to the q0 result "Zero-ID does not change the locomotion outcome". It would mean get-up is dominated by unmodelled contact/shell effects rather than by the identified body schema.
- **F5: Generalisation of region tasks (proposal item 6).**
  - The same solver and the same identified model must reach held-out (region, target) pairs, e.g. a hand target while sitting, or the head height while kneeling, without per-region tuning.
  - Falsified if per-region weights or gains have to be hand-tuned to reach a success rate within 20 % of the trained regions.
- **Risk R1: the primitive collision shell in the internal model differs from the truth shell.** This is by design (it is the contact-schema mismatch), but it can dominate. The mesh-vs-primitive gap must be sized in G0.
- **Risk R2: unplanned mode changes during exploration while lying (accidental roll-over).** This is less dangerous than falling while standing, but it corrupts ID data. σ-aware gating is required.
- **Risk R3: scope.** Get-up adds a planning research question (multi-contact sequencing) on top of an unfinished one (Phase 3 outer optimisation; walking falls). A hand-specified mode graph (G5) keeps the *identification* question central and avoids drifting into a planning paper.

---

## 6. Probe results (supporting facts)

**Probe 1** (`probe_supine.py`): which geoms can touch the floor in the current plant.
- Collision filter: floor `contype/conaffinity = 3/3`. Limb capsules `4/4` → `(4&3)|(3&4) = 0` → **no floor contact**.
- Floor-capable geoms: head box, 7 trunk boxes, 2 small calf boxes (0.5×1×2.8 cm half-sizes, official MJCF), 2 soles.
  - Not floor-capable: hip_roll, calf, shoulder_roll and elbow capsules.
  - Not present: thigh, hand and forearm-end geoms.
- Supine at zero pose: contacts are the trunk + both soles; base z = 0.080 m.
- Prone with folded legs: the robot ends up on head + soles (R22 = −0.45). The knee and elbow contacts that a real get-up would use cannot form.
- Trunk mass error vs joint-column CoM Jacobian: total mass 11.01 kg, trunk 3.45 kg. A dm of 0.8 kg gives a 6.8 % uniform scale error, which the identified trunk_dm removes.

**Probe 2** (`probe_contact_obs.py`): whether contact forces are observable from τ (+ floating-base rows). The test is the rank of `J_cᵀ`; 3 generic configurations gave identical ranks.

| Contact set | nf | Rank | s_min/s_max |
|---|---|---|---|
| A supine, trunk patch | 6 | 6 | 1 |
| B trunk + 2 soles | 18 | 18 | 6e-3 – 1.5e-2 |
| C sit: trunk + 2 soles + 2 hands | 24 | 24 | 7e-4 – 6e-3 (ill-conditioned) |
| D quadruped: 2 knees + 2 toes + 2 hands | 18 | 18 | 5e-3 – 9e-3 |
| E kneel with **flat** feet: 2 knees + 2 sole patches | 18 | **16** | 0 (2 directions unobservable) |
| F squat: 2 soles (the current WBC case) | 12 | 12 | 3e-2 – 5e-2 |
| G trunk split into 3 point regions | 9 | **6** | 0 (only the net wrench) |

Interpretation:
- The user's intuition that contact forces are observable through joint torques holds for every *net-per-link* contact set a get-up uses, except knee + flat foot on the same leg.
- It does **not** extend to a pressure distribution within a link.
- Multi-contact sets are 10–50× worse conditioned than today's squat case. Expect noisier contact estimates, and plan for priors and whitening (already in the code base).
