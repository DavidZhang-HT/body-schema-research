# De-networked combination — build and evaluation

Status: IMPLEMENTED, EVALUATION RUNNING. No completed success claimed yet.
Branch `combo-denetworked`, baseline `cb17ccd`. See CHECKPOINT.md for restart state.

## Plant and components

Official B 24-DoF pi+ mesh plant, 31 qpos / 30 qvel, CPU MuJoCo.
`combo/plant.py` imports real `codex/ground_schema.py` and the unchanged
`codex/ground_balance.py::balance`. E3's plant shim was inspected but cannot be
used with the real QP because it replaces `experiment` with a stub. Its
`BSchema` certifier is reused directly, with no changes to upstream code.
E3's four extra synergies (differential hip roll, differential ankle roll,
ankle-roll spread, asymmetric knee) augment B's 12-dimensional action basis.
All actions use B's torque envelope, smoothing and 0.6 s MuJoCo rollout.
There are no learned networks or attention heads. `attention_data` is imported
only by the unchanged B implementation for its deterministic motor envelope;
no network module or torch is imported (tested).

## Graph and oracle

Modes: SUPINE, SIDE_L, SIDE_R, PRONE, SIT, ALLFOURS, HAND_SUPPORT, CROUCH,
STAND and TRANSITION. Orientation convention follows A; contact regions follow
B. B's single trunk body prevents the pelvis/back split used by A, so SIT is
an approximation. Dynamic mode membership is independent of static feasibility.
Both side directions, descent, skipping modes and self-mode refinement are
allowed. A complete directed mode graph avoids making negative assumptions
about dynamic reachability. A P-F3 prior costs only 0.05 on direct supine lift.

A shared width-14 beam tests outgoing edges with 16 direct random physics
proposals per state (half coordinated; one zero action), no screening.
Four beam seats preserve distinct contact modes; remaining seats use B-style
posture diversity. Landing modes determine which edge a rollout witnesses.
Thus one batch tests multiple possible destination edges. The graph is a
search-structure/explanation layer, not a hand-coded mode sequence. No 1400-eval
edge cap or restart destroys the shared frontier. Success requires an eligible
candidate to pass the actual unchanged five-second QP balance call. Every
eligible candidate is tried, not merely the score winner.

## Score v3 rationale

The progress term is `4 COM_z + 6 base_z + 1.2 up_z - 2 max(foot_z-COM_z,0)`.
Score takes the **minimum** of progress samples in the last 0.2 s of each block,
then subtracts `0.35 log(1 + max KE / 0.03) + 0.3 max COM_speed` in that window.
The 0.2 s is the terminal portion of the existing rollout, not an extra hold.

* E3 transient pathology: temporal minimum penalizes brief high snapshots;
  energy and speed reduce their value. Retained parents expire after two levels,
  so a moving plateau cannot be kept forever without advancing physics.
  Identical geometry with greater terminal KE always ranks lower. This does not
  prove a universal ordering between different geometries, nor guarantee a good
  search landscape. Mid-block KE is not penalized, permitting momentum moves.
* E5 foot-frac moat: no foot fraction, foot-load bonus, nonfoot penalty, or
  STAND-destination bonus exists. A hand-supported state has exactly the same
  score as a feet-only state with identical geometry and motion. Removing those
  terms entirely is stronger and easier to inspect than height-gating them.

## Explanation semantics

Every returned action block is replayed and sampled every 100 ms plus endpoints;
all endpoints are certified. QP samples are also certified. The unchanged E3 LP
uses friction-pyramid mu=0.6, rho<=0.8, hull-vertex contacts within 5 mm,
penetration >=-1.5 mm, self-distance >=-1 mm. Reports include feasible/ok/reason,
region support loads in fractions of body weight, support polygon and signed
CoM margin, all joint torque ratios and binding joint, contact geometry,
kinetic energy and modes. LP loads are hypothetical equilibrium loads, not
measured dynamic contact forces. Sampled certificates describe the returned
trajectory; rejected search proposals are counted/classified but not certified.

**Necessary refinement of the requested binary labels:**
QUASI_STATIC means all samples pass the certificate and KE<=0.005 J.
STATICS_GAP means at least one sampled LP is infeasible or exceeds rho=0.8.
DYNAMIC_UNRESOLVED means motion is fast or geometry invalid without a sampled
LP gap. Fast motion, an endpoint certificate, or a collision violation alone
cannot establish that momentum is necessary. Even STATICS_GAP is path-specific
under the conservative LP, not proof that every possible path needs momentum.
No momentum ablation or exhaustive connectivity proof is claimed here.

## Evaluation protocol

Six runs: SUPINE and PRONE, seeds 0, 1, 2; B's original small-noise passive
start distribution (not E1's wide starts). 6000 action rollouts each, unless QP
success occurs first. Initial settle and QP integration are additional and
reported separately from action evaluations. Two workers, one BLAS thread each.
No saved successful actions seed the search. Results resume from per-level
checkpoints after restart. These are build-validation runs, not a preregistered
comparison against E1 or E5, whose budgets/distributions differ.

Results pending. Four regression tests pass: deterministic cloned rollout,
24-DoF standing/supine LP sanity, both score-pathology checks, graph/label semantics.

## Still missing

Quantitative outcomes and rendered success pending. No closed-loop get-up
policy: this is privileged-state cloned-simulator planning plus open-loop action
replay and QP stabilization. No global search completeness, quasi-static path
proof, necessity proof for momentum, robustness distribution, real-time bound,
or hardware/sim-to-real evidence. Mode geometry is a documented mesh adaptation
rather than an identical port of A's primitive-region vocabulary.

## Development update: dynamic exploration slots

Initial seed-0 runs both exhausted 6000 evaluations without any QP entry.
They retained very slow, low hand-supported states. An optional `--selection
dynamic` revision therefore reserves six seats for instantaneous progress,
two for modes, and the remainder for score v3. Exploration priority uses B's
weak KE/speed costs and an absolute foot-load bonus multiplied by
`clip((base_z-0.20)/0.12,0,1)`: exactly zero below 20 cm. It is a separate
exploration queue, not a settled-state certificate or terminal score. Score v3
and the two-level parent expiry remain unchanged. This variant has separate
output in `results/dynamic/`; baseline attempts remain visible. No variant
outcomes are claimed until runs complete. This is build iteration, not a
controlled causal attribution of the failed design.
