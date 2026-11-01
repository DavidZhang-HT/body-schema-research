# De-networked combination — build and evaluation

**Complete vertical slice on the official 24-DoF pi+ MuJoCo plant.** The default
configuration discovered **3/3 SUPINE→STAND and 1/3 PRONE→STAND successes**.
All four reproduce in uninterrupted torque-driven replay and pass the unchanged
five-second QP stabilizer. All twelve saved action paths (including failures) reproduce
with **zero full-snapshot error at the pre-QP endpoint**; QP outcomes also match. This is a working simulation build,
not a complete or generally reliable get-up system.

Branch: `combo-denetworked`; original baseline: `cb17ccd`. No upstream plant,
search, or balance-controller source was modified. No learned/attention network
is loaded or used. No historical successful action sequence seeds these runs.

[Full successful supine get-up video](results/dynamic/supine_0_getup.mp4) ·
[Final front/side frame](results/dynamic/supine_0_final.png) ·
[Detailed final-run transition tables](RESULTS_dynamic.md) ·
[Preserved failed development sweep](RESULTS.md).

The video contains the complete 24-block search trajectory and QP phase
(~19.4 s simulated time), displayed at labeled 2× playback: H.264, 1024×384,
12 fps, 117 frames, 9.75 s. Its 492,448-byte size permits committing it without
adding a large binary. Full traces and incomplete renders are ignored.

## Evaluation and accounting

Final configuration: `--selection dynamic` (now the default). Seeds 0, 1, 2
for each pose; B's original small-noise `passive` starts: roll ±0.05 rad,
yaw ±0.1 rad, joint noise σ=0.015 rad, followed by passive settling. All starts
settled. These are **not E1's wider starts**. Each search gets at most 6,000
0.6-second action rollouts. Initial settling and QP calls are additional.
Success requires `balance()`'s sustained terminal condition, not a momentary
STAND predicate. In particular, it must finish with at least 50 consecutive
20-ms checks of base>0.30 m, up_z>0.95, feet>80 N, nonfoot<5 N, COM speed<0.12 m/s.

| Start | Seed | Search + replay | Action evals | Winning/best blocks | QP attempts | QP simulated s (all attempts) | Search-loop wall s |
|---|---:|---|---:|---:|---:|---:|---:|
| supine | 0 | SUCCESS | 5607 | 24 | 5 | 11.62 | 391.4 |
| supine | 1 | SUCCESS | 4235 | 16 | 53 | 60.76 | 406.6 |
| supine | 2 | SUCCESS | 3542 | 14 | 6 | 21.60 | 248.9 |
| prone | 0 | FAIL | 6000 | 16 | 19 | 46.54 | 445.4 |
| prone | 1 | SUCCESS | 4986 | 19 | 77 | 181.74 | 631.6 |
| prone | 2 | FAIL | 6000 | 18 | 1 | 1.20 | 354.4 |

Observed final success rates: **SUPINE 100% (3/3), PRONE 33% (1/3)**. These small
samples do not establish robustness or an efficiency advantage over B/E1/E5.
The four successful executed motions last 13.4–19.4 simulated seconds, including
five seconds of QP stabilization. Planning takes minutes, not real time.

**All attempted development runs are retained.** The initial `--selection robust`
version ran the same six starts and had **0/6 successes**, exhausting 6,000 evaluations
each with zero QP entries. Its final bases were 0.101–0.172 m, foot loads 0–38.6 N,
and nonfoot loads 69.6–107.4 N. The revised configuration was then run afresh on
all six seeds. Thus there were **12 completed search runs in two configurations**,
not a selected subset of six. Do not pool the repeated starts as independent
samples. This is build iteration, not a preregistered comparison or an isolated
ablation of one design choice.

Two search workers ran with one BLAS thread each on the two-CPU VM. One short
low-priority replay overlapped the final pair after RAM became available
(maximum three tasks); the other explanations ran sequentially. Wall times
exclude model loading/initial settling and are not controlled benchmarks.
A VM restart interrupted the initial seed-0 development runs; they resumed at
240 supine / 464 prone evaluations. Their counters preserve checkpointed work,
but omit any discarded partial level and downtime. The revised runs completed
without such a restart.

## What was built

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


The default shared beam has width 14 and 16 physics proposals per state: half
coordinated, one null action, remaining asymmetric proposals. Six beam seats
retain dynamic exploration candidates; two seats represent modes; the remainder
uses robust score v3 with posture diversity. Every eligible candidate is tested
by the actual QP, even if it is not the highest-scoring candidate. For example,
supine seed 1 succeeded with score 2.255 after a preceding leader at 3.187.

A source-state expansion tests multiple outgoing mode edges in one batch:
rollouts are classified by their landing mode. The graph therefore organizes
shared exploration rather than prescribing a single sequence or restarting
search at each mode boundary. No per-edge 1,400-evaluation cap is used. A saved
representative witness for **every distinct observed edge** is independently
replayed and explained. Other proposal endpoints are classified/count-logged,
but not all thousands of rejected candidates receive an LP solve. Unobserved
edges are not declared impossible.

## Mode graph and porting choices

The graph is complete and directed over SUPINE, SIDE_L, SIDE_R, PRONE, SIT,
ALLFOURS, HAND_SUPPORT, CROUCH, STAND and TRANSITION. Self-mode refinement,
descent and both roll directions are allowed. P-F3 is only a 0.05 score penalty
on direct SUPINE→SIT/CROUCH/STAND transitions, never a hard rule. Static
feasibility is an explanation, not a gate blocking dynamic transitions.

The classifier tests these predicates in order; touching means an actual
MuJoCo floor contact with distance <=5 mm (the LP independently uses shell
vertices within 5 mm). Arms include wrist, elbow and shoulder bodies; lower
limbs include calf and ankle bodies.

| Mode | Predicate on the 24-DoF plant |
|---|---|
| STAND | Both ankle-roll bodies touch, base>0.30 m, head_z>0.95, nonfoot load<5 N |
| SUPINE | Trunk touches, chest_z>0.7 |
| SIDE_R / SIDE_L | Trunk or arm touches, base<0.25 m, left_z>0.7 / left_z<-0.7 |
| PRONE | Trunk touches, chest_z<-0.7 |
| ALLFOURS | Arm and lower-limb contact, trunk off, chest_z<-0.5; not necessarily four supports |
| HAND_SUPPORT | Arm and lower-limb contact, trunk off; includes forearm support |
| SIT | head_z>0.5 and trunk/hip/thigh contact |
| CROUCH | Both ankle-roll bodies touch, head_z>0.5, nonfoot load<8 N |
| TRANSITION | None of the above |


These are explicit adaptations of A's predicates, not an identical port of its
20-DoF shell regions. B merges pelvis/back/chest in one trunk body, so SIT cannot
use A's exact pelvis/back separation. ALLFOURS and HAND_SUPPORT have deliberately
broad limb-support meanings. There is no mode hysteresis: short contact changes
can produce classification flicker. Raw 100-ms sequences are retained.

## Score v3 and the two pathologies

The **robust ranking** uses
`P = 4 COM_z + 6 base_z + 1.2 up_z - 2 max(foot_z-COM_z,0)` and
`score_v3 = min_terminal_samples(P) - 0.35 log(1+max_terminal_KE/0.03)
- 0.3 max_terminal_COM_speed`. Samples span the final 0.2 s of each existing
0.6 s rollout. No additional passive hold is imposed.

* **E3 transient snapshots:** the sampled minimum discounts height spikes;
  terminal energy/speed lower their score. Retained parents expire after two
  levels, so a moving plateau cannot survive indefinitely without advancing
  physics. At identical geometry, more terminal KE always lowers robust rank.
  This is not a universal ordering of different geometries or a stability proof.
  QP tests on every eligible candidate keep a lower-ranked capturable state from
  being hidden behind the leader.
* **E5 foot-fraction moat:** robust rank contains no foot fraction, foot-load
  bonus, nonfoot penalty, or STAND-edge bonus. Hand support has no tax in this
  ranking. No edge restart destroys continuous exploration.
* **Preserving momentum:** the robust-only development sweep over-preferred slow
  low postures and never reached handover. The final design therefore reserves
  six exploratory seats ranked by B's original instantaneous priority. Its
  bounded bonus is **absolute** `0.6 min(foot_load/108,1)`, not `foot/total`.
  Adding a supporting hand at fixed foot force cannot reduce that priority
  (regression-tested). It retains B's weak energy/speed and foot-height terms.
  Exploratory rank makes no claim that a moving snapshot is settled progress;
  the same parent expiry and real QP validation still apply.

Thus v3 deliberately separates settled-progress ranking from bounded dynamic
exploration. Removing every foot-related exploration incentive and strongly
penalizing terminal motion did not yield a working get-up in the first six
runs; that failure is not hidden. No claim is made that the graph or revised
score is more efficient than flat B search.

## Explanation layer and label meaning

The executing trajectory is never reset after its initial restore. Diagnostic
reads use a separate `MjData` sharing the same immutable model; the LP has its
own data as well. Every returned block is sampled at 100 ms plus exact endpoints.
QP motion is sampled too. Every saved distinct-edge witness is also replayed.

E3's LP is reused unchanged: friction pyramid μ=0.6, rho cap 0.8, hull vertices
within 5 mm of the floor, shell penetration ≥−1.5 mm and self-distance ≥−1 mm.
Each record contains feasible/ok/reason, regional LP support loads in units of
body weight, support polygon and signed COM margin, all joint torque ratios,
a maximizing joint, contact geometry, mode and KE. LP loads are one hypothetical
static equilibrium distribution, not measured dynamic contact forces.

**The requested binary flag needed a truthful third case:**

* `QUASI_STATIC`: every sample is certified and KE≤0.005 J.
* `STATICS_GAP`: at least one sampled LP is infeasible or rho>0.8.
* `DYNAMIC_UNRESOLVED`: fast motion or invalid geometry without a sampled LP gap.

A fast but statically feasible path is not evidence that momentum is necessary.
Even STATICS_GAP describes this sampled path under the conservative LP; it does
not prove every possible route needs momentum. No momentum ablation or exhaustive
connectivity proof was run. Unsampled intervals can contain additional events.

**Certificate limitation discovered by reading its implementation:** E3 logs
joint-limit margin `jl` but does **not** include it in `cert_ok`. Its positive
certificate therefore must not be read as a hard-joint-limit-compliance proof.
The supine-0 path reaches jl=−0.02852 rad (~1.63° overshoot) under MuJoCo's soft
joint limits. All per-run minima are listed below. Self-contact forces are also
excluded from the static LP. No hard-limit or hardware-valid execution is claimed.

## Mode sequences and certificate summaries

The following are **block-end** sequences, collapsing consecutive repeats and
including the successful QP transition. Raw 100-ms mode changes, every block and
mode-edge label, support loads and torque bottlenecks are in
[RESULTS_dynamic.md](RESULTS_dynamic.md) and each linked explanation JSON.
Failed paths additionally receive a diagnostic QP call during replay; that call
is not counted as a new search attempt or included in their sequences below.

- **supine/0**: SUPINE → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → ALLFOURS → CROUCH → STAND. DYNAMIC_UNRESOLVED=20, STATICS_GAP=5. [Full explanation](results/dynamic/supine_0_explanation.json).
- **supine/1**: SUPINE → HAND_SUPPORT → SUPINE → HAND_SUPPORT → ALLFOURS → HAND_SUPPORT → TRANSITION → CROUCH → STAND. DYNAMIC_UNRESOLVED=11, STATICS_GAP=6. [Full explanation](results/dynamic/supine_1_explanation.json).
- **supine/2**: SUPINE → TRANSITION → SIT → HAND_SUPPORT → TRANSITION → CROUCH → HAND_SUPPORT → CROUCH → STAND. DYNAMIC_UNRESOLVED=8, STATICS_GAP=7. [Full explanation](results/dynamic/supine_2_explanation.json).
- **prone/0**: PRONE → TRANSITION → ALLFOURS → HAND_SUPPORT → CROUCH. QUASI_STATIC=1, STATICS_GAP=7, DYNAMIC_UNRESOLVED=8. [Full explanation](results/dynamic/prone_0_explanation.json).
- **prone/1**: PRONE → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → HAND_SUPPORT → TRANSITION → CROUCH → STAND. DYNAMIC_UNRESOLVED=15, STATICS_GAP=5. [Full explanation](results/dynamic/prone_1_explanation.json).
- **prone/2**: PRONE → ALLFOURS → HAND_SUPPORT. DYNAMIC_UNRESOLVED=14, STATICS_GAP=4. [Full explanation](results/dynamic/prone_2_explanation.json).

| Run | Sampled peak KE (J) | Minimum floor / self distance (mm) | Minimum joint margin (rad) | Final replay endpoint certificate | Final rho / maximizing joint |
|---|---:|---|---:|---|---|
| supine/0 | 2.309 | -0.92 / -0.66 | -0.02852 | True | 0.09 / l_hip_roll_joint |
| supine/1 | 3.525 | -1.97 / -1.63 | -0.02036 | True | 0.06 / l_calf_joint |
| supine/2 | 2.963 | -0.91 / -0.87 | -0.02174 | True | 0.103 / l_ankle_roll_joint |
| prone/0 | 5.052 | -2.25 / -0.18 | -0.02354 | True | 0.129 / l_ankle_pitch_joint |
| prone/1 | 3.892 | -0.79 / -0.65 | -0.02028 | True | 0.115 / l_hip_pitch_joint |
| prone/2 | 3.965 | -0.57 / -0.67 | -0.05663 | True | 0.081 / l_hip_roll_joint |

Supine/1 succeeds in the simulator but transiently exceeds both conservative
geometric certificate thresholds (floor −1.97 mm, self-distance −1.63 mm).
It is **not** a fully geometrically certified trajectory. The rendered supine/0
path stays within those thresholds, though its joint-limit overshoot remains
as disclosed above.

For failed runs, this table includes the diagnostic QP fall, so its final row
state is not the retained search endpoint. Full files distinguish both. A
certificate failure is reported, not used to relabel a physical get-up failure
as success or to hide a successful dynamic execution.

### Supine seed 0: rendered vertical slice

24 synergy blocks, then the unchanged QP. All 25 **block endpoints** satisfy
E3's certificate; seven interior samples have infeasible LPs, flagging blocks
2, 5, 10, 18 and 20 as STATICS_GAP (zero-based). Finite sampled rho peaks at
0.724, below 0.8; these gap samples are not torque-ratio-cap failures. Sampled
peak KE is 2.309 J. Minimum shell/self distances are −0.915/−0.658 mm, within
E3's thresholds, so those LP gaps are not merely penetration-threshold failures.

The final configuration has rho=0.08997, a maximizing left hip-roll joint,
left/right foot LP loads 0.5258/0.4742 body weights and COM margin +71.76 mm.
The terminal sampled 0.98-s tail is QUASI_STATIC, with peak KE 1.44e−5 J.
The complete CROUCH→STAND QP transition is nevertheless DYNAMIC_UNRESOLVED,
because it includes motion before that settled tail.

| Block | Endpoint modes | Transition label | End certified | End rho / maximizing joint | End LP support loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.074 / l_shoulder_pitch_joint | trunk:0.57, L_shank:0.29, R_shank:0.14 |
| 1 | SUPINE → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.078 / l_hip_pitch_joint | trunk:0.10, L_thigh:0.24, L_shank:0.05, R_thigh:0.12, R_shank:0.09, L_forearm:0.20, R_forearm:0.20 |
| 2 | HAND_SUPPORT → SIT | STATICS_GAP | True | 0.050 / l_hip_pitch_joint | trunk:0.50, L_foot:0.10, R_thigh:0.32, R_foot:0.08 |
| 3 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.021 / r_hip_pitch_joint | trunk:0.42, L_thigh:0.19, L_foot:0.10, R_thigh:0.16, R_foot:0.11, L_hand:0.01, R_hand:0.01 |
| 4 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.205 / l_hip_pitch_joint | R_thigh:0.46, R_foot:0.27, L_hand:0.27 |
| 5 | HAND_SUPPORT → SIT | STATICS_GAP | True | 0.163 / r_hip_pitch_joint | L_thigh:0.75, R_foot:0.25 |
| 6 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.193 / r_hip_pitch_joint | L_thigh:0.73, R_foot:0.27 |
| 7 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.074 / r_thigh_joint | L_thigh:0.66, L_foot:0.16, R_foot:0.18 |
| 8 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.085 / r_hip_pitch_joint | L_thigh:0.57, L_foot:0.18, R_foot:0.25 |
| 9 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.087 / r_thigh_joint | L_thigh:0.60, L_foot:0.22, R_foot:0.18 |
| 10 | SIT → SIT | STATICS_GAP | True | 0.092 / r_shoulder_pitch_joint | L_thigh:0.55, L_foot:0.19, R_foot:0.27 |
| 11 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.084 / l_hip_pitch_joint | L_thigh:0.49, L_foot:0.21, R_foot:0.31 |
| 12 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.37, L_foot:0.23, R_foot:0.40 |
| 13 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.129 / l_hip_pitch_joint | L_thigh:0.34, L_foot:0.31, R_foot:0.35 |
| 14 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.076 / l_hip_pitch_joint | L_thigh:0.30, L_foot:0.25, R_foot:0.45 |
| 15 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.42, L_foot:0.20, R_foot:0.37 |
| 16 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.38, L_foot:0.21, R_foot:0.41 |
| 17 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.285 / l_elbow_joint | R_foot:0.44, L_hand:0.56 |
| 18 | HAND_SUPPORT → ALLFOURS | STATICS_GAP | True | 0.082 / l_hip_pitch_joint | L_foot:0.30, R_foot:0.38, L_hand:0.32 |
| 19 | ALLFOURS → CROUCH | DYNAMIC_UNRESOLVED | True | 0.272 / l_hip_pitch_joint | L_foot:0.65, R_foot:0.35 |
| 20 | CROUCH → CROUCH | STATICS_GAP | True | 0.088 / l_hip_pitch_joint | L_foot:0.38, R_foot:0.41, L_hand:0.22 |
| 21 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.220 / l_hip_pitch_joint | L_foot:0.75, R_foot:0.25 |
| 22 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.211 / l_hip_pitch_joint | L_foot:0.73, R_foot:0.27 |
| 23 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.175 / l_hip_pitch_joint | L_foot:0.73, R_foot:0.27 |
| balance | CROUCH → STAND | DYNAMIC_UNRESOLVED | True | 0.090 / l_hip_roll_joint | L_foot:0.53, R_foot:0.47 |

## Where it fails and what binds

* **Initial robust-only search, all six:** low supported postures. Every final
  state misses feet>75 N and nonfoot<8 N; five also miss base>0.16 m. All six
  retained endpoints are statically certified (rho 0.037–0.161). Static torque
  authority at those endpoints does not explain the planning failure. The
  observed stopping constraints are support transfer and finite exploration,
  not a demonstrated physical impossibility.
* **Final prone/0:** 19 eligible handovers, all rejected by the actual QP.
  The retained feet-only crouch has base 0.252 m, up_z=0.966 and KE=0.026 J,
  yet it is outside the verified stabilizer capture set for this run. The last
  tested entry was faster (KE 1.862 J, COM speed 0.477 m/s) and fell to base
  0.080 m / nonfoot load 101.5 N in about 1.02 s. Geometric STAND and static
  feasibility are not sufficient controller-capture tests. This is a measured
  stabilization failure, not an isolated diagnosis of which QP task causes it.
* **Final prone/2:** one transient feet-only candidate failed QP after 1.2 s
  (up_z fell to 0.307). At budget exhaustion the retained state still had
  up_z=0.640 and nonfoot load 27.1 N: upright orientation and last nonfoot
  support release remain unmet. No global absence of a solution is established.

The successful runs also needed 5, 53, 6 and 77 QP attempts respectively. That
cost is substantial and explicitly outside the action-rollout counts. A better
capture predictor is still missing; this build uses real QP trials instead of
introducing an unvalidated learned screen.

## Validation, artifacts and reproduction

Six regression tests pass: cloned augmented-action determinism; standing/supine
LP sanity on the 24-DoF model; robust score pathology checks; no hand-support tax
in exploratory priority; graph/label semantics; terminal-success checkpoint
recovery and rejection of mismatched cached configurations. All twelve paths
have finite 140-value states and 16-dimensional actions. All upstream hashes in
[environment.json](environment.json) match.
All **252** saved distinct-edge witnesses (counted per run, including self-edges) replay with zero snapshot error.

Machine-readable audit: [validation.json](results/validation.json).
Detailed per-block/per-mode tables: [final](RESULTS_dynamic.md) and
[development](RESULTS.md). Video metadata/SHA256:
[video_metadata.json](results/dynamic/video_metadata.json).

From the repository root with the existing meshes and `.venv-exp`:

```bash
.venv-exp/bin/python -m combo.sweep                       # final default, two workers
.venv-exp/bin/python -m combo.sweep --selection robust    # retained development variant
.venv-exp/bin/python -m combo.artifacts                  # sequential replay/explanation + video
.venv-exp/bin/python -m unittest combo.test_combo
```

Matching completed results are skipped; checkpoints resume per level; different
requested configurations are rejected rather than silently reusing old results.
For a single path use `python -m combo.explain --pose supine --seed 0` and
`MUJOCO_GL=glfw xvfb-run -a python -m combo.render --pose supine --seed 0` within
that venv. Rendering uses CPU Mesa, system FFmpeg and Pillow. No EGL or GPU is
required in this VM. Partial renders and full trace archives are not committed.

## What is still missing from a complete get-up system

This remains privileged-state cloned-simulator planning plus open-loop synergy
execution and a terminal QP. It is not a closed-loop get-up policy, is not real
time, and is unreliable on the tested prone starts. It has no verified coverage
of wider poses, disturbances, terrain, model errors or longer standing horizons.
The mode vocabulary is approximate and flickers at contact boundaries. Static
certificates are sampled, conservative, and omit a hard joint-limit gate; they
are neither whole-path safety proofs nor proofs that momentum is unavoidable.
No graph-efficiency advantage, global search completeness, hardware result or
sim-to-real claim follows from this build. The concrete delivered capability is
a replay-verified, explainable dynamic get-up on the requested 24-DoF simulation
plant, with its failed attempts and unresolved cases preserved.
