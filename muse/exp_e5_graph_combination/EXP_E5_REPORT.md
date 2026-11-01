# EXP E5 — Graph-structured get-up search (B's model as edge-proposal oracle inside A's mode graph)

**Status: COMPLETE 2026-11-01.** 40/40 runs ok (0 timeouts, 0 errors).
Wall time ≈ 70 min on 4 workers (2 CPUs).

## 1. Question

E1 falsified the claim that B's contact-attention network adds *search value*
as a flat proposal screener (net 6/20 vs rand 4/20 vs heur 7/20; net not better
than rand p=0.42/0.72, not better than heur p=0.59/1.00). E5 asks the
combination question: **does B's network gain value when used as the
edge-proposal oracle inside A's interpretable mode graph?** I.e., is the
graph+model combination more successful *per evaluation* than flat search?

## 2. Arms (n=20 each, same 4000-eval budget, same 20 seeds as E1)

| Arm | Description |
|-----|-------------|
| F | **Flat baseline**: E1's net-arm runs, reused as-is from `../exp_e1_proposal_control/runs/e1_net_*.json` (NOT re-run). Beam search maximizing `score()`, network screening top-8+4random. |
| G | **Graph + network**: mode-structured search; per DAG edge, beam search with E1's network screening (`arms.select_network`, top-8+4random). |
| GR | **Graph + random**: identical graph structure, but 12 uniform-random proposals per beam state (`arms.select_random`) — separates the "graph" effect from the "model" effect. |

Poses/seeds: prone 1000–1009, supine 2000–2009 — identical to E1, and the
initial `passive_wide` settle is imported unchanged from E1's `run_e1.py`, so
initial conditions are paired by seed across all three arms.

## 3. Mode predicates (approximations on the 24-DoF mesh plant)

B's network only works on the 24-DoF mesh plant (`pi_plus_24dof_250826`);
A's predicates were written for the 20-DoF primitive plant. For fairness ALL
E5 arms run on the same 24-DoF plant, with these lightweight re-implemented
predicates (`modes.py`) — documented approximations:

- Signals: base rotation `R` (world-from-body); `up_z = R[2,2]` (spine axis),
  `lx_z = R[2,0]` (local x = chest direction, world-z), `ly_z = R[2,1]`
  (local y = left direction, world-z); contact node weights `w[0..24]` from
  `GroundWorld.contacts()`; `foot = w[6]+w[12]` (ankle_roll bodies).
- **SUPINE**: `lx_z > +0.6`, `base_height < 0.25` (chest faces up, low)
- **PRONE**: `lx_z < -0.6`, `base_height < 0.25` (chest faces down, low)
- **SIDE**: `|ly_z| > 0.7`, `base_height < 0.25`
- **STAND**: `base_height > 0.30`, `up_z > 0.95`, `foot_load/total_load > 0.80`
- else **TRANSITION** (unclassified)

Grounding (measured 2026-11-01 on settled `passive_wide` states): the local-z
(spine) axis is horizontal in both supine and prone (`up_z ≈ 0.1`), so
supine/prone are NOT separable by `up_z` on this plant — the separating axis
is the chest direction (`lx_z ≈ +0.995` supine vs `≈ -0.994` prone).
`classify()` was verified on settled states before the sweep.

Mode DAG: `SUPINE → SIDE → STAND`, `SUPINE → PRONE → STAND` (descent allowed:
the achieved state is re-classified after every edge and the controller
continues from whatever DAG node it actually landed in).

## 4. Graph controller (per run)

1. Initial settle via E1's `passive_wide`; mode = `classify(state)`.
2. Loop while `evals < 4000`:
   - Pick an outgoing DAG edge (SUPINE tries SIDE first, then PRONE; prone
     initial states start at PRONE → STAND). Max 2 attempts per directed edge.
   - **Edge search**: width-14 beam, ≤12 levels, **≤1400 evals/edge**, ranked by
     `edge_score = score(world) + 3.0 * mode_term(dst)` with
     `mode_term(PRONE) = -lx_z`, `mode_term(SIDE) = |ly_z|`,
     `mode_term(STAND) = 0.5*up_z + 0.5*min(base_height/0.45,1) + 0.5*foot_frac`.
     Same 64-proposal pool, same diversity cells, same `advance()` as E1.
   - **Edge feasible** when a physics rollout lands in the destination
     predicate AND still satisfies it after a 150-step passive settle
     (settle costs 0.5 eval, counted — a "mode" must be a resting config,
     not a transient pass-through).
   - STAND reached → **final handover = E1's criterion**
     (`base_height>.16, up_z>.9, foot_load>75, nonfoot_load<8`) **+
     `balance()`** (vendor, unchanged). If handover fails, a STAND→STAND
     refinement edge (feasibility = handover criterion) runs on the remaining
     budget before giving up.
3. `evals_to_success` = total physics evals (`advance()` = 1, settle-verify =
   0.5). `balance()` physics is NOT counted (same convention as E1).

**Budget split (documented)**: 1400 evals per edge attempt, 4000 total per run.
A run that burns two failed first-edge attempts has ≤1200 evals left for the
STAND edge — this asymmetry vs flat search is inherent to the graph design and
is exactly what the experiment measures.

**Matched candidates**: G and GR share the candidate RNG namespace
`rng([seed, edge_idx, attempt, level, beam_idx])`, so at any (seed, edge,
level, beam state) both see identical 64-proposal pools; only the selection
rule differs. Arm F is E1's stored runs (different candidate stream, but
seed-paired initial conditions).

## 5. Results

Bottom line: **the graph failed completely — 0/20 in both graph arms vs 6/20
flat. The STAND edge was never feasible in 40 attempts.**

| Arm | Success | Rate | 95% Wilson CI | Median evals | ok/timeout/err |
|-----|---------|------|---------------|--------------|----------------|
| G (graph+network) | 0/20 | 0.00 | [0.00, 0.16] | — | 20/0/0 |
| GR (graph+random) | 0/20 | 0.00 | [0.00, 0.16] | — | 20/0/0 |
| F (flat+network, E1 reused) | 6/20 | 0.30 | [0.15, 0.52] | 2940 | 20/0/0 |

Edge feasibility (40 runs, 100 edge attempts total):

| Edge | Attempts | Feasible |
|------|----------|----------|
| SUPINE→SIDE | 20 (10 G + 10 GR) | **20/20** |
| PRONE→STAND | 40 (20 G + 20 GR) | **0/40** |
| SIDE→STAND | 40 (20 G + 20 GR) | **0/40** |

The lying-mode edges work fine (SUPINE→SIDE feasible in ~170–790 evals, both
arms). The blocker is systematic: **no beam, network-guided or random, ever
reached the STAND predicate** (base_height>0.30, up_z>0.95, foot_frac>0.80,
quasi-static) within the 1400-eval edge cap. An instrumented re-run of the
PRONE→STAND edge on seed prone/1004 (where E1's flat search succeeded in
1176 evals) recorded **zero STAND-predicate hits in 1400 evals**; the beam
plateaued at edge_score ≈ 6.7 in a feet-planted crouch
(base_h 0.205, up_z 0.728, foot_frac 1.0) — rising early, then stuck.

## 6. Statistics

Pre-registered comparisons (log-rank on evals-to-handover, Fisher on counts;
E1's machinery reused):

| Comparison | log-rank p | O1−E1 (favours) | Fisher p |
|------------|-----------|-----------------|----------|
| G vs F | 0.0814 | −1.77 (favours F) | **0.0202** (favours F) |
| GR vs F | 0.0856 | −1.67 (favours F) | **0.0202** (favours F) |
| G vs GR (exploratory) | 1.0000 | 0.00 | 1.0000 |

G vs GR are identical (p=1.0): **the network adds nothing inside the graph
either** — with the STAND edge unreachable, the selector never matters.

## 7. Falsification verdict

**FALSIFIED on efficiency — and then some.** Graph search (G) is not merely
"not better" than flat search (F): flat search is *significantly more*
successful (Fisher p=0.020, 6/20 vs 0/20). Per the fixed falsifier, the graph
as designed contributes **interpretability only, not efficiency** — here it
contributes *negative* efficiency: the mode decomposition (1400-eval edge
caps with beam restarts, plus the hand-specified `mode_term` shaping) made
the search strictly worse than the flat baseline it was meant to structure.

Mechanistic note (post-hoc, not pre-registered): the STAND edge objective
`score + 3·mode_term(STAND)` rewards `foot_frac` even at low base height
(+1.5 bonus for feet-planted crouches), plausibly penalizing the
hand-supported transitional configurations that flat `score()` permits on
the way up — a "foot-frac moat" around standing. The beam's plateau in a
feet-planted crouch is consistent with this. The edge-cap/restart structure
is a second suspect: E1's flat successes needed up to 3696 *continuous*
evals; the graph restarts the beam from the entry state after each failed
1400-eval attempt, discarding exploration progress.

## 8. Caveats

- Mode predicates are lightweight approximations re-implemented for the
  24-DoF plant, not A's original 20-DoF predicates.
- n=20/arm; arm F reuses E1's runs (not re-run — same seeds, same harness).
- Edge guidance bonuses (`mode_term`, weight 3.0) and the 1400/4000 budget
  split are heuristic design choices, fixed before results.
- **Mid-sweep bug fix (2026-11-01, fully disclosed):** the first version
  required EVERY edge (including STAND) to survive a 150-step *passive*
  (`ctrl=0`) settle-verify. A standing humanoid is not passively stable, so
  this was a guaranteed reject of genuine STAND arrivals. After 5/40 runs
  (all PRONE→STAND failures under the buggy rule) the rule was changed:
  lying-mode edges keep the passive settle-verify; the STAND edge requires
  the predicate at end of advance + `kinetic_j < 0.10` (quasi-static gate;
  measured end-of-advance kinetic median 0.037 / p90 0.079). The 5 affected
  run JSONs were deleted and re-run under the fixed code. The fix can only
  have *helped* G/GR reach STAND; any remaining failures are the design's
  honest outcome.
- Settle-verify (0.5 eval, lying edges only) may push an edge ≤0.5 eval over
  its 1400 cap; the 4000 total is hard.

## 9. Reproduction

```
cd muse/exp_e5_graph_combination
<venv>/python run_e5.py --workers 4     # 40 runs, streams runs/e5_*.json
<venv>/python analyze_e5.py             # summary.json + tables
```
venv: `/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python`
(mujoco 3.17, torch CPU; `torch.set_num_threads(1)` perf fix as in E1).
Per-run JSONs: `runs/e5_{g,gr}_{prone,supine}_{seed}.json`.
