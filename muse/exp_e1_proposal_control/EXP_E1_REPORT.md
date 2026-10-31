# E1: Matched-Budget Proposal Control — Report

## Methods

**Question.** Does attention-based proposal screening (the contact-attention
ensemble's `value()`) add search value over (b) uniform-random proposals and
(c) a hand-crafted FK heuristic, when every arm gets an *identical* physics
budget? Arm (d), a parameter-shuffled copy of the ensemble, controls for
"any scoring function helps".

**Design.** 4 arms x 2 poses (prone, supine) x 10 seeds = 80 runs.
Prone seeds 1000-1009, supine seeds 2000-2009. Every run: budget of 4000
`GroundWorld.advance()` calls (one call = one physics evaluation of a 0.6 s
bounded-schema action, 300 MuJoCo steps at 2 ms), beam width 14, depth <= 24,
the same diversity cells as `vendor/ground_hybrid.py`
(`com_height/0.025, up_z/0.12, (q0+q6)/0.6, (q3+q9)/0.7`), the same
handover gate (`base_height > 0.16`, `up_z > 0.9`, `foot_load > 75 N`,
`nonfoot_load < 8 N`) followed by `ground_balance.balance()`. Starts:
`passive_wide(pose, seed)` — base yaw U(+/-0.6), roll U(+/-0.30),
pitch +/-pi/2 by pose, joint noise N(0,0.08), base height 0.26, same settle
criterion as vendor `passive()`; up to 3 attempts (seed+10000*k), retries logged.

**Matched proposals.** At each (seed, level, beam index) the 64 candidates are
drawn from a dedicated RNG seeded `(seed, level, beam_idx)`, independent of
arm — all arms see identical candidate pools; only the selection rule differs.
Candidate distribution (identical to `vendor/ground_hybrid.py`): 64 proposals
~ N(0,0.7)*BOUNDS clipped; first half have differential synergies zeroed and
hip-spread/elbow damped; proposal 0 is the null action. Each arm selects 12
per beam state for physics rollout; the winning path records a provenance tag
per action (`net`/`heur`/`rand`/`shuf`).

**Arms** (`arms.py`, kept import-clean for E5 reuse):
- (a) `net`: score 64 proposals with the 3-member contact-attention ensemble
  via `value(pred, unc)`; top 8 + 4 uniform random.
- (b) `rand`: 12 uniform-random proposals from the same 64.
- (c) `heur`: FK-only proxy per proposal — set `qpos[7:] = clip(q + B@a)`,
  `mj_forward`, then `h = 4*com_z + 1.2*up_z - 2.0*foot_z` where `com_z` is the
  base `subtree_com` z, `up_z` the base z-axis world-z component, `foot_z`
  the mean z of the two ankle_roll_link positions; true state restored via
  snapshot/restore after every probe; top 8 + 4 uniform random. No learning,
  no network.
- (d) `shuf`: identical to (a) but every parameter tensor of every ensemble
  member independently permuted with fixed seed 777 (deep copy;
  normalization and registered buffers kept as-is); provenance tags recorded
  as `shuf`.

**Harness fixes applied before/during the sweep (documented):**
1. `run_e1.py` originally passed top_tag `'net'` for the `shuf` arm, which
   would have conflated the two arms' winning-path provenance. Changed to
   `'shuf'` (one word; tags are record-only, search behavior unchanged).
2. Performance: `Ensemble.predict` on 64 proposals took ~3.0 s with the
   vendor's `torch.set_num_threads(3)` (OpenMP overhead dominates the tiny
   contact-attention forward) vs ~0.40 s with 1 thread (measured 2026-11-01;
   0.098 s in the sweep workers). At 14 beam members x 3.0 s = ~49 s/level
   of screening overhead, net/shuf-arm runs would have needed ~60+ min and
   hit the 45-min TIMEOUT for non-scientific reasons. Fix:
   `torch.set_num_threads(1)` in `run_e1.py` (top-level after imports,
   re-asserted in `_init_worker` for forked Pool workers). Wall-clock only:
   scores, the top-8+4-random rule, and matched candidate pools are
   unchanged; all arms affected equally. No vendor files were modified.

**Commands.**
- Smoke: `<venv>/python run_e1.py --smoke` (in-process `rand/prone/1000`;
  PASSED: success=true, 2688/4000 evals, handover level 17, 772.8 s wall)
- Sweep: `<venv>/python run_e1.py --workers 1` (default is 4; see caveats)
- Analysis: `<venv>/python analyze_e1.py` → `summary.json`
- venv: `/home/hatch/workspace/research/body-schema-gait/.venv-exp/bin/python`
- workdir: `/home/hatch/workspace/research/body-schema-gait/muse/exp_e1_proposal_control/`

**Execution caveats.** The VM (4 CPU, 8 GB RAM) was heavily contended:
(a) a first sweep attempt with `--workers 4` made zero progress in 15 min
under memory pressure (OOM killer also took two in-process smoke tests) and
was killed; (b) the hypervisor ballooned 3.5-5.5 GB of the VM's RAM
(`/proc/meminfo` `Balloon:`), leaving ~0.4-2.1 GB available — a 2-worker
relaunch was SIGKILLed (137), so the sweep ran with `--workers 1`;
(c) the VM restarted twice mid-sweep (~19:33 and ~21:33 UTC); all completed
run JSONs survived on disk and the sweep resumed via the `done()` skip
(71, then 38 pending) — no run was ever repeated. Per-run wall was 138-773 s
(mean 210 s); the 45-min `--timeout 2700` was never triggered (zero TIMEOUTs,
zero ERRORs; 80/80 `ok`). Total run wall time: 4.67 h. A cosmetic logging
quirk: the parent's `[N]` progress counter undercounts when a worker's JSON
lands before the parent collects its future (`if done(t): continue` skips
the print); run counts in this report are from `runs/*.json`, not the log.

## Results

All 80 runs completed `ok`. Budget respected in every run (max 3864/4000 evals).

### Per-arm success rates (n=20/arm)

| arm | succ | rate | 95% Wilson CI | median evals-to-handover (among successes) |
|-----|------|------|---------------|---------------------------------------------|
| (a) network     | 6/20 | 0.30 | [0.15, 0.52] | 2940 |
| (b) random      | 4/20 | 0.20 | [0.08, 0.42] | 3360 |
| (c) heuristic   | 7/20 | 0.35 | [0.18, 0.57] | 1848 |
| (d) shuffled-net| 2/20 | 0.10 | [0.03, 0.30] | 2520 |

### Per-pose breakdown

| arm | prone | median | supine | median |
|-----|-------|--------|--------|--------|
| (a) network      | 5/10 | 3024 | 1/10 | 2856 |
| (b) random       | 2/10 | 2940 | 2/10 | 3612 |
| (c) heuristic    | 6/10 | 1848 | 1/10 | 1512 |
| (d) shuffled-net | 2/10 | 2520 | 0/10 | —    |

Supine is much harder than prone for every arm. KM median evals-to-handover
is undefined (None) for all arms — fewer than half the runs had handover
events, so the survival curve never crosses 0.5.

### Log-rank on evals-to-handover (event = handover+balance; H0: identical curves)

| pair | chi2 | p | O1-E1 (positive favours first) |
|------|------|---|--------------------------------|
| net vs rand | 0.642 | 0.4230 | +1.26 |
| net vs heur | 0.284 | 0.5938 | -0.95 |
| net vs shuf | 2.348 | 0.1254 | +2.16 |
| rand vs heur| 1.645 | 0.1997 | -2.09 |
| rand vs shuf| 0.651 | 0.4198 | +0.99 |
| heur vs shuf| 3.884 | **0.0487** | +2.89 |

### Fisher exact on success counts (two-sided)

| pair | p |
|------|---|
| net vs rand | 0.7164 |
| net vs heur | 1.0000 |
| net vs shuf | 0.2351 |
| rand vs heur| 0.4801 |
| rand vs shuf| 0.6614 |
| heur vs shuf| 0.1274 |

All numbers above were independently recomputed from `runs/*.json` with
separate code (not just `analyze_e1.py` output): success counts, Fisher
p-values, and log-rank statistics all match exactly.

### Winning-path provenance (tag counts over successful runs)

| arm | paths | tag counts |
|-----|-------|------------|
| (a) network      | 6 | net: 66, rand: 23 |
| (b) random       | 4 | rand: 69 |
| (c) heuristic    | 7 | heur: 62, rand: 11 |
| (d) shuffled-net | 2 | shuf: 17, rand: 11 |

The selectors drive the winning paths (not the 4 exploratory random picks):
74% of net-arm winning actions came from network top-8, 85% of heur-arm from
heuristic top-8. Spot-checked JSONs (`e1_heur_prone_1003`,
`e1_shuf_supine_2005`) confirm spec compliance: budget respected, provenance
logged, passive_wide settled, handover gate + balance() applied.

## Falsification verdict

**FALSIFIED: attention screening adds no search value under this
matched-budget design.**

Pre-registered rule: (a) net must be better than (b) rand AND better than
(c) heur at p<0.05 (log-rank on evals-to-handover + Fisher on success rate).
Observed:
- net vs rand: log-rank p=0.423 (direction +1.26, n.s.), Fisher p=0.716 → NOT better.
- net vs heur: log-rank p=0.594 (direction -0.95, favours heur, n.s.), Fisher p=1.000 → NOT better.

Both conditions fail, so the falsifier triggers under either the strict
("NOT better than b AND NOT better than c") or the lenient reading. The
network arm (30%) is statistically indistinguishable from uniform-random
proposal selection (20%) and from the hand-crafted FK heuristic (35%) when
every arm gets the same 4000-evaluation physics budget.

## Honest caveats and nuances

1. **The heuristic is a strong baseline, not a strawman.** `h = 4*com_z +
   1.2*up_z - 2.0*foot_z` had the highest raw success rate (7/20) and the
   fastest median handover (1848 evals — ~1000 fewer than net). A cheap
   FK proxy for "get the torso up and the feet under it" beats a learned
   contact-attention model at this budget. That is the substantive finding,
   not just a null.
2. **The network is not pure noise.** The shuffled control (2/20, 10%) did
   worse than the intact network (6/20, 30%), and heur significantly beats
   shuf on log-rank (p=0.049). The learned weights carry *some* signal — just
   not enough to beat a good heuristic or random search at n=20/arm.
3. **Power.** n=20/arm with Wilson CIs spanning ~0.35 means only large
   effects are detectable. A small true advantage of the network (e.g. 30%
   vs 20%) cannot be ruled in or out; what is ruled out is a *large,
   practically meaningful* advantage of attention screening at this budget.
4. **Compute.** 4.67 h total run wall on a balloon-constrained VM; per-run
   138-773 s. The 45-min timeout never triggered. Throughput, not science,
   was the binding constraint (single worker due to hypervisor balloon).
5. **Scope.** Wide starts covered yaw ±0.6, roll ±0.30, pitch ±pi/2 by pose,
   joint noise N(0,0.08); only prone/supine; only the pi+ 24-DoF MuJoCo sim
   (no physical robot). Results do not speak to other budgets, other
   proposal distributions, or closed-loop use of the network.
6. **No TIMEOUTs, no ERRORs, no repeated runs.** 80/80 `ok`; every run
   respected the 4000-eval budget (max 3864).
