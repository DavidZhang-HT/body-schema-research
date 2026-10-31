# Body-Schema Self-Exploration for Bipedal Gait

Research codebase (academic, non-commercial): turning a **pure kinematic reference** (human walking motion) into a **physically feasible biped gait** via *predict → calibrate → control*, as an explicit-model alternative to brute-force RL motion imitation (DeepMimic/AMP).

Target platform: pi+ humanoid (24-DoF model in `assets/`), MuJoCo CPU simulation, deterministic seeds.

## Pipeline

- **Phase 1 — Kinematic retargeting**: human walking (LAFAN1 BVH, `data/`) → robot joint space via URDF geometry (`sim/phase1_retarget.py`); nominal joint + CoM trajectories.
- **Phase 2 — Safe active postural exploration**: multi-frequency sway / squat / single-foot unloading in double support; identifies task-relevant dynamic deviations (trunk mass + CoM offset, actuation delay, friction) — `sim/experiment.py`, `sim/scenarios.py`.
- **Phase 2b — Joint-zero active identification**: loop-closure estimator (`sim/zero_id.py`) + full-joint-torque residual channel (`sim/zero_torque*.py`); observable subspace to 0.17–0.38°, triple-gated updates.
- **Phase 3 — Dynamic feasibility projection**: constrained trajectory optimization / WBC (`sim/wbc.py`).
- **Phase 4 — Surprise-driven closed-loop update**: fast feedback / state-contact estimation / slow gated model adaptation.
- **Get-up thread**: gap analysis (`analysis-getup.md`), attention-based explicit MPC F3 test (`sim/getup_*.py`), autonomous interpretable contact-mode strategy search (`results/getup_search/`).

## Honest headline results

- 3-way comparison (fixed / passive ID / active ID) on the mismatched plant: **9/9 runs fell**; ablation traced it to a delay-compensation usage bug (removing it lets 2/3 seeds reach the forward phase).
- Zero-ID: partial — observable subspace identified, intra-leg common-mode allocation is a near-nullspace for kinematic residuals; the torque channel makes it observable (validated).
- F3 get-up test: **NOT_FOUND** — no quasi-static supine→stand path via direct trunk lift (not torque-bound: peak ratio 0.181); autonomous strategy search verdict: **QUASI_STATIC_EXHAUSTION**.

See `plan.md`, `results.md`, `report-cn.md` for the full record.

## Run

```bash
cd sim
python -m venv .venv && .venv/bin/pip install -r requirements.txt  # requirements TBD from imports: mujoco, numpy, scipy
.venv/bin/python experiment.py fixed 0        # 3-way comparison episode
.venv/bin/python render_videos.py fixed 0     # side-view stick-figure video (matplotlib, no GL)
```

Videos: `videos/gait_fixed_s0.mp4`, `videos/gait_active_nodc_s0.mp4`.
