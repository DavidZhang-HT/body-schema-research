# Combo checkpoint

2026-11-01: branch combo-denetworked, baseline cb17ccd. Required comparison
sections, E1–E5 reports, B search/schema/balance, A modes, E3 plant/cert read.
24-DoF GroundWorld instantiates in .venv-exp. Existing untracked assets/ is
user/environment content: leave alone. No upstream code changes planned.

Building shared-budget mode-diverse beam, score v3 with temporal/energy
penalties and expiring parents, E3 certifier imported directly, unchanged QP.
No network imports. Use two simulation workers maximum, single BLAS thread.
Next: smoke/test, six seeds (0,1,2 each pose), replay/explain/render, report.

Restart recovery: both seed-0 runs resumed (supine level 2 / 240 evals;
prone level 3 / 464 evals). explain.py and test_combo.py survived as untracked
files. Four unittest checks PASSED. Two shell queues now run seeds 0,1,2 for
supine and prone respectively, 6000 eval budget each. Logs live in
combo/results/{pose}_{seed}.log (not /tmp). Each level writes atomic checkpoint.
Explanation semantics include DYNAMIC_UNRESOLVED: high KE alone is not proof of
momentum necessity. Need replay/explain all six completed paths, render successful
supine path, summarize outcomes. No new successes asserted yet.

02:45 UTC: first completed runs supine_0/prone_0 both FAILED, 6000 evals,
zero QP entries, low hand-supported endpoints. Original queues still doing
seeds 1 then 2. Added OPTIONAL --selection dynamic, separate results/dynamic/
campaign (NOT started). It reserves six exploratory seats ranked by instantaneous
progress with weak KE cost and height-gated foot-load bonus (zero below 20 cm),
two mode seats, remainder robust score. Score v3 itself unchanged. Need compare
this build refinement if baseline sweep fails; preserve all attempted results.

Before any dynamic runs: exploration priority changed to original B absolute
foot-load bonus (not foot fraction; adding hand load at fixed feet load has no
cost), including B foot-height and total-load corrections. Primary v3 unchanged.
The earlier gated variant was never run. Robust baseline behavior unchanged.

A lightweight launcher is waiting for all six robust JSONs, then runs six
`--selection dynamic` jobs (same seeds/budget) with two child processes:
`.venv-exp/bin/python -m combo.sweep --selection dynamic --after-robust`
Log combo/results/dynamic_sweep.log. On restart check processes before relaunch;
resume robust first via `python -m combo.sweep --selection robust`, and then
dynamic. Completed JSONs are skipped, in-progress jobs resume per-level pickle.
Do not explain/render while two searches run (mesh memory). After both sweeps,
run explain for all attempts sequentially, render a verified successful supine
trajectory, generate summary and finish report. Original first sweep is not lost.

Rendering runtime verified: EGL absent; initial apt attempt made no changes
(bootstrap apt update owns lock). Existing Xvfb + GLX software renderer passes
small MuJoCo render test. Use `MUJOCO_GL=glfw xvfb-run -a ... -m combo.render`.
Renderer now pipes raw frames to system ffmpeg (imageio absent), Pillow present.
Environment package versions and unchanged upstream source hashes recorded.

02:56 UTC: robust sweep 4/6 finished, all failures with zero QP entries.
Seed-2 searches running (~3500/6000 each); dynamic launcher still waiting.
Latest code saves one replayable witness per distinct discovered graph edge
for dynamic campaign; offline explanations certify these plus every returned
block endpoint/interior and feet-only endpoint. Mode-change tables implemented.
Five regression checks total (four full earlier, three lightweight after edits).
No successful trajectory yet. No upstream code modified.

03:00 UTC: robust sweep COMPLETE, 0/3 supine + 0/3 prone, all 6000 evals,
zero QP handover entries across all six. results/{pose}_{seed}.json and paths
are preserved; RESULTS.md generated (explanations still pending).
Dynamic sweep started automatically, seed 0 both poses, code frozen at 333e6ad
for search semantics. Six exploratory slots use original B absolute-load
priority, two mode seats and remaining robust-score slots; no network.
Next after dynamic sweep: explain BOTH campaigns sequentially, render verified
supine success, regenerate tables, complete report and final tests/commit.

03:06 UTC IMPORTANT: dynamic SUPINE seed 0 SEARCH/QP SUCCESS at 5607 evals,
5 handover attempts. Saved results/dynamic/supine_0.json, _path.npz, _edges.npz.
Independent continuous replay/explanation/render still pending (do not claim
replay verified yet). Dynamic prone_0 near budget, many geometric STAND hits
but failed QP attempts so far. Queues continue remaining seeds automatically.
This is fresh physics search, no historical action path used.

03:11 UTC: artifacts launcher waiting for both sweeps to finish:
`python -m combo.artifacts --wait` -> combo/results/artifacts.log.
Then sequentially replays/certifies all 12 results (skips existing JSON+trace),
generates both RESULTS tables, renders first verified supine success via Xvfb,
and records ffprobe metadata/SHA256. On restart relaunch only if not running.
Pending final work after artifacts: inspect movie/frame, inspect explanation
labels and failures, finish COMBO_REPORT/README, run full tests, commit.

03:13 UTC: dynamic SUPINE seed 1 also SEARCH/QP SUCCESS, 4235 evaluations,
53 QP attempts. Both supine successes await independent replay. Seed 2 now
running. Prone seed 1 still searching, seed 0 failed at 6000 (19 QP attempts).
Operational-only fix committed 039eb79: terminal success/winner retained in
checkpoint, NPZ writes atomic; new mocked terminal-resume regression PASSED.
Search rankings/proposals unchanged. Artifacts launcher still waiting.

03:17 UTC: VM available RAM increased to ~2.9 GB. Started ONE low-priority
independent replay/certificate process for dynamic supine_0 alongside the two
search workers (three CPU tasks, within user's <=4 cap): `nice -n 10 python -m
combo.explain --pose supine --seed 0 --selection dynamic`, log
results/dynamic/supine_0_explain.log. Artifacts launcher later skips completed
JSON+trace. Do not launch another analyzer until this one finishes.

03:19 UTC: dynamic SUPINE seed 2 SUCCESS at 3542 evals / 6 QP attempts;
PRONE seed 1 SUCCESS at 4986 / 77 QP attempts. Only prone_2 still searching.
First supine_0 replay succeeded with EXACT full-snapshot endpoint match (0.0).
Improved explanation reader to separate MjData sharing model, so certificate
sampling never resets executing trajectory. Re-running supine_0 once to validate
this stricter uninterrupted replay. Old trace was exact too; this is validation
hardening, not search change. Three supine successes / three completed, prone
one success / two completed so far. Need finish remaining replay/render/report.

03:22 UTC: strict uninterrupted supine_0 replay PASSED with qpos/qvel/full
snapshot errors all exactly 0.0, sustained stand true. Explanation: 5 block
STATICS_GAP flags, 20 DYNAMIC_UNRESOLVED; all 25 block endpoints certified;
7 interior LP failures; sampled geometry stays within E3 penetration/self bounds.
Final rho .08997, feet loads .5258/.4742 mg, COM margin .07176 m. Terminal
sampled .98 s tail is QS (peak KE 1.44e-5 J).
Manual CPU renderer restarted WITHOUT shadow/reflection after default shadows
proved ~3 s/frame; partial file is *.partial.mp4 (ignored). New render writes
*.tmp.mp4 then atomic final rename, logs every 50 frames. No physics rerun.
Still only prone_2 search running; artifacts supervisor will start remaining
sequential explanations after it finishes, skipping already completed supine_0.
