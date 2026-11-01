# Combo checkpoint — COMPLETE

2026-11-01, branch `combo-denetworked`, starting baseline cb17ccd.
All requested reports/code were read before implementation. No upstream code
or plant was changed. Pre-existing untracked root assets/ was left alone.

## Delivered

- Official 24-DoF pi+ physics search, 16 synergies, bidirectional contact-mode
  graph, no learned model/attention, robust score v3 plus dynamic exploration.
- Unchanged B QP handover; E3 statics LP explanations with support polygons,
  loads, torque ratios, per-block/per-mode labels and saved edge witnesses.
- Default `--selection dynamic`: SUPINE 3/3, PRONE 1/3 (prone seed 1).
  Initial `--selection robust`: 0/6; all failures retained.
- All 12 saved action paths replay without resets and with exact 0.0 full-state
  error at pre-QP endpoints. All search/QP success/failure outcomes reproduce.
  All 252 saved graph-edge witnesses replay exactly as well.
- Six regression checks pass, including terminal checkpoint recovery and cached
  configuration mismatch rejection. Upstream hashes match environment.json.
- Full successful video: results/dynamic/supine_0_getup.mp4, 492448 bytes,
  H.264 1024x384, 117 frames / 12 fps / 9.75 s, labeled 2x playback. Start/end
  frames visually inspected. Small movie and final PNG committed; large/full
  trace archives, checkpoints and incomplete media ignored.
- COMBO_REPORT.md complete; RESULTS_dynamic.md / RESULTS.md detailed tables;
  results/validation.json audit; results/dynamic/video_metadata.json metadata.

## Important limits (in report)

DYNAMIC_UNRESOLVED is deliberately added rather than falsely claiming momentum
necessity from high KE alone. STATICS_GAP is sampled/path-specific. E3 cert_ok
logs but does not gate joint-limit margin (supine_0 min -0.02852 rad). Supine_1
also transiently violates E3 floor/self thresholds despite simulator success.
No hard-limit compliance, general robustness, real-time operation, hardware or
sim-to-real claim. Prone failures include QP capture and remaining hand support.

## Reproduction / resume

No simulations or artifact jobs remain running or pending. Do not redo completed
searches. `python -m combo.sweep` defaults to dynamic; matching completed results
are skipped, local per-level checkpoints resume. `--selection robust` selects
the preserved initial configuration. `python -m combo.artifacts` skips existing
JSON+trace artifacts, uses per-movie locks and atomic final video rename.
CPU render uses `MUJOCO_GL=glfw xvfb-run -a ...` (EGL absent); Mesa software,
no shadows/reflections, system FFmpeg. The initial apt attempts made no changes.
The early VM restart affected only initial robust seed-0 partial-level accounting,
explicitly disclosed in the report. Do not push main; no push was performed.
