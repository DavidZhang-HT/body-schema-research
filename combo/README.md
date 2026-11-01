# De-networked combination

CPU/headless MuJoCo on B's official 24-DoF mesh plant. No learned model,
attention screen, or scripted get-up. B's bounded 0.6-second synergy PD and
unchanged QP balance controller; E3's A-style statics LP; explicit contact modes.

Run from repository root, with existing meshes and `.venv-exp`:

```bash
OPENBLAS_NUM_THREADS=1 .venv-exp/bin/python -m combo.search --pose supine --seed 0
```

Use at most two concurrent runs. Default budget: 6000 action rollouts, plus
separately logged QP attempts. Search saves an atomic local checkpoint each
level and resumes it automatically; matching completed results are skipped; configuration mismatches are rejected. Checkpoints
are trusted local pickle files and must not be imported from other sources.
Small paths/results and the ~0.5 MB demonstration movie are committed; full
traces and partial renders stay local.

Graph is complete and directed (including descent and both sides); self-mode
refinement is allowed. Each expansion samples outgoing edges through
shared dynamic proposals and classifies their landing modes. Six exploration seats retain dynamic candidates, two seats represent modes,
and the remaining seats follow robust score v3. There are no per-edge caps or mode restart boundaries. P-F3 is
a 0.05 score prior only. TRANSITION keeps unclassified dynamic states searchable.
The graph supplies structure and explanations; no efficiency advantage is claimed.

Score v3 uses worst progress in the final 0.2 s, logarithmic terminal KE and
speed penalties, and expires retained parents after two levels. The robust rank has no contact-load reward. Exploration seats retain B's
bounded absolute foot-load bonus, never a foot fraction; adding hand support at
fixed foot force has no score tax.
High-energy intermediate motion is allowed. This is a heuristic, not a guarantee
that every settled state outranks every moving state of different geometry.

See COMBO_REPORT.md for evaluated outcomes and explanation semantics.

After each search finishes:

```bash
.venv-exp/bin/python -m combo.explain --pose supine --seed 0
MUJOCO_GL=glfw xvfb-run -a .venv-exp/bin/python -m combo.render --pose supine --seed 0
.venv-exp/bin/python -m combo.summarize
.venv-exp/bin/python -m unittest combo.test_combo
```

`results/dynamic/*_explanation.json` contains per-block labels and interior certificates;
`RESULTS_dynamic.md` is generated from the final configuration; `RESULTS.md`
preserves the initial failed configuration. Video is CPU-rendered with
simultaneous front/side views at labeled 2x playback (12 fps). Run explanation/rendering
after the simulation queues finish to avoid memory contention from mesh models.

Rendering uses system FFmpeg, Pillow, and CPU Mesa through Xvfb/GLX in this VM
(EGL is absent). No imageio dependency. To run the default dynamic-exploration
revision, use `python -m combo.sweep --selection dynamic`; pass the same
`--selection dynamic` to explain/render/summarize. Results live separately.

The default is now `--selection dynamic`, which found 3/3 supine and 1/3 prone
search/QP successes in the six-seed build evaluation. The initial unsuccessful
variant remains reproducible with `--selection robust`. See COMBO_REPORT.md
and RESULTS_dynamic.md for independent replay and certificate results.
