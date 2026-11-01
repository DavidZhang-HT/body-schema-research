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
level and resumes it automatically; completed results are skipped. Checkpoints
are trusted local pickle files and must not be imported from other sources.
Small paths/results are committed; full traces and movies stay local.

Graph is complete and directed (including descent and both sides); self-mode
refinement is allowed. Each expansion tests all possible outgoing edges through
shared dynamic proposals and classifies the landing mode. Four beam seats retain
different modes. There are no per-edge caps or mode restart boundaries. P-F3 is
a 0.05 score prior only. TRANSITION keeps unclassified dynamic states searchable.
The graph supplies structure and explanations; no efficiency advantage is claimed.

Score v3 uses worst progress in the final 0.2 s, logarithmic terminal KE and
speed penalties, and expires retained parents after two levels. It removes
foot-fraction/foot-load shaping entirely, so hand support has no score tax.
High-energy intermediate motion is allowed. This is a heuristic, not a guarantee
that every settled state outranks every moving state of different geometry.

See COMBO_REPORT.md for evaluated outcomes and explanation semantics.

After each search finishes:

```bash
.venv-exp/bin/python -m combo.explain --pose supine --seed 0
.venv-exp/bin/python -m combo.render --pose supine --seed 0
.venv-exp/bin/python -m combo.summarize
.venv-exp/bin/python -m unittest combo.test_combo
```

`results/*_explanation.json` contains per-block labels and interior certificates;
`RESULTS.md` is generated from actual result files. Video is CPU-rendered with
simultaneous front/side views at real-time playback. Run explanation/rendering
after the simulation queues finish to avoid memory contention from mesh models.
