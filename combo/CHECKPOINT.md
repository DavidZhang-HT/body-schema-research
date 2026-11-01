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
