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
