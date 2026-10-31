# Get-up strategy search verdict: **QUASI_STATIC_EXHAUSTION**

Expanded (in order): SUPINE; wall clock 124.3 s.

Reachable modes: ['SUPINE']; never reached: ['SIDE', 'PRONE', 'SIT', 'QUAD', 'HF', 'SQUAT', 'STAND']

## Frontier transitions (every one diagnosed)

| edge | status | rule / failing sub-goal | diagnosis |
|---|---|---|---|
| SUPINE -> SIDE | INFEASIBLE | G2_orient STALLED | structural: no single constraint-family relaxation unlocks progress (the sub-goal has no quasi-static descent direction from this configuration; not torque-/limit-bound); residual {'trunk_left(+y)': 59.9895, 'upper_trunk_land': 0.0001, 'com_over_pivot': 0.0069}; unmet ['orientation fails [left_z >= 0.7 (body +y up, i.e. right side down)] (chest_z +0.98 left_z +0.16 head_z +0.10)'] |
| SUPINE -> PRONE | PRUNED | P-TOPO | orientation LYING_UP -> LYING_DOWN not adjacent: rotation must pass through an intermediate class (not elementary) |
| SUPINE -> SIT | PRUNED | P-F3 | F3 run_a (results/f3/run_a.json): direct supine trunk lift STALLED at P1_head_up, tick 121; trunk_upright cost stuck at 2.84 (tilt 11-17 deg); peak rho 0.181 << 0.8 (NOT torque-bound); relaxing self-collision / joint limits / trust region / rho_cap->inf / mu x2 all failed to unstick (getup_f3_diag.py) -> binding = trunk-flexion work unavailable quasi-statically while the back is the support (structural, not a tuning issue) |
| SUPINE -> QUAD | PRUNED | P-TOPO | orientation LYING_UP -> PITCHED not adjacent: rotation must pass through an intermediate class (not elementary) |
| SUPINE -> HF | PRUNED | P-TOPO | orientation LYING_UP -> PITCHED not adjacent: rotation must pass through an intermediate class (not elementary) |
| SUPINE -> SQUAT | PRUNED | P-F3 | F3 run_a (results/f3/run_a.json): direct supine trunk lift STALLED at P1_head_up, tick 121; trunk_upright cost stuck at 2.84 (tilt 11-17 deg); peak rho 0.181 << 0.8 (NOT torque-bound); relaxing self-collision / joint limits / trust region / rho_cap->inf / mu x2 all failed to unstick (getup_f3_diag.py) -> binding = trunk-flexion work unavailable quasi-statically while the back is the support (structural, not a tuning issue) |
| SUPINE -> STAND | PRUNED | P-F3 | F3 run_a (results/f3/run_a.json): direct supine trunk lift STALLED at P1_head_up, tick 121; trunk_upright cost stuck at 2.84 (tilt 11-17 deg); peak rho 0.181 << 0.8 (NOT torque-bound); relaxing self-collision / joint limits / trust region / rho_cap->inf / mu x2 all failed to unstick (getup_f3_diag.py) -> binding = trunk-flexion work unavailable quasi-statically while the back is the support (structural, not a tuning issue) |

## All evaluated (non-pruned) edges

| edge | status | cost | ticks | reason |
|---|---|---|---|---|
| SUPINE -> SIDE | INFEASIBLE | 4.0847 | 344 | G2_orient STALLED: structural: no single constraint-family relaxation unlocks progress (the sub-goal has no quasi-static descent direction from this configuration; not torque-/limit-bound) |
