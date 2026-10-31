# E3 pre-registration (written 2026-11-01 ~01:54 +08, BEFORE running steps 2 and 3)

Written after step 1 (plateau certification) was run, before any feet-only search or augmented search.

## Step-1 facts already known when this was written
- Plateau snapshot (hybrid_supine_101 'final'): support = L_foot + R_foot + R_hand (tripod); all-contacts
  certificate OK (rho 0.160 at mu 0.6). Feet-only LP INFEASIBLE (CoM x = 0.234 m is behind both feet,
  x = 0.41-0.51 m). KE 0.126 J: the snapshot is not at rest.
- Holding the plateau with B's own zero action (PD hold) falls back to supine within 3 blocks (1.8 s).

## Definitions fixed now
- **Certified feet-only equilibrium**: A's certify on B's plant restricted to foot contacts
  (`*_ankle_roll_link` points within 5 mm), statics LP feasible, rho <= 0.8, pen >= -1.5 mm,
  selfd >= -1 mm, at mu = 0.6 (A's planning mu). mu = 0.8 reported as secondary. All non-foot shell
  points >= 1 cm above the floor (A's keyframe 'forbid' convention).
- **Within reach (primary, tier A)**: both feet stay where they are: each foot's sole centre within
  6 cm (about half a foot length) of that foot's plateau ground contact point (the foot may roll/flatten
  about its contact, no stepping), joints within B's limits minus 0.02 rad (B's qlo/qhi).
  Secondary (tier B): feet may be re-placed within 20 cm (requires a step; reported, not used for the verdict).
- **Quasi-static connectivity (supporting evidence)**: a "lift-off" configuration exists in which
  the plateau's non-foot support (R_hand) is still touching the floor at its plateau xy (within 3 cm) and
  the feet-only certificate already holds; and the joint/base interpolation chain from the plateau to it
  (20 keyframes, feet and hand kept down by IK repair) is certified at every keyframe with all contacts.
- **Step 3 success**: B's handover criterion (ground_hybrid.py): base_height > 0.16, up_z > 0.9,
  foot_load > 75 N, nonfoot_load < 8 N, AND it survives a 1.2 s zero-action hold with the same
  criterion (B's QP `balance()` is not reproduced: it needs osqp/experiment.Controller; the hold is our
  stand-in, and this is documented). B's stricter search stop (base > 0.30, up > 0.92, foot > 75,
  nonfoot < 8, com speed < 0.2) is reported too.
- Step 3 arms, all starting from the plateau snapshot with the UNMODIFIED score():
  (1) original 12-synergy basis, physics beam search (ground_search.search protocol: width 14,
      16 branches, half symmetric, branch 0 = zero), 3 seeds;
  (2) the same with the basis augmented by differential hip roll, differential ankle roll and asymmetric
      knee (15 synergies), 3 seeds, equal evaluation budget;
  The learned-network screening of the hybrid is NOT used (the network takes 12-dim actions only);
  this is the physics-only oracle variant, which is B's stronger planner.
- Verdict rule = the report's falsifier, applied literally:
  (a) tier-A certified feet-only equilibrium exists AND (augmented basis OR score) succeeds -> SEARCH/ACTION-BASIS.
  (b) no tier-A feet-only equilibrium -> STATICS GAP (support-polygon jump).
  If the equilibrium exists but no searched variant succeeds, the outcome is NEITHER (a) nor (b): the
  statics-gap story is falsified for this plateau, but the search/basis claim is not established.
