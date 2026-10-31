"""Reduced re-run of step 2b variant L3 only (trunk touching lift-off), w_post 0.3, both seeds.
(The full step-2b driver was stopped after L1, max-margin and 2 of 4 L2 solves because of the wall-clock budget.)"""
import json, numpy as np
from e3_step2b_liftoff import KeepIK, liftoff, OUT
from e3_plant import GroundWorld
from e3_cert import BSchema
import experiment
w = GroundWorld(); S = BSchema(w.m); q0 = np.load(OUT / 'plateau_snapshot.npy')[:31]; ik = KeepIK(S, q0)
qs = experiment.standing(w.m)
seeds = [('plateau', np.zeros(30)), ('stand_joints', np.r_[np.zeros(6), qs[7:] - q0[7:]])]
log = []
best = liftoff(S, ik, q0, 'L3_trunk_touching', 'trunk', None, 1.0, log, w_posts=(0.3,), seeds=seeds)
(OUT / 'step2b_L3_log.json').write_text(json.dumps(log, indent=1))
print('L3 certified lift-off found:', best is not None)
