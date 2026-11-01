"""Post-hoc mechanism diagnostics for E5 (NOT part of the pre-registered design).

D1: PRONE->STAND edge with cap raised 1400->4000, single attempt, network
    screening, same edge_score. Tests whether the 1400-eval cap / restart
    structure is the blocker.
D2: flat E1-style beam (4000 budget, width 14, depth 24, network screening)
    maximizing edge_score = score + 3*mode_term(STAND) instead of score.
    Tests whether the mode_term shaping alone breaks search.

Seed prone/1004: E1's flat+score+network succeeded here in 1176 evals.
"""
import os, sys, time
import numpy as np

WORKDIR = '/home/hatch/workspace/research/body-schema-gait/muse/exp_e5_graph_combination'
E1DIR = '/home/hatch/workspace/research/body-schema-gait/muse/exp_e1_proposal_control'
sys.path.insert(0, WORKDIR); sys.path.insert(0, E1DIR)
sys.path.insert(0, os.path.join(E1DIR, 'vendor'))
import torch
torch.set_num_threads(1)

from ground_schema import GroundWorld, B, BOUNDS
from ground_model import Ensemble
from ground_search import score
from ground_imagine import value
import arms, modes
from run_e1 import passive_wide
import run_e5

arms.configure(B, BOUNDS, value)
net = Ensemble()
run_e5._NETS = {'net': net}  # run_edge's _select reads the module-global

def edge_score(world):
    return score(world) + 3.0 * modes.mode_term('STAND', modes.orientation(world))

# ---------------- D1: edge with 4000 cap ----------------
run_e5.EDGE_EVAL_CAP = 4000
run_e5.EDGE_LEVELS = 24
world = GroundWorld()
passive_wide(world, 'prone', 1004)
entry = world.snapshot()
sel_rng = np.random.default_rng(60_000 + (1004 % 100_000))
t0 = time.time()
er = run_edge = run_e5.run_edge(world, 'g', 'PRONE', 'STAND', entry, sel_rng, 1004, 0, 0.0)
print('D1 edge cap=4000: feasible=%s evals=%s levels=%s wall=%.0fs' % (
    er['feasible'], er['evals'], er['levels'], time.time() - t0), flush=True)

# ---------------- D2: flat beam maximizing edge_score ----------------
run_e5.EDGE_EVAL_CAP = 1400  # restore (not used below anyway)
world2 = GroundWorld()
passive_wide(world2, 'prone', 1004)
initial = world2.snapshot()
sel_rng2 = np.random.default_rng(50_000 + 0 * 100_000 + (1004 % 100_000))  # E1 net sel namespace
beam = [(edge_score(world2), initial, [])]
evals = 0
hit = False
t0 = time.time()
for level in range(24):
    cand, states = [], []
    for b, (_, snap, _) in enumerate(beam):
        world2.restore(snap); states.append(world2.observe())
        cand.append(arms.generate_proposals(np.random.default_rng([1004, level, b])))
    pool = list(beam); exhausted = False
    for b, (_, snap, path) in enumerate(beam):
        chosen = arms.select_network(states[b], cand[b], net, sel_rng2, 'net')
        for (j, tag) in chosen:
            if evals >= 4000: exhausted = True; break
            world2.restore(snap); world2.advance(cand[b][j]); evals += 1
            m2, s2 = modes.classify(world2)
            if m2 == 'STAND' and world2.kinetic() < 0.10:
                hit = True; break
            pool.append((edge_score(world2), world2.snapshot(), path + [(cand[b][j].copy(), tag)]))
        if hit or exhausted: break
    if hit: break
    pool.sort(key=lambda e: e[0], reverse=True)
    beam, cells = [], set()
    for e_ in pool:
        world2.restore(e_[1]); r = world2.metrics(); q = world2.d.qpos[7:]
        cell = (int(r['com_height']/.025), int(r['up_z']/.12), int((q[0]+q[6])/.6), int((q[3]+q[9])/.7))
        if cell not in cells: beam.append(e_); cells.add(cell)
        if len(beam) >= 14: break
    if len(beam) < 14:
        for e_ in pool:
            if all(e_ is not b_ for b_ in beam): beam.append(e_)
            if len(beam) >= 14: break
    if exhausted: break
print('D2 flat+edge_score: STAND_reached=%s evals=%d wall=%.0fs' % (hit, evals, time.time() - t0), flush=True)
