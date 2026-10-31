#!/usr/bin/env python
"""E1: matched-budget proposal control for the pi+ get-up falsification program.

4 arms x 2 poses x 10 seeds = 80 runs. Every run gets an IDENTICAL budget of
4000 physics evaluations (world.advance() calls), beam width 14, depth <= 24,
the same diversity cells, handover criterion and balance() as the reference
hybrid (vendor/ground_hybrid.py). Only the *proposal selection* differs:

  net  - (a) network screening: 64 proposals scored by the contact-attention
         ensemble via value(); top 8 + 4 uniform random get physics rollouts
  rand - (b) 12 uniform-random proposals from the same 64-proposal pool
  heur - (c) FK-only heuristic h = 4*com_z + 1.2*up_z - 2.0*foot_z; top 8 + 4 random
  shuf - (d) same as (a) but every parameter tensor of every ensemble member
         is permuted with fixed seed 777 (normalization kept as-is)

Starts: passive_wide(pose, seed) - wider than the original passive():
  base yaw U(+/-0.6), roll U(+/-0.30), pitch +/-pi/2 by pose,
  joint noise N(0,0.08), base height 0.26; same settle criterion; on failure
  retry with seed+10000 (up to 3 attempts) and log the retries.

Matched-proposal design: proposal candidates are drawn from a dedicated RNG
seeded by (run_seed, level, beam_index) INDEPENDENT of the arm, so at any
given (seed, level, beam state) all arms see the same 64 candidates; only the
selection rule differs. Arm-internal selection randomness uses a separate RNG
seeded by (arm, seed).

Usage:
  python run_e1.py            # full 80-run sweep, Pool(4), streams runs/*.json
  python run_e1.py --arms net,rand
  python run_e1.py --smoke    # single in-process validation run (rand/prone/1000)
"""
import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing as mp

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WORKDIR, 'vendor'))
RUNS = os.path.join(WORKDIR, 'runs')
os.makedirs(RUNS, exist_ok=True)

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from ground_schema import GroundWorld, B, BOUNDS
from ground_model import Ensemble
from ground_search import score
from ground_imagine import value
from ground_balance import balance
from experiment import standing
import arms

# PERFORMANCE FIX (not a science change): vendor/ground_model.py sets
# torch.set_num_threads(3) at import, which makes the tiny contact-attention
# forward ~7.5x SLOWER on this CPU (measured 2026-11-01: 2.98s vs 0.40s per
# 64-proposal predict with 3 vs 1 threads; OpenMP overhead dominates the
# small workload). With 3 threads the net/shuf arms would need ~60+ min/run
# and hit the 45-min TIMEOUT for non-scientific reasons. Single-threaded
# inference is used for all E1 runs; the selection mathematics (scores,
# top-8+4-random rule, matched candidate pools) are unchanged and every arm
# is affected equally.
import torch
torch.set_num_threads(1)

arms.configure(B, BOUNDS, value)

BUDGET = 4000
WIDTH = 14
DEPTH = 24
ARMS = ['net', 'rand', 'heur', 'shuf']
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
POSES = ['prone', 'supine']
SEEDS = {'prone': list(range(1000, 1010)), 'supine': list(range(2000, 2010))}
RUN_TIMEOUT = 2700  # 45 min per run; kill, mark TIMEOUT, document
SHUFFLE_SEED = 777

_NETS = {}  # worker-global: {'net': Ensemble, 'shuf': shuffled Ensemble}


def run_path(arm, pose, seed):
    return os.path.join(RUNS, f'e1_{arm}_{pose}_{seed}.json')


def done(task):
    return os.path.exists(run_path(*task))


def passive_wide(world, pose, seed, max_attempts=3):
    """Wider passive settle than vendor passive(): yaw U(+-0.6), roll U(+-0.30),
    pitch +-pi/2 by pose, joint noise N(0,0.08), base height 0.26.
    Same settle criterion. Retries with seed+10000*k on failure."""
    retries = 0
    settled = False
    info = {}
    for att in range(max_attempts):
        s = seed + att * 10000
        rng = np.random.default_rng(s)
        mj.mj_resetData(world.m, world.d)
        world.actual[:] = 0
        world.d.qpos[:] = standing(world.m)
        world.d.qpos[2] = .26
        pitch = (-1 if pose == 'supine' else 1) * np.pi / 2
        world.d.qpos[3:7] = R.from_euler(
            'xyz', [rng.uniform(-.30, .30), pitch, rng.uniform(-.6, .6)]
        ).as_quat()[[3, 0, 1, 2]]
        world.d.qpos[7:] += rng.normal(0, .08, 24)
        mj.mj_forward(world.m, world.d)
        stable = 0
        for k in range(6000):
            world.d.ctrl[:] = 0
            mj.mj_step(world.m, world.d)
            if k % 20 == 0:
                kinetic = world.kinetic()
                stable = (stable + 1 if kinetic < .002
                          and np.linalg.norm(world.d.qvel[:6]) < .03
                          and np.max(abs(world.d.qvel[6:])) < .15 else 0)
                if stable >= 15:
                    break
        settled = stable >= 15
        info = {'attempt_seed': s, 'seconds': float(world.d.time),
                'kinetic_j': float(world.kinetic())}
        if settled:
            break
        retries += 1
    info.update({'pose': pose, 'seed': seed, 'settled': settled,
                 'retries': retries})
    return info


def _select(arm, world, state, candidates, sel_rng):
    if arm == 'net':
        return arms.select_network(state, candidates, _NETS['net'], sel_rng, 'net')
    if arm == 'shuf':
        return arms.select_network(state, candidates, _NETS['shuf'], sel_rng, 'shuf')
    if arm == 'heur':
        return arms.select_heuristic(world, candidates, sel_rng)
    if arm == 'rand':
        return arms.select_random(candidates, sel_rng)
    raise ValueError(arm)


def run_one(task):
    """One full beam-search run. Writes its own JSON (crash recovery)."""
    arm, pose, seed = task
    t0 = time.time()
    result = {'arm': arm, 'pose': pose, 'seed': seed, 'status': 'ok',
              'budget': BUDGET, 'width': WIDTH, 'depth': DEPTH,
              'shuffle_seed': SHUFFLE_SEED if arm == 'shuf' else None}
    try:
        world = GroundWorld()
        pw = passive_wide(world, pose, seed)
        result['passive'] = pw
        initial = world.snapshot()
        result['initial_metrics'] = {k: float(v) for k, v in world.metrics().items()}
        sel_rng = np.random.default_rng(50_000 + ARM_INDEX[arm] * 100_000 + (seed % 100_000))

        beam = [(score(world), initial, [])]
        evals = 0
        success = False
        evals_to_handover = BUDGET
        handover_level = None
        history = []
        for level in range(DEPTH):
            # Matched proposals: dedicated RNG per (seed, level, beam idx),
            # independent of arm -> identical candidate pools across arms.
            cand, states = [], []
            for b, (_, snap, _) in enumerate(beam):
                world.restore(snap)
                states.append(world.observe())
                cand.append(arms.generate_proposals(np.random.default_rng([seed, level, b])))
            pool = list(beam)
            exhausted = False
            for b, (_, snap, path) in enumerate(beam):
                if arm == 'heur':
                    world.restore(snap)  # heuristic probes from this beam state
                chosen = _select(arm, world, states[b], cand[b], sel_rng)
                for (j, tag) in chosen:
                    if evals >= BUDGET:
                        exhausted = True
                        break
                    action = cand[b][j]
                    world.restore(snap)
                    world.advance(action)  # ONE advance() = ONE physics evaluation
                    evals += 1
                    pool.append((score(world), world.snapshot(),
                                 path + [(action.copy(), tag)]))
                if exhausted:
                    break
            # Diversity selection: identical cells to vendor/ground_hybrid.py
            pool.sort(key=lambda e: e[0], reverse=True)
            beam, cells = [], set()
            for entry in pool:
                world.restore(entry[1])
                r = world.metrics()
                q = world.d.qpos[7:]
                cell = (int(r['com_height'] / .025), int(r['up_z'] / .12),
                        int((q[0] + q[6]) / .6), int((q[3] + q[9]) / .7))
                if cell not in cells:
                    beam.append(entry)
                    cells.add(cell)
                if len(beam) >= WIDTH:
                    break
            if len(beam) < WIDTH:
                for entry in pool:
                    if all(entry is not b_ for b_ in beam):
                        beam.append(entry)
                    if len(beam) >= WIDTH:
                        break
            world.restore(beam[0][1])
            r = world.metrics()
            n_top = sum(1 for _, t in beam[0][2] if t != 'rand')
            history.append({'level': level + 1, 'physics_evaluations': evals,
                            'winning_path_len': len(beam[0][2]),
                            'winning_path_nontop': n_top,
                            **{k: float(v) for k, v in r.items()}})
            if arms.handover_ok(r):
                try:
                    ok, _, _ = balance(world)
                except Exception:
                    result['balance_error'] = traceback.format_exc(limit=3)
                    ok = False
                if ok:
                    success = True
                    evals_to_handover = evals
                    handover_level = level + 1
                    break
            if exhausted:
                break

        path = beam[0][2]
        tags = [t for _, t in path]
        result.update({
            'success': bool(success),
            'physics_evaluations': evals,
            'evals_to_handover': int(evals_to_handover),
            'handover_level': handover_level,
            'final_metrics': {k: float(v) for k, v in world.metrics().items()},
            'provenance': {
                'tags': tags,
                'tag_counts': {t: int(tags.count(t)) for t in sorted(set(tags))},
                'actions': [a.tolist() for a, _ in path],
            },
            'history': history,
            'wall_seconds': time.time() - t0,
        })
    except Exception:
        result['status'] = 'error'
        result['error'] = traceback.format_exc(limit=5)
        result['wall_seconds'] = time.time() - t0

    tmp = run_path(arm, pose, seed) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f)
    os.replace(tmp, run_path(arm, pose, seed))
    return {'task': task, 'status': result['status'],
            'success': result.get('success'),
            'evals': result.get('physics_evaluations'),
            'wall': round(result.get('wall_seconds', 0), 1)}


def _init_worker():
    global _NETS
    torch.set_num_threads(1)  # re-assert: forked Pool workers must not use 3 threads (see note above)
    net = Ensemble()
    _NETS = {'net': net, 'shuf': arms.build_shuffled_ensemble(net, seed=SHUFFLE_SEED)}


def _write_marker(task, status, note=''):
    arm, pose, seed = task
    rec = {'arm': arm, 'pose': pose, 'seed': seed, 'status': status,
           'success': False, 'physics_evaluations': None,
           'evals_to_handover': BUDGET, 'note': note,
           'wall_seconds': None}
    tmp = run_path(*task) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, run_path(*task))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=int, default=RUN_TIMEOUT)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()

    if a.smoke:
        _init_worker()
        print('SMOKE: single in-process run (rand/prone/1000)', flush=True)
        print(json.dumps(run_one(('rand', 'prone', 1000)), indent=2), flush=True)
        return

    want = [x.strip() for x in a.arms.split(',') if x.strip()]
    tasks = [(arm, pose, seed) for arm in want for pose in POSES for seed in SEEDS[pose]]
    tasks = [t for t in tasks if not done(t)]
    print(f'E1: {len(tasks)} runs pending ({a.workers} workers, '
          f'timeout {a.timeout}s/run)', flush=True)

    remaining = list(tasks)
    n_done = 0
    while remaining:
        batch = [t for t in remaining if not done(t)][:a.workers]
        if not batch:
            break
        remaining = [t for t in remaining if t not in batch]
        pool = mp.get_context('fork').Pool(a.workers, initializer=_init_worker)
        futs = [(t, pool.apply_async(run_one, (t,))) for t in batch]
        timed_out = False
        for t, fut in futs:
            if done(t):
                continue
            try:
                r = fut.get(timeout=a.timeout)
                n_done += 1
                print(f'[{n_done}] {t} -> {r["status"]} '
                      f'success={r["success"]} evals={r["evals"]} '
                      f'wall={r["wall"]}s', flush=True)
            except mp.TimeoutError:
                print(f'TIMEOUT {t} after {a.timeout}s - killing worker, '
                      f'marking TIMEOUT', flush=True)
                _write_marker(t, 'timeout',
                              f'exceeded {a.timeout}s wall; worker killed')
                timed_out = True
                break
            except Exception:
                print(f'ERROR collecting {t}: {traceback.format_exc(limit=3)}',
                      flush=True)
                _write_marker(t, 'error', 'parent-side collection error')
        if timed_out:
            pool.terminate()
        else:
            pool.close()
        pool.join()
        # Re-queue anything in this batch that has no JSON yet (e.g. killed
        # by a sibling's timeout); completed runs wrote their own files.
        for t in batch:
            if not done(t) and t not in remaining:
                remaining.append(t)
    print('E1 sweep complete.', flush=True)


if __name__ == '__main__':
    main()
