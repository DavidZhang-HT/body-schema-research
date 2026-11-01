#!/usr/bin/env python
"""E5: graph-structured get-up search with B's network as edge-proposal oracle.

Builds on the E1 harness (same plant, same 4000-eval budget, same beam
machinery, same initial conditions). Mode predicates (modes.py) are
lightweight approximations on the 24-DoF mesh plant.

Arms (2 poses x 10 seeds = 20 runs each):
  g  - graph + network: mode-structured search; each DAG edge gets a
       beam search whose 64 proposals are screened by E1's network
       (top-8 + 4 random via arms.select_network)
  gr - graph + random: identical graph structure, but 12 uniform-random
       proposals per beam state (arms.select_random) -- separates the
       "graph" effect from the "model" effect
  f  - flat baseline: E1's net-arm runs, REUSED as-is (not re-run)

Graph controller (per run):
  1. passive_wide initial settle (identical to E1, same seeds -> paired
     initial conditions with arm F).
  2. mode = classify(state); while evals < 4000:
       - pick an outgoing DAG edge (SUPINE tries SIDE then PRONE, alternating
         retries; max 2 attempts per directed edge; achieved state is
         re-classified after each edge -- descent allowed)
       - edge search: width-14 beam, <=12 levels, <=1400 evals/edge,
         ranked by edge_score = score(world) + 3.0*mode_term(dst)
       - edge feasible when a physics rollout lands in the destination
         predicate. Lying-mode edges (SIDE/PRONE) additionally require the
         predicate to survive a 150-step passive settle (0.5 eval, counted):
         a "mode" there must be a resting config, not a transient
         pass-through. The STAND edge instead requires kinetic_j < 0.10
         (quasi-static gate): a standing humanoid is NOT passively stable,
         so a passive settle-verify would guaranteed-reject genuine STAND
         arrivals (bug fixed 2026-11-01 after 5/40 runs; those runs were
         discarded and re-run, see report).
       - STAND edge feasible -> STAND mode; then the STAND->STAND refinement
         edge (feasibility = E1 handover criterion) with remaining budget
       - final success = handover_ok (E1) + balance() (vendor), same as E1
  3. evals_to_success = total physics evals (advance()=1, settle-verify=0.5).
     balance() physics is NOT counted (same convention as E1: handover gate).

Matched candidates: G and GR share the candidate RNG namespace
  rng([seed, edge_idx, attempt, level, beam_idx])  (modes.EDGE_INDEX)
so at any (seed, edge, level, beam state) both arms see identical 64
candidates; only the selection rule differs. Arm F is E1's stored runs
(different candidate stream, same initial conditions via same seeds).

Usage:
  python run_e5.py            # full 40-run sweep, Pool(4), streams runs/*.json
  python run_e5.py --arms g,gr
  python run_e5.py --smoke    # single in-process validation run (gr/prone/1000)
"""
import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing as mp

WORKDIR = os.path.dirname(os.path.abspath(__file__))
E1DIR = '/home/hatch/workspace/research/body-schema-gait/muse/exp_e1_proposal_control'
sys.path.insert(0, WORKDIR)
sys.path.insert(0, E1DIR)
sys.path.insert(0, os.path.join(E1DIR, 'vendor'))
RUNS = os.path.join(WORKDIR, 'runs')
os.makedirs(RUNS, exist_ok=True)

import numpy as np
import mujoco as mj

from ground_schema import GroundWorld, B, BOUNDS
from ground_model import Ensemble
from ground_search import score
from ground_imagine import value
from ground_balance import balance
import arms
import modes
from modes import SUPINE, PRONE, SIDE, STAND, TRANSITION
from run_e1 import passive_wide  # identical initial conditions to E1

# Same perf fix as E1: vendor sets torch.set_num_threads(3) at import, which
# makes the tiny contact-attention forward ~7.5x slower on this CPU. Single
# thread for all runs; selection mathematics unchanged.
import torch
torch.set_num_threads(1)

arms.configure(B, BOUNDS, value)

BUDGET = 4000
WIDTH = 14
EDGE_LEVELS = 12
EDGE_EVAL_CAP = 1400
SETTLE_VERIFY_STEPS = 150   # passive steps after a predicate hit (LYING modes only)
SETTLE_VERIFY_COST = SETTLE_VERIFY_STEPS / 300.0  # in advance() units (0.5)
STAND_KINETIC_GATE = 0.10  # quasi-static gate for STAND arrivals (see below)
ARMS = ['g', 'gr']
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
POSES = ['prone', 'supine']
SEEDS = {'prone': list(range(1000, 1010)), 'supine': list(range(2000, 2010))}
RUN_TIMEOUT = 2700  # 45 min per run; kill, mark TIMEOUT, document
MAX_EDGE_ATTEMPTS = 2  # per directed edge per run

_NETS = {}  # worker-global: the E1 ensemble


def run_path(arm, pose, seed):
    return os.path.join(RUNS, f'e5_{arm}_{pose}_{seed}.json')


def done(task):
    return os.path.exists(run_path(*task))


def _select(arm, state, candidates, sel_rng):
    if arm == 'g':
        return arms.select_network(state, candidates, _NETS['net'], sel_rng, 'net')
    if arm == 'gr':
        return arms.select_random(candidates, sel_rng)
    raise ValueError(arm)


def _settle_verify(world):
    """150 passive steps (ctrl=0). Caller re-checks the predicate after."""
    for _ in range(SETTLE_VERIFY_STEPS):
        world.d.ctrl[:] = 0
        mj.mj_step(world.m, world.d)


def run_edge(world, arm, src, dst, entry_snap, sel_rng, seed, attempt,
             evals_used):
    """Beam search for one directed edge. Returns dict with feasibility,
    evals spent, achieved snapshot, path provenance, per-level history."""
    cap = EDGE_EVAL_CAP
    spent = 0.0
    world.restore(entry_snap)
    beam = [(score(world) + 3.0 * modes.mode_term(dst, modes.orientation(world)),
             entry_snap, [])]
    feasible = False
    achieved = entry_snap
    feas_path = []
    history = []
    levels_run = 0
    for level in range(EDGE_LEVELS):
        if evals_used + spent >= BUDGET or spent >= cap:
            break
        levels_run = level + 1
        cand, states = [], []
        for b, (_, snap, _) in enumerate(beam):
            world.restore(snap)
            states.append(world.observe())
            cand.append(arms.generate_proposals(
                np.random.default_rng([seed, modes.EDGE_INDEX[(src, dst)],
                                       attempt, level, b])))
        pool = list(beam)
        exhausted = False
        for b, (_, snap, path) in enumerate(beam):
            chosen = _select(arm, states[b], cand[b], sel_rng)
            for (j, tag) in chosen:
                if evals_used + spent >= BUDGET or spent >= cap:
                    exhausted = True
                    break
                action = cand[b][j]
                world.restore(snap)
                world.advance(action)
                spent += 1.0
                m2, s2 = modes.classify(world)
                # edge feasibility: destination predicate.
                # Lying modes (SIDE/PRONE) are passively stable: require the
                # predicate to survive a 150-step passive settle (0.5 eval,
                # counted) so a "mode" is a resting config, not a transient
                # pass-through. STAND is NOT passively stable (an uncontrolled
                # standing humanoid always topples), so a passive settle there
                # would guaranteed-reject genuine arrivals: instead require
                # the predicate at end of advance + kinetic_j < 0.10
                # (quasi-static gate; measured end-of-advance kinetic median
                # 0.037 / p90 0.079, so 0.10 rejects only the wildest ~5-10%).
                # The refinement edge (STAND->STAND) uses E1's handover_ok
                # unchanged (no gate), for comparability with E1.
                hit = False
                if dst == STAND and src == STAND:
                    hit = arms.handover_ok(world.metrics())
                else:
                    hit = (m2 == dst)
                if hit:
                    if dst == STAND:
                        if world.kinetic() < STAND_KINETIC_GATE:
                            feasible = True
                            achieved = world.snapshot()
                            feas_path = path + [(action.copy(), tag)]
                            break
                    elif evals_used + spent + SETTLE_VERIFY_COST <= BUDGET:
                        _settle_verify(world)
                        spent += SETTLE_VERIFY_COST
                        m3, _ = modes.classify(world)
                        if m3 == dst:
                            feasible = True
                            achieved = world.snapshot()
                            feas_path = path + [(action.copy(), tag)]
                            break
                es = score(world) + 3.0 * modes.mode_term(dst, modes.orientation(world))
                pool.append((es, world.snapshot(), path + [(action.copy(), tag)]))
            if feasible or exhausted:
                break
        if feasible:
            break
        # diversity selection: identical cells to E1/vendor
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
        m_now, _ = modes.classify(world)
        history.append({'level': level + 1, 'edge_evals': round(spent, 1),
                        'mode': m_now,
                        'winning_edge_score': float(beam[0][0])})
        if exhausted:
            break
    if not feasible:
        # best-effort achieved state = beam head (for diagnostics only)
        achieved = beam[0][1]
        feas_path = beam[0][2]
    tags = [t for _, t in feas_path]
    return {
        'src': src, 'dst': dst, 'attempt': attempt,
        'feasible': bool(feasible),
        'evals': round(spent, 1),
        'levels': levels_run,
        'achieved_mode': None,  # filled by caller after re-classify
        '_achieved_snap': achieved,  # numpy snapshot; popped before JSON dump
        'path_len': len(feas_path),
        'tag_counts': {t: int(tags.count(t)) for t in sorted(set(tags))},
        'actions': [a.tolist() for a, _ in feas_path],
        'history': history,
    }


def run_one(task):
    arm, pose, seed = task
    t0 = time.time()
    result = {'arm': arm, 'pose': pose, 'seed': seed, 'status': 'ok',
              'budget': BUDGET, 'width': WIDTH,
              'edge_levels': EDGE_LEVELS, 'edge_eval_cap': EDGE_EVAL_CAP,
              'max_edge_attempts': MAX_EDGE_ATTEMPTS}
    try:
        world = GroundWorld()
        pw = passive_wide(world, pose, seed)
        result['passive'] = pw
        initial = world.snapshot()
        result['initial_metrics'] = {k: float(v) for k, v in world.metrics().items()}
        m0, _ = modes.classify(world)
        result['initial_mode'] = m0
        sel_rng = np.random.default_rng(60_000 + ARM_INDEX[arm] * 100_000 + (seed % 100_000))

        evals = 0.0
        entry = initial
        mode = m0 if m0 in modes.SUCCESSORS or m0 == STAND else SUPINE
        if m0 != mode:
            result['initial_mode_fallback'] = {'measured': m0, 'used': mode}
        edges = []
        attempts = {}
        success = False
        evals_to_success = BUDGET
        balance_error = None

        def finish_edge(er):
            """Pop the numpy snapshot, re-classify achieved state, log edge."""
            ach = er.pop('_achieved_snap')
            world.restore(ach)
            m2, _ = modes.classify(world)
            er['achieved_mode'] = m2
            edges.append(er)
            return ach, m2

        while evals < BUDGET:
            world.restore(entry)
            mode, _ = modes.classify(world)
            if mode == TRANSITION:
                mode = SUPINE  # fallback: treat as SUPINE outgoing set
            if mode == STAND:
                # final handover: E1's criterion + balance()
                r = world.metrics()
                if arms.handover_ok(r):
                    try:
                        ok, _, _ = balance(world)
                    except Exception:
                        balance_error = traceback.format_exc(limit=3)
                        ok = False
                    if ok:
                        success = True
                        evals_to_success = evals
                        break
                # refinement edge STAND->STAND (feasibility = handover_ok)
                key = (STAND, STAND)
                att = attempts.get(key, 0)
                if att >= MAX_EDGE_ATTEMPTS:
                    break
                attempts[key] = att + 1
                er = run_edge(world, arm, STAND, STAND, entry, sel_rng,
                              seed, att, evals)
                evals += er['evals']
                ach, _ = finish_edge(er)
                if not er['feasible']:
                    break  # refinement failed -> run fails
                entry = ach  # loop re-checks handover_ok + balance()
                continue
            # choose outgoing edge (SUPINE tries SIDE first, then PRONE)
            cands = [d for d in modes.SUCCESSORS[mode]
                     if attempts.get((mode, d), 0) < MAX_EDGE_ATTEMPTS]
            if not cands:
                break
            dst = cands[0]
            key = (mode, dst)
            att = attempts.get(key, 0)
            attempts[key] = att + 1
            er = run_edge(world, arm, mode, dst, entry, sel_rng, seed, att,
                          evals)
            evals += er['evals']
            ach, m2 = finish_edge(er)
            if not er['feasible']:
                continue  # try next outgoing edge (or fail if none left)
            # edge feasible: continue from achieved state. Descent allowed:
            # m2 is the re-classified mode and may differ from dst.
            entry = ach

        result.update({
            'success': bool(success),
            'physics_evaluations': round(evals, 1),
            'evals_to_handover': float(evals_to_success),
            'final_mode': mode,
            'n_edges': len(edges),
            'edges': [{k: v for k, v in er.items() if k != '_achieved_snap'}
                      for er in edges],
            'final_metrics': {k: float(v) for k, v in world.metrics().items()},
            'wall_seconds': time.time() - t0,
        })
        if balance_error:
            result['balance_error'] = balance_error
        # provenance: aggregate tag counts over the successful chain
        tag_tot = {}
        for er in edges:
            for t, c in er.get('tag_counts', {}).items():
                tag_tot[t] = tag_tot.get(t, 0) + c
        result['provenance'] = {'tag_counts': tag_tot,
                                'n_edges': len(edges)}
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
    torch.set_num_threads(1)
    _NETS = {'net': Ensemble()}


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
        print('SMOKE: single in-process run (gr/prone/1000)', flush=True)
        print(json.dumps(run_one(('gr', 'prone', 1000)), indent=2), flush=True)
        return

    want = [x.strip() for x in a.arms.split(',') if x.strip()]
    tasks = [(arm, pose, seed) for arm in want for pose in POSES for seed in SEEDS[pose]]
    tasks = [t for t in tasks if not done(t)]
    print(f'E5: {len(tasks)} runs pending ({a.workers} workers, '
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
        for t in batch:
            if not done(t) and t not in remaining:
                remaining.append(t)
    print('E5 sweep complete.', flush=True)


if __name__ == '__main__':
    main()
