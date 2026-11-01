"""Shared-budget multi-target edge oracle; physics is the only proposal evaluator."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from .plant import GroundWorld, BOUNDS, balance, np, ROOT
from .modes import classify, prior, handover, edge_key
from .score import evaluate, sample, score_v3, exploration_priority

OUT = ROOT / 'combo/results'


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


@dataclass
class Node:
    value: float
    snap: np.ndarray
    path: list
    mode: str
    born: int
    metrics: dict


def choose(pool, width, selection="robust"):
    pool = sorted(pool, key=lambda n: n.value, reverse=True)
    selected, modes, cells = [], set(), set()
    if selection == "dynamic":
        # A separate exploration queue can retain useful momentum without
        # claiming that a fast snapshot is settled progress. Same parent expiry.
        dynamic_cells = set()
        for n in sorted(pool, key=lambda n: exploration_priority(n.metrics), reverse=True):
            r, q = n.metrics, n.snap[7:31]
            cell = (n.mode, int(r['base_height']/.025), int(r['up_z']/.12),
                    int((q[0]+q[6])/.6), int((q[3]+q[9])/.7))
            if cell not in dynamic_cells:
                selected.append(n)
                dynamic_cells.add(cell)
            if len(selected) >= min(6, width//2):
                break
    # Four total seats protect alternative contact strategies in robust mode;
    # dynamic mode adds up to two mode representatives to its exploration seats.
    mode_seats = len(selected) + (2 if selection == "dynamic" else 4)
    for n in pool:
        if n.mode not in modes and all(n is not v for v in selected):
            selected.append(n)
            modes.add(n.mode)
        if len(selected) >= min(mode_seats, width):
            break
    for n in pool:
        q, r = n.snap[7:31], n.metrics
        cell = (n.mode, int(r['com_height']/.025), int(r['up_z']/.12),
                int((q[0]+q[6])/.6), int((q[3]+q[9])/.7))
        if cell not in cells and all(n is not v for v in selected):
            selected.append(n)
            cells.add(cell)
        if len(selected) >= width:
            break
    return selected[:width] if selection == "robust" else sorted(selected[:width], key=lambda n: n.value, reverse=True)


def run(pose, seed, budget=6000, width=14, branches=16, selection="robust"):
    out = OUT if selection == "robust" else OUT / selection
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{pose}_{seed}'
    result_path, checkpoint = out/f'{tag}.json', out/f'{tag}.checkpoint.pkl'
    if result_path.exists():
        return json.loads(result_path.read_text())
    w = GroundWorld()
    rng = np.random.default_rng(2027+seed)
    history, attempts, edges, evals, level, elapsed = [], [], {}, 0, 0, 0.
    if checkpoint.exists():
        # Only load this locally generated checkpoint, never untrusted pickle.
        state = pickle.loads(checkpoint.read_bytes())
        assert state['config'] == (pose, seed, budget, width, branches)
        assert state.get('selection', 'robust') == selection
        initial, beam, history, attempts, edges, evals, level, elapsed = [state[k] for k in
            ('initial','beam','history','attempts','edges','evals','level','elapsed')]
        rng.bit_generator.state = state['rng']
        settle = state['settle']
    else:
        settle = w.passive(pose, seed)
        settle.pop('trace')
        initial = w.snapshot()
        beam = [Node(score_v3([sample(w)]), initial, [], classify(w), 0, w.metrics())]
    witnesses = state.get('witnesses', {}) if checkpoint.exists() else {}
    start = time.time()
    success, winner = False, None
    while evals < budget and not success:
        level += 1
        # Parents are retained briefly, never indefinitely frozen at a lucky peak.
        pool = [n for n in beam if level-n.born <= 2]
        for parent in beam:
            for j in range(branches):
                if evals >= budget:
                    break
                action = rng.normal(0, .7, 16)*BOUNDS
                if j < branches//2:
                    action[9:14] = 0
                    action[15] = 0
                    action[[3,4,8,14]] *= .3
                if j == 0:
                    action[:] = 0
                action = np.clip(action, -BOUNDS, BOUNDS)
                w.restore(parent.snap)
                _, frames = w.advance(action, trace=True)
                evals += 1
                value = evaluate(w, frames)
                r, mode = w.metrics(), classify(w)
                if selection == 'dynamic':
                    r['foot_height'] = float(np.mean(w.d.xipos[[w.bodies[6], w.bodies[12]], 2]))
                n = Node(value-prior(parent.mode, mode), w.snapshot(),
                         parent.path+[action], mode, level, r)
                edge = edge_key(parent.mode, mode)
                edges[edge] = edges.get(edge, 0)+1
                if selection == 'dynamic' and (edge not in witnesses or value > witnesses[edge]['value']):
                    witnesses[edge] = dict(value=value, source=parent.snap.copy(),
                                           destination=n.snap.copy(), action=action.copy())
                pool.append(n)
                if handover(r):
                    ok, bt, br = balance(w)
                    attempts.append(dict(evaluation=evals, success=ok, entry=r,
                                         final=w.metrics(), seconds=len(br)*.02))
                    if ok:
                        success, winner = True, n
                        break
            if success:
                break
        if not success:
            beam = choose(pool, width, selection)
        best = winner if success else beam[0]
        history.append(dict(level=level, evaluations=evals, score=best.value,
                            mode=best.mode, metrics=best.metrics))
        print(tag, level, evals, best.mode, round(best.value,3),
              'handover', len(attempts), 'success', success, flush=True)
        np.savez_compressed(out/f'{tag}_path.npz', initial=initial,
                            actions=np.asarray(best.path), final=best.snap)
        state = dict(config=(pose,seed,budget,width,branches), selection=selection, initial=initial,
                     beam=beam, history=history, attempts=attempts, edges=edges,
                     evals=evals, level=level, elapsed=elapsed+time.time()-start,
                     rng=rng.bit_generator.state, settle=settle, witnesses=witnesses)
        tmp = checkpoint.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(checkpoint)
    best = winner if success else beam[0]
    if witnesses:
        np.savez_compressed(out/f'{tag}_edges.npz', labels=np.array(list(witnesses)),
                            sources=np.array([v['source'] for v in witnesses.values()]),
                            destinations=np.array([v['destination'] for v in witnesses.values()]),
                            actions=np.array([v['action'] for v in witnesses.values()]))
    result = dict(pose=pose, seed=seed, selection=selection, success=success, evaluations=evals,
                  budget=budget, width=width, branches=branches, settle=settle,
                  seconds=elapsed+time.time()-start, history=history,
                  handovers=attempts, observed_edges=edges, final=best.metrics,
                  action_blocks=len(best.path), source_sha256=hashlib.sha256(
                      Path(__file__).read_bytes()).hexdigest())
    write_json(result_path, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pose', choices=['supine','prone'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=int, default=6000)
    p.add_argument("--selection", choices=["robust", "dynamic"], default="robust")
    a = p.parse_args()
    run(a.pose, a.seed, a.budget, selection=a.selection)
