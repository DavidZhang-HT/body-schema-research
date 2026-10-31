"""E3 step 3: re-run B's physics beam search FROM the plateau snapshot, with the original 12-synergy
basis and with an augmented basis, under the UNMODIFIED score() of codex/ground_search.py.

Protocol = codex/ground_search.search() (physics-guided oracle), copied loop-for-loop:
  width 14, 16 branches per beam state, actions ~ N(0, 0.7)*BOUNDS clipped, first half of the
  branches symmetric (differential dims zeroed, spread/yaw dims x0.3), branch 0 = zero action,
  pool = beam + children sorted by score(), diversity cells on (com_h/.025, up/.12, hip-pitch sum, knee sum).
Differences (documented): start = plateau snapshot instead of the passive state; rng seed 2027+s;
B's learned-network screening is not used (the network only accepts 12-dim actions); success =
B's hybrid handover criterion on beam[0] followed by a 1.2 s zero-action hold (stand-in for balance()).

Augmented basis (ground_schema.B / BOUNDS are swapped in-process, the file is not edited):
  12 differential_hip_roll   l_hip_roll +1, r_hip_roll +1   bound 0.30   (asymmetric)
  13 differential_ankle_roll l_ankle_roll +1, r_ankle_roll +1 bound 0.30 (asymmetric)
  14 ankle_roll_spread       l_ankle_roll +1, r_ankle_roll -1 bound 0.30 (mirror-symmetric, x0.3 in sym half)
  15 asymmetric_knee         l_calf +1, r_calf -1           bound 0.50   (asymmetric)
"""
import json, sys, time, argparse
import numpy as np
from pathlib import Path
import e3_plant
from e3_plant import GroundWorld, score
import ground_schema as gs

OUT = Path(__file__).resolve().parent / 'results' / 'step3'; OUT.mkdir(parents=True, exist_ok=True)
B12, BOUNDS12 = gs.B.copy(), gs.BOUNDS.copy()


def augmented():
    B = np.zeros((24, 16)); B[:, :12] = B12
    B[1, 12] = 1; B[7, 12] = 1
    B[5, 13] = 1; B[11, 13] = 1
    B[5, 14] = 1; B[11, 14] = -1
    B[3, 15] = 1; B[9, 15] = -1
    bounds = np.r_[BOUNDS12, .30, .30, .30, .50]
    return B, bounds


def handover(r):
    return r['base_height'] > .16 and r['up_z'] > .9 and r['foot_load'] > 75 and r['nonfoot_load'] < 8


def strict(r):
    return r['base_height'] > .3 and r['up_z'] > .92 and r['foot_load'] > 75 and r['nonfoot_load'] < 8 and r['com_speed'] < .2


def hold_test(w, snap, blocks=2, nact=12):
    w.restore(snap); ok = True; hist = []
    for _ in range(blocks):
        w.advance(np.zeros(nact)); r = w.metrics(); hist.append(r); ok = ok and handover(r)
    return ok, hist


def score_v2(w):
    """POST-HOC score variant (not pre-registered): B's score() minus a non-foot-load penalty."""
    r = w.metrics()
    return score(w) - 0.8 * min(r['nonfoot_load'] / 108, 1)


def run(arm, seed, depth, width=14, branches=16, start='plateau_snapshot', score_name='orig'):
    global score
    if score_name == 'v2':
        _orig = score
        score = lambda w: _orig(w) - 0.8 * min(w.metrics()['nonfoot_load'] / 108, 1)
    if arm == 'aug16':
        gs.B, gs.BOUNDS = augmented()
    else:
        gs.B, gs.BOUNDS = B12.copy(), BOUNDS12.copy()
    B, BOUNDS = gs.B, gs.BOUNDS; n = B.shape[1]
    w = GroundWorld()
    snap0 = np.load(Path(__file__).resolve().parent / 'results' / f'{start}.npy')
    w.restore(snap0); initial = w.snapshot(); start_score = float(score(w))
    rng = np.random.default_rng(2027 + seed)
    beam = [(score(w), initial, [])]; history = []; evals = 0; t0 = time.time()
    best_seen = dict(score=-1e9); any_handover = 0; success = False; succ_info = None; ho_states = []
    tag = f'{arm}_s{seed}' + ('' if score_name == 'orig' else '_score' + score_name)
    for level in range(depth):
        pool = list(beam)
        for value, snap, path in beam:
            for j in range(branches):
                action = rng.normal(0, .7, n) * BOUNDS
                if j < branches // 2:
                    action[9:12] = 0; action[3:5] *= .3; action[8] *= .3
                    if n > 12:
                        action[[12, 13, 15]] = 0; action[14] *= .3
                if j == 0:
                    action = np.zeros(n)
                action = np.clip(action, -BOUNDS, BOUNDS)
                w.restore(snap); w.advance(action); v = score(w); evals += 1
                r = w.metrics()
                if handover(r):
                    any_handover += 1
                    ho_states.append(dict(level=level + 1, evals=evals, score=float(v), metrics=r,
                                          snap=w.snapshot(), path=[a.copy() for a in path] + [action.copy()]))
                pool.append((v, w.snapshot(), path + [action.copy()]))
        pool.sort(key=lambda e: e[0], reverse=True); beam = []; cells = set()
        for entry in pool:
            w.restore(entry[1]); r = w.metrics(); q = w.d.qpos[7:]
            cell = (int(r['com_height'] / .025), int(r['up_z'] / .12), int((q[0] + q[6]) / .6), int((q[3] + q[9]) / .7))
            if cell not in cells: beam.append(entry); cells.add(cell)
            if len(beam) >= width: break
        if len(beam) < width:
            for entry in pool:
                if all(entry is not b for b in beam): beam.append(entry)
                if len(beam) >= width: break
        w.restore(beam[0][1]); r = w.metrics()
        history.append({'level': level + 1, 'evals': evals, 'score': float(beam[0][0]), 'n_actions': len(beam[0][2]), **r})
        print(tag, 'level', level + 1, 'evals', evals, 'score', round(beam[0][0], 3), 'base', round(r['base_height'], 3),
              'up', round(r['up_z'], 3), 'feet', round(r['foot_load'], 1), 'nonfoot', round(r['nonfoot_load'], 1),
              'any_handover_rollouts', any_handover, 'wall', round(time.time() - t0), flush=True)
        np.savez_compressed(OUT / f'{tag}_best.npz', initial=initial, actions=np.array(beam[0][2]), final=beam[0][1])
        if handover(r):
            ok, hh = hold_test(w, beam[0][1], nact=n)
            history[-1]['hold_ok'] = ok; history[-1]['hold'] = hh
            if ok:
                success = True; succ_info = dict(level=level + 1, evals=evals, strict=strict(r)); break
        (OUT / f'{tag}.json').write_text(json.dumps(dict(arm=arm, seed=seed, history=history, partial=True), indent=1))
    w.restore(beam[0][1]); final = w.metrics()
    # diagnostic: rollouts that met B's handover criterion but were NOT ranked first by score()
    ho_report = []
    for h in ho_states:
        ok, hh = hold_test(w, h['snap'], nact=n)
        ho_report.append(dict(level=h['level'], evals=h['evals'], score=h['score'], metrics=h['metrics'],
                              hold_ok=ok, hold_end=hh[-1], n_actions=len(h['path'])))
    if ho_states:
        np.savez_compressed(OUT / f'{tag}_handover_states.npz', snaps=np.array([h['snap'] for h in ho_states]),
                            scores=np.array([h['score'] for h in ho_states]), initial=initial,
                            paths=np.array([np.array(h['path']) for h in ho_states], dtype=object))
    w.restore(beam[0][1]); final = w.metrics()
    res = dict(arm=arm, seed=seed, score_name=score_name, start=start, depth=depth, width=width, branches=branches, n_synergies=n,
               success=success, success_info=succ_info, evaluations=evals, any_handover_rollouts=any_handover,
               final=final, final_score=float(beam[0][0]), final_qpos=w.d.qpos.tolist(),
               n_actions=len(beam[0][2]), wall=time.time() - t0, history=history, start_score=start_score,
               handover_rollouts=ho_report)
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    print('DONE', tag, 'success', success, 'evals', evals, 'final', {k: round(v, 3) for k, v in final.items()}, flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', default='orig12'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--depth', type=int, default=24); p.add_argument('--start', default='plateau_snapshot'); p.add_argument('--score', default='orig')
    a = p.parse_args(); run(a.arm, a.seed, a.depth, start=a.start, score_name=a.score)
