"""Replay and certify every returned block, including interior samples and QP.

A static failure at a sampled configuration is evidence about THAT path under
E3's conservative contact LP, not proof that every route requires momentum.
"""
import argparse
import math
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from .plant import GroundWorld, BSchema, balance, np, ROOT
from .modes import classify
from .search import write_json


def finite(obj):
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def certificate(schema, snap):
    c = schema.certify(snap[:31])
    points = schema.points_world()
    xy = points[points[:, 2] <= .005, :2]
    c['support_polygon_xy'] = []
    c['com_margin_m'] = None
    if len(xy) >= 3:
        try:
            hull = ConvexHull(xy)
            c['support_polygon_xy'] = xy[hull.vertices].tolist()
            equations = hull.equations
            c['com_margin_m'] = float(-np.max(equations[:, :2] @ np.array(c['com'][:2]) + equations[:, 2]))
        except QhullError:
            pass
    return finite(c)


def transition_label(samples):
    gaps = [i for i, s in enumerate(samples) if not s['certificate']['feasible'] or
            (s['certificate']['rho'] is not None and s['certificate']['rho'] > .8)]
    invalid = [i for i, s in enumerate(samples) if s['certificate']['pen'] < -.0015 or
               s['certificate']['selfd'] < -.001]
    if gaps:
        return dict(label='STATICS_GAP', gap_samples=gaps,
                    interpretation='Sampled path lacks a static certificate; momentum necessity for all paths unproven.',
                    geometry_invalid_samples=invalid, momentum_required_proven=False)
    if not invalid and max(s['metrics']['kinetic_j'] for s in samples) <= .005:
        return dict(label='QUASI_STATIC', gap_samples=[], geometry_invalid_samples=[],
                    interpretation='All sampled states certified with KE <= 0.005 J; unsampled intervals unproven.',
                    momentum_required_proven=False)
    return dict(label='DYNAMIC_UNRESOLVED', gap_samples=[], geometry_invalid_samples=invalid,
                interpretation='Fast motion or invalid geometry; no sampled statics gap establishes momentum necessity.',
                momentum_required_proven=False)


def explain(pose, seed, selection="robust"):
    out = ROOT/'combo/results'
    if selection != 'robust':
        out = out / selection
    tag = f'{pose}_{seed}'
    data = np.load(out/f'{tag}_path.npz')
    w = GroundWorld()
    schema = BSchema(w.m)
    w.restore(data['initial'])
    trace = [w.snapshot()]
    transitions = []
    sequence = [classify(w)]

    def inspect(snap):
        w.restore(snap)
        return dict(time=float(snap[-1]), mode=classify(w), metrics=w.metrics(),
                    certificate=certificate(schema, snap))

    for i, action in enumerate(data['actions']):
        begin = w.snapshot()
        _, frames = w.advance(action, trace=True)
        end = w.snapshot()
        trace.extend(frames)
        trace.append(end)
        # 100-ms sampling plus both exact endpoints; every block end certified.
        samples = [inspect(s) for s in [begin]+frames[4::5]+[end]]
        for s in samples:
            if sequence[-1] != s['mode']:
                sequence.append(s['mode'])
        transitions.append(dict(block=i, source=samples[0]['mode'], destination=samples[-1]['mode'],
                                **transition_label(samples), samples=samples))
        w.restore(end)
    replay_error = float(np.max(np.abs(w.snapshot()-data['final'])))
    ok, bt, records = balance(w)
    trace.extend(bt)
    if bt:
        samples = [inspect(s) for s in bt[::5]] + [inspect(bt[-1])]
        for s in samples:
            if sequence[-1] != s['mode']:
                sequence.append(s['mode'])
        transitions.append(dict(block='balance', source=samples[0]['mode'], destination=samples[-1]['mode'],
                                **transition_label(samples), samples=samples))
    result = dict(pose=pose, seed=seed, replay_success=ok, replay_snapshot_max_error=replay_error,
                  mode_sequence=sequence, sample_period_s=.1, transitions=transitions,
                  final=records[-1] if records else None)
    write_json(out/f'{tag}_explanation.json', result)
    np.savez_compressed(out/f'{tag}_trace.npz', trace=np.asarray(trace))
    print(tag, 'replay', ok, 'error', replay_error, sequence, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pose', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument("--selection", choices=["robust", "dynamic"], default="robust")
    a = p.parse_args()
    explain(a.pose, a.seed, a.selection)
