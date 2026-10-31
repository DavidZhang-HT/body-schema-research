#!/usr/bin/env python
"""E1 analysis: per-arm success rates, Kaplan-Meier, manual log-rank, Fisher
exact, provenance breakdown. Reads runs/e1_*.json, writes summary.json and
prints report tables.

Event model: event = handover+balance success, time = evals_to_handover
(physics evaluations). Runs that never hand over are censored at their final
physics_evaluations (4000 = full budget; TIMEOUT/ERROR runs censored at 4000
with a flag).
"""
import os
import sys
import json
import glob
import math
import numpy as np
from scipy.stats import chi2, fisher_exact

WORKDIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(WORKDIR, 'runs')
ARMS = ['net', 'rand', 'heur', 'shuf']
ARM_LABEL = {'net': '(a) network', 'rand': '(b) random',
             'heur': '(c) heuristic', 'shuf': '(d) shuffled-net'}


def load():
    recs = []
    for p in sorted(glob.glob(os.path.join(RUNS, 'e1_*.json'))):
        with open(p) as f:
            r = json.load(f)
        recs.append(r)
    return recs


def km_curve(times, events):
    """Kaplan-Meier step function. Returns (t_grid, surv) including t=0."""
    order = np.argsort(times, kind='stable')
    times = np.asarray(times, dtype=float)[order]
    events = np.asarray(events, dtype=int)[order]
    ts, surv = [0.0], [1.0]
    s = 1.0
    i = 0
    n = len(times)
    while i < n:
        t = times[i]
        j = i
        while j < n and times[j] == t:
            j += 1
        d = int(events[i:j].sum())
        at_risk = n - i
        if d > 0 and at_risk > 0:
            s *= (at_risk - d) / at_risk
        ts.append(float(t))
        surv.append(s)
        i = j
    return ts, surv


def km_median(times, events):
    ts, surv = km_curve(times, events)
    for t, s in zip(ts, surv):
        if s <= 0.5:
            return t
    return None


def logrank(times1, events1, times2, events2):
    """Two-group log-rank test (group 1 first). Returns dict with
    chi2, p, and direction sign of (O1 - E1): positive => group 1 has MORE
    events than expected under H0 (here: faster handover)."""
    t1 = np.asarray(times1, dtype=float)
    e1 = np.asarray(events1, dtype=int)
    t2 = np.asarray(times2, dtype=float)
    e2 = np.asarray(events2, dtype=int)
    O1 = E1 = V = 0.0
    for t in sorted(set(t1.tolist()) | set(t2.tolist())):
        n1 = int((t1 >= t).sum())
        n2 = int((t2 >= t).sum())
        d1 = int(((t1 == t) & (e1 == 1)).sum())
        d2 = int(((t2 == t) & (e2 == 1)).sum())
        d, n = d1 + d2, n1 + n2
        if d == 0 or n <= 1:
            continue
        E1 += d * n1 / n
        O1 += d1
        V += (n1 * n2 * d * (n - d)) / (n * n * (n - 1))
    if V <= 0:
        return {'chi2': 0.0, 'p': 1.0, 'O1-E1': 0.0, 'note': 'no variance'}
    z = (O1 - E1) / math.sqrt(V)
    c2 = z * z
    return {'chi2': c2, 'p': float(chi2.sf(c2, 1)), 'O1-E1': O1 - E1,
            'O1': O1, 'E1': E1}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / den), min(1.0, (c + m) / den))


def main():
    recs = load()
    print(f'loaded {len(recs)} run files', flush=True)
    by_arm = {a: [] for a in ARMS}
    for r in recs:
        by_arm[r['arm']].append(r)

    summary = {'n_files': len(recs), 'arms': {}}
    for a in ARMS:
        rs = by_arm[a]
        ok = [r for r in rs if r.get('status') == 'ok']
        timeout = [r for r in rs if r.get('status') == 'timeout']
        err = [r for r in rs if r.get('status') == 'error']
        succ = [r for r in ok if r.get('success')]
        # Survival data: event at evals_to_handover; censored at final evals
        # (TIMEOUT/ERROR censored at full budget, flagged).
        times, events = [], []
        for r in ok:
            if r.get('success'):
                times.append(r['evals_to_handover'])
                events.append(1)
            else:
                times.append(r.get('physics_evaluations') or 4000)
                events.append(0)
        for r in timeout + err:
            times.append(4000)
            events.append(0)
        med = km_median(times, events)
        med_succ = (float(np.median([r['evals_to_handover'] for r in succ]))
                    if succ else None)
        k, n = len(succ), len(rs)
        lo, hi = wilson(k, n)
        walls = [r['wall_seconds'] for r in ok if r.get('wall_seconds')]
        summary['arms'][a] = {
            'n_runs': n, 'n_ok': len(ok), 'n_timeout': len(timeout),
            'n_error': len(err), 'n_success': k,
            'success_rate': k / n if n else 0.0,
            'success_rate_wilson95': [lo, hi],
            'km_median_evals_to_handover': med,
            'median_evals_among_successes': med_succ,
            'mean_wall_s': float(np.mean(walls)) if walls else None,
            'total_wall_s': float(np.sum(walls)) if walls else 0.0,
        }
        # Provenance over winning paths of successful runs
        tag_tot = {}
        for r in succ:
            for t, c in r['provenance']['tag_counts'].items():
                tag_tot[t] = tag_tot.get(t, 0) + c
        summary['arms'][a]['winning_path_tag_counts'] = tag_tot
        summary['arms'][a]['n_successful_paths'] = len(succ)

    # Pairwise tests pre-registered: (a)vs(b), (a)vs(c)
    pairs = [('net', 'rand'), ('net', 'heur'), ('net', 'shuf'),
             ('rand', 'heur'), ('rand', 'shuf'), ('heur', 'shuf')]
    summary['logrank'] = {}
    summary['fisher'] = {}
    arm_data = {}
    for a in ARMS:
        ts, ev = [], []
        for r in by_arm[a]:
            if r.get('status') == 'ok' and r.get('success'):
                ts.append(r['evals_to_handover']); ev.append(1)
            else:
                ts.append(r.get('physics_evaluations') or 4000); ev.append(0)
        arm_data[a] = (ts, ev)
    for x, y in pairs:
        lr = logrank(*arm_data[x], *arm_data[y])
        summary['logrank'][f'{x}_vs_{y}'] = lr
        kx, nx = summary['arms'][x]['n_success'], summary['arms'][x]['n_runs']
        ky, ny = summary['arms'][y]['n_success'], summary['arms'][y]['n_runs']
        _, p = fisher_exact([[kx, nx - kx], [ky, ny - ky]])
        summary['fisher'][f'{x}_vs_{y}'] = {'p': float(p)}

    # Falsifier (fixed before results): (a) must beat (b) AND beat (c) at
    # p<0.05 (log-rank, direction favouring (a)) for attention screening to
    # have search value.
    def beats(x, y):
        lr = summary['logrank'][f'{x}_vs_{y}']
        return lr['p'] < 0.05 and lr['O1-E1'] > 0
    a_beats_b = beats('net', 'rand')
    a_beats_c = beats('net', 'heur')
    summary['falsifier'] = {
        'a_beats_b_p05': bool(a_beats_b),
        'a_beats_c_p05': bool(a_beats_c),
        'verdict': ('attention screening has search value'
                    if (a_beats_b and a_beats_c)
                    else 'FALSIFIED: attention screening adds no search value '
                         '(network arm not better than random AND/OR not '
                         'better than the FK heuristic at p<0.05)'),
    }

    with open(os.path.join(WORKDIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Console tables
    print('\n=== per-arm results (n=20/arm) ===')
    print(f"{'arm':16} {'succ':>5} {'rate':>6} {'95%CI':>16} "
          f"{'KMmed':>7} {'med|succ':>8} {'ok/timeout/err':>14}")
    for a in ARMS:
        s = summary['arms'][a]
        lo, hi = s['success_rate_wilson95']
        print(f"{ARM_LABEL[a]:16} {s['n_success']:>5} "
              f"{s['success_rate']:>6.2f} [{lo:.2f},{hi:.2f}] "
              f"{str(s['km_median_evals_to_handover']):>7} "
              f"{str(s['median_evals_among_successes']):>8} "
              f"{s['n_ok']:>2}/{s['n_timeout']:>2}/{s['n_error']:>2}")
    print('\n=== log-rank (H0: identical handover curves; O1-E1>0 favours first) ===')
    for k, v in summary['logrank'].items():
        print(f"{k:14} chi2={v['chi2']:.3f} p={v['p']:.4f} O1-E1={v['O1-E1']:+.2f}")
    print('\n=== Fisher exact on success counts ===')
    for k, v in summary['fisher'].items():
        print(f"{k:14} p={v['p']:.4f}")
    print('\n=== winning-path provenance (tag counts over successful runs) ===')
    for a in ARMS:
        s = summary['arms'][a]
        print(f"{ARM_LABEL[a]:16} paths={s['n_successful_paths']} "
              f"{s['winning_path_tag_counts']}")
    print('\n=== FALSIFIER ===')
    print(summary['falsifier']['verdict'])
    print('\nsaved summary.json', flush=True)


if __name__ == '__main__':
    main()
