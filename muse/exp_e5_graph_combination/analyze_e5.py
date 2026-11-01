#!/usr/bin/env python
"""E5 analysis: per-arm success rates, Kaplan-Meier, log-rank, Fisher exact.

Arms:
  g  - graph + network  (e5_g_*.json, this workdir)
  gr - graph + random   (e5_gr_*.json, this workdir)
  f  - flat baseline    (e1_net_*.json from E1's runs/, REUSED as-is)

Event model: event = handover+balance success, time = evals_to_handover
(physics evaluations). Non-success runs censored at final physics_evaluations
(TIMEOUT/ERROR censored at full budget 4000, flagged).

Statistical machinery (km_curve, km_median, logrank, wilson) is reused from
E1's analyze_e1.py; only the arm set, loading, and falsifier differ.

Pre-registered comparisons: G vs F (graph+model vs flat+model),
GR vs F (graph effect without model). Exploratory: G vs GR.

FALSIFIER (fixed before results): the graph adds EFFICIENCY value only if G
beats F (log-rank p<0.05 with O1-E1>0 favouring G). Otherwise the verdict is
that the graph contributes INTERPRETABILITY ONLY, not efficiency.
"""
import os
import sys
import json
import glob
import numpy as np
from scipy.stats import fisher_exact

WORKDIR = os.path.dirname(os.path.abspath(__file__))
E1DIR = '/home/hatch/workspace/research/body-schema-gait/muse/exp_e1_proposal_control'
sys.path.insert(0, E1DIR)  # reuse analyze_e1's stats
from analyze_e1 import km_curve, km_median, logrank, wilson

ARMS = ['g', 'gr', 'f']
ARM_LABEL = {'g': 'G graph+network', 'gr': 'GR graph+random',
             'f': 'F flat+network (E1 net, reused)'}
RUN_GLOB = {
    'g': os.path.join(WORKDIR, 'runs', 'e5_g_*.json'),
    'gr': os.path.join(WORKDIR, 'runs', 'e5_gr_*.json'),
    'f': os.path.join(E1DIR, 'runs', 'e1_net_*.json'),
}


def load():
    recs = {a: [] for a in ARMS}
    for a in ARMS:
        for p in sorted(glob.glob(RUN_GLOB[a])):
            with open(p) as fh:
                r = json.load(fh)
            r['_arm'] = a
            recs[a].append(r)
    return recs


def surv_data(rs):
    """(times, events): event at evals_to_handover; censored at final evals."""
    ts, ev = [], []
    for r in rs:
        if r.get('status') == 'ok' and r.get('success'):
            ts.append(float(r['evals_to_handover']))
            ev.append(1)
        else:
            ts.append(float(r.get('physics_evaluations') or 4000))
            ev.append(0)
    return ts, ev


def main():
    recs = load()
    print('loaded: ' + ', '.join(f'{a}={len(recs[a])}' for a in ARMS),
          flush=True)
    by_arm = recs
    summary = {'n_files': sum(len(v) for v in recs.values()), 'arms': {}}
    for a in ARMS:
        rs = by_arm[a]
        ok = [r for r in rs if r.get('status') == 'ok']
        timeout = [r for r in rs if r.get('status') == 'timeout']
        err = [r for r in rs if r.get('status') == 'error']
        succ = [r for r in ok if r.get('success')]
        times, events = surv_data(rs)
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
        tag_tot = {}
        for r in succ:
            for t, c in r.get('provenance', {}).get('tag_counts', {}).items():
                tag_tot[t] = tag_tot.get(t, 0) + c
        summary['arms'][a]['winning_path_tag_counts'] = tag_tot
        summary['arms'][a]['n_successful_paths'] = len(succ)
        # graph diagnostics (E5 arms only): edges per run, edge feasibility
        if a in ('g', 'gr'):
            feas = []
            for r in ok:
                for er in r.get('edges', []):
                    feas.append((er['src'], er['dst'], bool(er['feasible'])))
            from collections import Counter
            c = Counter(feas)
            summary['arms'][a]['edge_feasibility'] = {
                f'{s}->{d}': {'attempts': sum(1 for x in feas
                                             if x[0] == s and x[1] == d),
                              'feasible': sum(1 for x in feas
                                              if x[0] == s and x[1] == d and x[2])}
                for s, d in set((x[0], x[1]) for x in feas)}
            summary['arms'][a]['mean_edges_per_run'] = (
                float(np.mean([r.get('n_edges', 0) for r in ok])) if ok else None)

    pairs = [('g', 'f'), ('gr', 'f'), ('g', 'gr')]
    summary['logrank'] = {}
    summary['fisher'] = {}
    arm_data = {a: surv_data(by_arm[a]) for a in ARMS}
    for x, y in pairs:
        lr = logrank(*arm_data[x], *arm_data[y])
        summary['logrank'][f'{x}_vs_{y}'] = lr
        kx, nx = summary['arms'][x]['n_success'], summary['arms'][x]['n_runs']
        ky, ny = summary['arms'][y]['n_success'], summary['arms'][y]['n_runs']
        _, p = fisher_exact([[kx, nx - kx], [ky, ny - ky]])
        summary['fisher'][f'{x}_vs_{y}'] = {'p': float(p)}

    def beats(x, y):
        lr = summary['logrank'][f'{x}_vs_{y}']
        return lr['p'] < 0.05 and lr['O1-E1'] > 0

    g_beats_f = beats('g', 'f')
    summary['falsifier'] = {
        'g_beats_f_logrank_p05': bool(g_beats_f),
        'verdict': ('graph search adds EFFICIENCY value (G beats F, p<0.05)'
                    if g_beats_f
                    else 'FALSIFIED on efficiency: graph search (G) is NOT '
                         'more successful per evaluation than flat search '
                         '(F) -> the graph contributes INTERPRETABILITY '
                         'ONLY, not efficiency'),
    }

    with open(os.path.join(WORKDIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n=== per-arm results (n=20/arm) ===')
    print(f"{'arm':28} {'succ':>5} {'rate':>6} {'95%CI':>16} "
          f"{'KMmed':>7} {'med|succ':>8} {'ok/timeout/err':>14}")
    for a in ARMS:
        s = summary['arms'][a]
        lo, hi = s['success_rate_wilson95']
        print(f"{ARM_LABEL[a]:28} {s['n_success']:>5} "
              f"{s['success_rate']:>6.2f} [{lo:.2f},{hi:.2f}] "
              f"{str(s['km_median_evals_to_handover']):>7} "
              f"{str(s['median_evals_among_successes']):>8} "
              f"{s['n_ok']:>2}/{s['n_timeout']:>2}/{s['n_error']:>2}")
    print('\n=== log-rank (O1-E1>0 favours first arm) ===')
    for k, v in summary['logrank'].items():
        print(f"{k:10} chi2={v['chi2']:.3f} p={v['p']:.4f} O1-E1={v['O1-E1']:+.2f}")
    print('\n=== Fisher exact on success counts ===')
    for k, v in summary['fisher'].items():
        print(f"{k:10} p={v['p']:.4f}")
    print('\n=== winning-path provenance (tag counts over successful runs) ===')
    for a in ARMS:
        s = summary['arms'][a]
        print(f"{ARM_LABEL[a]:28} paths={s['n_successful_paths']} "
              f"{s['winning_path_tag_counts']}")
    print('\n=== edge feasibility (E5 arms) ===')
    for a in ('g', 'gr'):
        s = summary['arms'][a]
        print(f"{ARM_LABEL[a]:28} mean_edges/run={s['mean_edges_per_run']}")
        for e, d in sorted(s.get('edge_feasibility', {}).items()):
            print(f"    {e:16} attempts={d['attempts']:>3} feasible={d['feasible']:>3}")
    print('\n=== FALSIFIER ===')
    print(summary['falsifier']['verdict'])
    print('\nsaved summary.json', flush=True)


if __name__ == '__main__':
    main()
