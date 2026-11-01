"""Generate outcome tables from completed search and replay evidence only."""
import json
import argparse
from collections import Counter
from .plant import ROOT
from .explain import transition_label


def summarize(selection="dynamic"):
    out = ROOT/'combo/results'
    if selection != "robust":
        out = out/selection
    lines = ['# Measured outcomes', '',
             '| Start | Seed | Search/QP success | Replay success | Evals | Blocks | Wall s |',
             '|---|---:|---|---|---:|---:|---:|']
    runs = []
    for pose in ('supine','prone'):
        for seed in (0,1,2):
            path = out/f'{pose}_{seed}.json'
            if not path.exists():
                lines.append(f'| {pose} | {seed} | PENDING | — | — | — | — |')
                continue
            r = json.loads(path.read_text())
            ep = out/f'{pose}_{seed}_explanation.json'
            e = json.loads(ep.read_text()) if ep.exists() else None
            runs.append((r,e))
            lines.append(f'| {pose} | {seed} | {r["success"]} | {e["replay_success"] if e else "PENDING"} | '
                         f'{r["evaluations"]} | {r["action_blocks"]} | {r["seconds"]:.1f} |')
    for pose in ('supine','prone'):
        rr = [r for r,e in runs if r['pose']==pose]
        lines += ['', f'{pose}: {sum(r["success"] for r in rr)}/{len(rr)} completed search runs succeeded.']
    for r,e in runs:
        lines += ['', f'## {r["pose"]} / {r["seed"]}', '',
                  f'Initial passive settle: {r["settle"]["settled"]}; QP attempts: {len(r["handovers"])}.']
        if not r['success']:
            f = r['final']
            gate = dict(base_height=(f['base_height'], .16, '>'), up_z=(f['up_z'], .9, '>'),
                        foot_load=(f['foot_load'],75,'>'), nonfoot_load=(f['nonfoot_load'],8,'<'))
            failed = [f'{key}={v:.4f} (requires {op}{threshold})' for key,(v,threshold,op) in gate.items()
                      if not (v>threshold if op=='>' else v<threshold)]
            lines += ['Budget exhausted. Final retained state misses handover gates: '+('; '.join(failed) or 'none; QP rejected entry')+'.',
                      'These are measured algorithmic stopping constraints, not a proof of physical impossibility.']
        if not r['success'] and r['handovers']:
            h = r['handovers'][-1]
            lines += [f'Last QP attempt ended at base={h["final"]["base_height"]:.3f} m, '
                      f'up_z={h["final"]["up_z"]:.3f}, nonfoot load={h["final"]["nonfoot_load"]:.1f} N '
                      f'after about {h["seconds"]:.2f} s.']
        if e is None:
            continue
        lines += ['', 'Sampled mode sequence: '+' → '.join(e['mode_sequence'])+'.',
                  f'Replay full-snapshot maximum error: {e["replay_snapshot_max_error"]:.3g}.', '',
                  '| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |',
                  '|---|---|---|---|---|---|']
        for t in e['transitions']:
            c = t['samples'][-1]['certificate']
            rho = 'infeasible' if c['rho'] is None else f'{c["rho"]:.3f}'
            loads = ', '.join(f'{k}:{v:.2f}' for k,v in c['loads'].items() if v>.01)
            lines.append(f'| {t["block"]} | {t["source"]} → {t["destination"]} | {t["label"]} | '
                         f'{c["ok"]} | {rho} / {c["binding"]} | {loads} |')
        lines += ['', '| Mode edge (sampled interval) | Label | Destination certified |',
                  '|---|---|---|']
        for t in e.get('mode_transitions', []):
            lines.append(f'| {t["source"]} → {t["destination"]} ({t["start_time"]:.2f}–{t["end_time"]:.2f} s) | '
                         f'{t["label"]} | {t["endpoint_certificate"]["ok"]} |')
        witnesses = e.get('graph_witnesses', [])
        if witnesses:
            lines += ['', f'{len(witnesses)} distinct graph-edge witnesses replayed and explained; '
                      f'{sum(t["destination_certificate"]["ok"] for t in witnesses)} certified destination configurations.']
        samples = [s for t in e['transitions'] for s in t['samples']]
        counts = Counter(t['label'] for t in e['transitions'])
        terminal = e['transitions'][-1]['samples']
        tail = [s for s in terminal if s['time'] >= terminal[-1]['time']-1.]
        lines += ['', f'Terminal sampled tail: {transition_label(tail)["label"]}, '
                  f'span {tail[-1]["time"]-tail[0]["time"]:.2f} s, '
                  f'peak KE {max(s["metrics"]["kinetic_j"] for s in tail):.6f} J.']
        lines += ['', 'Transition labels: '+', '.join(f'{k}={v}' for k,v in counts.items())+'.',
                  f'Sampled peak KE: {max(s["metrics"]["kinetic_j"] for s in samples):.3f} J; '
                  f'minimum shell height: {1000*min(s["certificate"]["pen"] for s in samples):.2f} mm; '
                  f'minimum self distance: {1000*min(s["certificate"]["selfd"] for s in samples):.2f} mm.',
                  'Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.']
    (ROOT/'combo'/('RESULTS.md' if selection == 'robust' else 'RESULTS_'+selection+'.md')).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection', choices=['robust','dynamic'], default='dynamic')
    summarize(p.parse_args().selection)
