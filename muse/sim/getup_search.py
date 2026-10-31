"""Autonomous, interpretable get-up STRATEGY SEARCH (sequencer layer): A* over the contact-mode
graph of getup_modes.py, with getup_oracle.py as the per-edge feasibility oracle.

    getup_search.py   (THIS FILE)   A* over the mode graph; sees ONLY EdgeResult records
    getup_oracle.py   (interface)   edge (mode A -> mode B) -> region-to-target sub-goals
    getup_mpc.py      (workhorse)   general region-to-target "imagine" QP solver (unchanged)

No mode sequence is supplied by hand: the only inputs are the mode definitions + edge rules
(getup_modes), the start configuration (F3 q_supine_settled) and the goal mode STAND.

COST_DOC (documented edge cost; every term is logged per edge):
    c(A->B) = C_SWITCH                                  1.0   per contact-mode switch
            + W_EFFORT * sum_ticks mean_j (tau_j/tau_max_j)^2   effort along the certified path
                                                        (tau = exact statics-LP minimiser at every
                                                         executed quasi-static keyframe)
            + W_MARGIN * peak_rho / RHO_CAP             feasibility margin: how close the edge
                                                        came to the torque cap (0.8)
            + W_LEN    * ticks / 100                    path length (number of quasi-static steps)
    weights: C_SWITCH = W_EFFORT = W_MARGIN = 1, W_LEN = 0.5 (getup_oracle constants)
    heuristic h(A) = C_SWITCH * hops(A -> STAND) over CANDIDATE edges (getup_modes.hops_to_goal);
    admissible because every edge costs >= C_SWITCH and hops is a lower bound on switches.
    Infeasible edges are not traversable (cost = inf) and keep their diagnosis.

Configuration-lazy A*: a mode node carries ONE representative configuration (the end keyframe of
the best edge that reached it so far); outgoing edges are evaluated at expansion time from that
representative.  Optimality is therefore w.r.t. the evaluated representatives (documented
limitation), and an INFEASIBLE edge means "the oracle found no quasi-static path from this
representative", with the binding diagnosis attached.

Terminal verdict, exactly one of:
    STRATEGY_FOUND                     mode sequence + per-transition keyframes + attention/cost
    QUASI_STATIC_EXHAUSTION            open list empty; every reachable mode expanded; every
                                       frontier transition carries a per-edge diagnosis
                                       (pruned by rule, or oracle-infeasible with binding)

Usage:  python getup_search.py [OUTDIR]       (default ../results/getup_search)
Deterministic: no randomness; caller forces single-threaded BLAS (OMP/OPENBLAS/MKL=1).
"""
from __future__ import annotations

import heapq
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from getup_modes import MODES, MODE, START, GOAL, edge_rules, hops_to_goal, render_markdown, classify
from getup_oracle import OracleContext, evaluate_edge, certify, C_SWITCH, cert_summary

HERE = os.path.dirname(os.path.abspath(__file__))


def jl(x):
    if isinstance(x, dict):
        return {str(k): jl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jl(v) for v in x]
    if isinstance(x, np.ndarray):
        return jl(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def edge_record(r):
    d = dict(src=r.src, dst=r.dst, status=r.status, rule=r.rule, reason=r.reason, cost=r.cost,
             subgoals=r.subgoals, diagnosis=r.diagnosis, ticks=r.ticks, wall_s=r.wall_s,
             keyframes=[dict(name=k["name"], cert=k["cert"], z=float(k["q"][2])) for k in r.keyframes])
    return jl(d)


def search(outdir):
    os.makedirs(outdir, exist_ok=True)
    trace_path = os.path.join(outdir, "trace.jsonl")
    tf = open(trace_path, "w")

    def log(ev):
        tf.write(json.dumps(jl(ev)) + "\n"); tf.flush()
        print("[trace]", ev.get("event"), {k: v for k, v in ev.items() if k in ("node", "src", "dst", "status",
                                                                             "rule", "f", "g", "h", "total")},
              flush=True)

    with open(os.path.join(outdir, "graph.md"), "w") as f:
        f.write(render_markdown())
    ctx = OracleContext()
    S = ctx.S
    q0 = np.load(os.path.join(HERE, "..", "results", "f3", "q_supine_settled.npy"))
    c0 = certify(S, q0, ctx.p.mu)
    cls = classify(S, S.kin(q0, with_G=False, with_pairs=False), c0)
    h = hops_to_goal()
    H = lambda m: C_SWITCH * h.get(m, np.inf)
    log(dict(event="start", node=START, start_modes=cls, cert=cert_summary(c0), heuristic=h))
    rules = edge_rules()
    out_of = {m.key: [e for e in rules if e.src == m.key] for m in MODES}

    g = {START: 0.0}
    rep = {START: q0}
    parent = {START: None}
    edge_of = {}                     # dst -> EdgeResult that produced its representative
    all_edges = []                   # every EdgeResult (feasible / infeasible / pruned)
    closed = []
    openh = [(H(START), 0, START)]
    tie = 1
    t0 = time.time()
    found = False
    while openh:
        f_, _, a = heapq.heappop(openh)
        if a in closed:
            continue
        closed.append(a)
        log(dict(event="expand", node=a, g=g[a], h=H(a), f=f_, order=len(closed)))
        if a == GOAL:
            found = True
            break
        for e in out_of[a]:
            b = e.dst
            if e.status == "pruned":
                r = evaluate_edge(ctx, rep[a], a, b)          # returns PRUNED with rule + reason
                all_edges.append(r)
                if e.rule not in ("P-START",):
                    log(dict(event="prune", src=a, dst=b, rule=r.rule, reason=r.reason))
                continue
            if b in closed:
                log(dict(event="skip_closed", src=a, dst=b, reason="destination already expanded (A* closed set)"))
                continue
            print(f"== oracle {a} -> {b}", flush=True)
            r = evaluate_edge(ctx, rep[a], a, b)
            all_edges.append(r)
            log(dict(event="edge", **edge_record(r)))
            if r.status != "FEASIBLE":
                continue
            gb = g[a] + r.cost["total"]
            if gb < g.get(b, np.inf):
                g[b] = gb; rep[b] = r.q_end; parent[b] = a; edge_of[b] = r
                heapq.heappush(openh, (gb + H(b), tie, b)); tie += 1
                log(dict(event="relax", src=a, dst=b, g=gb, h=H(b), f=gb + H(b)))
    wall = time.time() - t0

    summary = dict(wall_clock_s=round(wall, 1), expanded=closed,
                   edges=[edge_record(r) for r in all_edges])
    if found:
        seq = [GOAL]
        while parent[seq[-1]] is not None:
            seq.append(parent[seq[-1]])
        seq = seq[::-1]
        summary.update(verdict="STRATEGY_FOUND", sequence=seq, total_cost=g[GOAL],
                       plan=[edge_record(edge_of[b]) for b in seq[1:]])
        np.savez(os.path.join(outdir, "plan_keyframes.npz"),
                 names=np.array([k["name"] for b in seq[1:] for k in edge_of[b].keyframes]),
                 q=np.array([k["q"] for b in seq[1:] for k in edge_of[b].keyframes]))
    else:
        frontier = [edge_record(r) for r in all_edges if r.src in closed and r.status != "FEASIBLE"
                    and r.rule not in ("P-START", "P-GOAL")]
        summary.update(verdict="QUASI_STATIC_EXHAUSTION", reachable=closed,
                       unreached=[m.key for m in MODES if m.key not in closed], frontier=frontier)
    np.savez(os.path.join(outdir, "representatives.npz"), **{k: v for k, v in rep.items()})
    log(dict(event="verdict", status=summary["verdict"], expanded=closed, wall=wall))
    tf.close()
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(jl(summary), f, indent=1)
    with open(os.path.join(outdir, "verdict.md"), "w") as f:
        f.write(render_verdict(summary))
    return summary


def render_verdict(s):
    L = [f"# Get-up strategy search verdict: **{s['verdict']}**", "",
         f"Expanded (in order): {' -> '.join(s['expanded'])}; wall clock {s['wall_clock_s']} s.", ""]
    if s["verdict"] == "STRATEGY_FOUND":
        L += [f"**Mode sequence:** {' -> '.join(s['sequence'])}  (total cost {s['total_cost']:.3f})", ""]
        for e in s["plan"]:
            c = e["cost"]
            L += [f"## {e['src']} -> {e['dst']}", "",
                  f"cost {c['total']} = switch {c['switch']} + effort {c['effort']} + margin {c['margin']} "
                  f"(peak rho {c['peak_rho']}) + length {c['length']} ({c['ticks']} ticks)", ""]
            for sg in e["subgoals"]:
                L.append(f"- **{sg['name']}**: {sg['outcome']}, {sg.get('ticks', 0)} ticks")
                at = sg.get("attention")
                if at:
                    L.append(f"  - support attention top: {at.get('support_top')}")
                    L.append(f"  - motion attention top: {at.get('motion_top')}")
                    L.append(f"  - effort bottleneck: {at.get('effort_bottleneck')}; landing sub-tasks "
                             f"{at.get('landing_subtasks')}")
            for k in e["keyframes"]:
                L.append(f"- keyframe `{k['name']}`: base z {k['z']:.3f} m, cert {k['cert']}")
            L.append("")
    else:
        L += [f"Reachable modes: {s['reachable']}; never reached: {s['unreached']}", "",
              "## Frontier transitions (every one diagnosed)", "",
              "| edge | status | rule / failing sub-goal | diagnosis |", "|---|---|---|---|"]
        for e in s["frontier"]:
            if e["status"] == "PRUNED":
                L.append(f"| {e['src']} -> {e['dst']} | PRUNED | {e['rule']} | {e['reason']} |")
            else:
                d = e["diagnosis"]
                L.append(f"| {e['src']} -> {e['dst']} | INFEASIBLE | {d.get('subgoal')} {d.get('outcome')} | "
                         f"{d.get('verdict', e['reason'])}; residual {d.get('residual_task_terms')}; "
                         f"unmet {d.get('unmet_mode_predicates')} |")
    L += ["", "## All evaluated (non-pruned) edges", "", "| edge | status | cost | ticks | reason |",
          "|---|---|---|---|---|"]
    for e in s["edges"]:
        if e["status"] != "PRUNED":
            L.append(f"| {e['src']} -> {e['dst']} | {e['status']} | {e['cost'].get('total')} | {e['ticks']} | "
                     f"{e['reason'] or '-'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "getup_search")
    s = search(out)
    print("VERDICT:", s["verdict"])
