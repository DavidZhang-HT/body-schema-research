"""Explicit, inspectable CONTACT-MODE GRAPH for autonomous get-up strategy search.

Layer 3 of the get-up stack (see plan.md "Autonomous interpretable get-up strategy search"):

    getup_search.py   (sequencer)   A* over THIS graph; sees only EdgeResult records
    getup_oracle.py   (interface)   edge (mode A -> mode B) -> region-to-target sub-goals
    getup_mpc.py      (workhorse)   general region-to-target "imagine" QP solver (unchanged)

Nothing here calls a solver.  A mode is a human-readable predicate over the body schema
(getup_schema REGIONS, built from the G0 primitive contact shell in model.py):

    * req      : list of "any-of" region sets; each set must have at least one region touching
                 (region lowest point <= TOUCH_EPS above the floor)
    * allowed  : regions that MAY touch in addition (every other region must be off the floor)
    * orient   : trunk-orientation predicate on the base rotation columns
                 chest = R[:,0] (body +x), left = R[:,1] (body +y), head = R[:,2] (body +z)
    * extra    : optional predicate on base height
    * statics  : always: the touching set must carry the body quasi-statically
                 (Schema.statics_lp feasible, peak |tau|/tau_max <= RHO_CAP)

Candidate edges are generated from the node definitions by explicit rules, and every
ordered pair that is NOT a candidate carries a stated pruning reason (rule id + text).
The F3 diagnosis is encoded as rule P-F3 with its evidence, never as a silent omission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from getup_schema import REGIONS

TOUCH_EPS = 0.005          # [m] a region "touches" if its lowest shell point is within 5 mm
RHO_CAP = 0.8              # F3 keyframe criterion (peak torque ratio)
Z_SQUAT_MAX = 0.26         # [m] base height upper bound for "deep squat"
Z_STAND_MIN = 0.32         # [m] = 0.85 * Z_STAND_NOM (0.376): standing (same as F3 P4)

ALL = frozenset(REGIONS)
TRUNK = frozenset({"upper_trunk", "lower_back", "pelvis"})
LIMBS_LOWER = frozenset({"L_shank", "R_shank", "L_foot", "R_foot"})

# trunk-orientation classes (used by the adjacency rule P-TOPO)
LYING_UP, LYING_SIDE, LYING_DOWN, PITCHED, UPRIGHT = "LYING_UP", "LYING_SIDE", "LYING_DOWN", "PITCHED", "UPRIGHT"
# support levels (used by the level rules P-SKIP / P-DESC)
#   0 = trunk on the floor, 1 = trunk off the floor carried by limbs / pelvis, 2 = feet only
ADJ_CLASSES = {frozenset(p) for p in [
    (LYING_UP, LYING_SIDE),      # roll a quarter turn
    (LYING_SIDE, LYING_DOWN),    # roll a quarter turn
    (LYING_DOWN, PITCHED),       # push the chest up (prone push-up)
    (LYING_SIDE, PITCHED),       # roll-and-push onto hands
    (LYING_SIDE, UPRIGHT),       # side push-up to side-sit
    (LYING_UP, UPRIGHT),         # sit-up (direct trunk lift) -- adjacent, but see P-F3
    (PITCHED, UPRIGHT),          # push the hips back / up over the feet
    (PITCHED,), (UPRIGHT,),      # within-class edges
]}


@dataclass(frozen=True)
class Mode:
    key: str
    name: str
    req: tuple                  # tuple of frozensets ("any-of" groups)
    allowed: frozenset          # regions that may additionally touch
    orient_txt: str
    orient: callable            # (R) -> bool
    oclass: str
    level: int
    extra_txt: str = ""
    extra: callable = None      # (qpos) -> bool
    meaning: str = ""

    def forbidden(self):
        need = set().union(*self.req)
        return sorted(ALL - self.allowed - need)


def _f(*r):
    return frozenset(r)


MODES = [
    Mode("SUPINE", "supine {back}",
         req=(_f("upper_trunk", "lower_back"),), allowed=ALL,
         orient_txt="chest_z >= 0.7 (chest up)", orient=lambda R: R[2, 0] >= 0.7,
         oclass=LYING_UP, level=0,
         meaning="lying on the back; limbs anywhere (start state, F3 q_supine_settled)"),
    Mode("SIDE", "side-lying {right side}",
         req=(_f("upper_trunk", "lower_back", "pelvis", "R_upper_arm"),), allowed=ALL,
         orient_txt="left_z >= 0.7 (body +y up, i.e. right side down)", orient=lambda R: R[2, 1] >= 0.7,
         oclass=LYING_SIDE, level=0,
         meaning="lying on the right flank (left mirror image omitted: model is L/R symmetric)"),
    Mode("PRONE", "prone {chest}",
         req=(_f("upper_trunk", "lower_back"),), allowed=ALL,
         orient_txt="chest_z <= -0.7 (chest down)", orient=lambda R: R[2, 0] <= -0.7,
         oclass=LYING_DOWN, level=0,
         meaning="lying on the chest; limbs anywhere"),
    Mode("SIT", "long-sit {pelvis}",
         req=(_f("pelvis"),),
         allowed=_f("L_thigh", "R_thigh", "L_knee", "R_knee", "L_shank", "R_shank", "L_foot", "R_foot",
                    "L_hand", "R_hand", "L_forearm", "R_forearm"),
         orient_txt="head_z >= 0.7 (trunk within 45 deg of vertical)", orient=lambda R: R[2, 2] >= 0.7,
         oclass=UPRIGHT, level=1,
         meaning="sitting on the butt, back off the floor (the F3 P1_head_up target)"),
    Mode("QUAD", "quadruped {hands+knees}",
         req=(_f("L_hand"), _f("R_hand"), _f("L_knee"), _f("R_knee")),
         allowed=LIMBS_LOWER,
         orient_txt="chest_z <= -0.5 (chest facing the floor)", orient=lambda R: R[2, 0] <= -0.5,
         oclass=PITCHED, level=1,
         meaning="on hands and knees (shins / foot backs may rest on the floor); trunk, pelvis, head off"),
    Mode("HF", "plantigrade {hands+feet}",
         req=(_f("L_hand"), _f("R_hand"), _f("L_foot"), _f("R_foot")), allowed=frozenset(),
         orient_txt="chest_z <= -0.2 (chest facing the floor, pitched)", orient=lambda R: R[2, 0] <= -0.2,
         oclass=PITCHED, level=1,
         meaning="on hands and feet only ('bear' / plantigrade)"),
    Mode("SQUAT", "deep squat {feet}",
         req=(_f("L_foot"), _f("R_foot")), allowed=frozenset(),
         orient_txt="head_z >= 0.5 (trunk within 60 deg of vertical)", orient=lambda R: R[2, 2] >= 0.5,
         oclass=UPRIGHT, level=2,
         extra_txt=f"base z <= {Z_SQUAT_MAX} m", extra=lambda q: q[2] <= Z_SQUAT_MAX,
         meaning="feet only, pelvis low"),
    Mode("STAND", "standing {feet}",
         req=(_f("L_foot"), _f("R_foot")), allowed=frozenset(),
         orient_txt="head_z >= 0.94 (tilt <= 20 deg)", orient=lambda R: R[2, 2] >= 0.94,
         oclass=UPRIGHT, level=2,
         extra_txt=f"base z >= {Z_STAND_MIN} m (0.85 Z_stand)", extra=lambda q: q[2] >= Z_STAND_MIN,
         meaning="goal: feet only, upright, base at >= 85 % standing height (F3 P4 criterion)"),
]
MODE = {m.key: m for m in MODES}
START, GOAL = "SUPINE", "STAND"

# ----------------------------------------------------------------------------- F3 seeding
F3_EVIDENCE = ("F3 run_a (results/f3/run_a.json): direct supine trunk lift STALLED at P1_head_up, tick 121; "
               "trunk_upright cost stuck at 2.84 (tilt 11-17 deg); peak rho 0.181 << 0.8 (NOT torque-bound); "
               "relaxing self-collision / joint limits / trust region / rho_cap->inf / mu x2 all failed to "
               "unstick (getup_f3_diag.py) -> binding = trunk-flexion work unavailable quasi-statically "
               "while the back is the support (structural, not a tuning issue)")


@dataclass
class EdgeRule:
    src: str
    dst: str
    status: str                 # "candidate" | "pruned"
    rule: str = ""
    reason: str = ""


def edge_rules():
    """Every ordered pair (A, B), A != B, with an explicit status.  Rules, in order:

    P-START  B is the start node: never useful (start is closed at expansion 0)
    P-GOAL   A is the goal: search terminates there
    P-DESC   level(B) < level(A): descent edges are outside this (level-monotone) search scope
    P-TOPO   orientation classes not adjacent: a continuous trunk rotation must pass through an
             intermediate class -> not an elementary edge (it decomposes into graph edges)
    P-F3     LYING_UP -> UPRIGHT (direct trunk lift from supine): F3 structural failure
    P-SKIP   level jump 0 -> 2: lifting the whole trunk from the floor straight onto the feet
             with no limb/pelvis support level in between is the same trunk-lift sub-problem F3
             diagnosed (no support group shared; must decompose through a level-1 mode)
    """
    out = []
    for a in MODES:
        for b in MODES:
            if a.key == b.key:
                continue
            e = EdgeRule(a.key, b.key, "candidate")
            if b.key == START:
                e.status, e.rule, e.reason = "pruned", "P-START", "target is the start node"
            elif a.key == GOAL:
                e.status, e.rule, e.reason = "pruned", "P-GOAL", "source is the goal (search stops)"
            elif b.level < a.level:
                e.status, e.rule, e.reason = "pruned", "P-DESC", (
                    f"support level descends {a.level}->{b.level}; scope = level-monotone search "
                    "(documented incompleteness)")
            elif frozenset((a.oclass, b.oclass)) not in ADJ_CLASSES and a.oclass != b.oclass:
                e.status, e.rule, e.reason = "pruned", "P-TOPO", (
                    f"orientation {a.oclass} -> {b.oclass} not adjacent: rotation must pass through an "
                    "intermediate class (not elementary)")
            elif a.oclass == LYING_UP and b.oclass == UPRIGHT:
                e.status, e.rule, e.reason = "pruned", "P-F3", F3_EVIDENCE
            elif a.level == 0 and b.level == 2:
                e.status, e.rule, e.reason = "pruned", "P-SKIP", (
                    "level 0 -> 2: trunk lifted from the floor directly onto the feet with no shared "
                    "support group = the F3 trunk-lift sub-problem; must pass through a level-1 mode")
            out.append(e)
    return out


def candidates():
    return {(e.src, e.dst) for e in edge_rules() if e.status == "candidate"}


def hops_to_goal():
    """BFS hop count to GOAL over candidate edges (admissible A* heuristic x c_switch)."""
    cand = candidates()
    h = {GOAL: 0}
    frontier = [GOAL]
    while frontier:
        nxt = []
        for b in frontier:
            for (a, bb) in cand:
                if bb == b and a not in h:
                    h[a] = h[b] + 1
                    nxt.append(a)
        frontier = nxt
    return h


# ----------------------------------------------------------------------------- classifier
def mode_check(S, k, cert, key):
    """Explain membership of configuration k in mode `key`: returns (ok, reasons list)."""
    m = MODE[key]
    d = S.region_height(k)
    touch = {r for r, v in zip(REGIONS, d) if v <= TOUCH_EPS}
    why = []
    for grp in m.req:
        if not (touch & grp):
            why.append(f"req {sorted(grp)} not touching (min h {1e3 * min(d[REGIONS.index(r)] for r in grp):.0f} mm)")
    bad = touch - m.allowed - set().union(*m.req)
    if bad:
        why.append(f"forbidden touching {sorted(bad)}")
    if not m.orient(k.R):
        why.append(f"orientation fails [{m.orient_txt}] (chest_z {k.R[2, 0]:+.2f} left_z {k.R[2, 1]:+.2f} "
                   f"head_z {k.R[2, 2]:+.2f})")
    if m.extra is not None and not m.extra(k.qpos):
        why.append(f"extra fails [{m.extra_txt}] (base z {k.qpos[2]:.3f})")
    if not (cert["feasible"] and cert["rho"] <= RHO_CAP):
        why.append(f"statics: feasible={cert['feasible']} rho={cert['rho']:.3f}")
    return len(why) == 0, why


def classify(S, k, cert):
    return [m.key for m in MODES if mode_check(S, k, cert, m.key)[0]]


# ----------------------------------------------------------------------------- docs
def render_markdown():
    L = ["# Contact-mode graph (generated by sim/getup_modes.py)", "",
         f"Touch threshold {TOUCH_EPS * 1e3:.0f} mm; every mode additionally requires quasi-static "
         f"feasibility of the touching set (statics LP, peak |tau|/tau_max <= {RHO_CAP}). "
         "Regions are the G0 shell regions of `getup_schema.REGIONS`.", "",
         "## Nodes", "",
         "| key | mode | required touching (each any-of group) | may also touch | must be off the floor | "
         "trunk orientation | extra | class / level | meaning |", "|---|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        req = "; ".join("{" + ",".join(sorted(g)) + "}" for g in m.req)
        alw = "ALL" if m.allowed == ALL else ("-" if not m.allowed else ",".join(sorted(m.allowed)))
        forb = ",".join(m.forbidden()) or "-"
        L.append(f"| {m.key} | {m.name} | {req} | {alw} | {forb} | {m.orient_txt} | {m.extra_txt or '-'} | "
                 f"{m.oclass} / {m.level} | {m.meaning} |")
    L += ["", "## Edges (all ordered pairs; rules in `edge_rules()` docstring)", "",
          "| src -> dst | status | rule | reason |", "|---|---|---|---|"]
    for e in edge_rules():
        if e.rule in ("P-START", "P-GOAL"):
            continue
        L.append(f"| {e.src} -> {e.dst} | {e.status} | {e.rule or '-'} | {e.reason or '-'} |")
    n_st = sum(1 for e in edge_rules() if e.rule in ("P-START", "P-GOAL"))
    L += ["", f"({n_st} trivial pairs pruned by P-START / P-GOAL omitted from the table.)", "",
          "Heuristic hop counts to STAND over candidate edges: " +
          ", ".join(f"{k}={v}" for k, v in sorted(hops_to_goal().items(), key=lambda x: x[1]))]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(render_markdown())
