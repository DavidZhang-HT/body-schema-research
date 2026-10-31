"""Feasibility ORACLE for one contact-mode transition (interface layer of the get-up stack).

    getup_search.py   (sequencer)   A* over getup_modes; sees ONLY EdgeResult records
    getup_oracle.py   (THIS FILE)   edge (mode A -> mode B)  ->  region-to-target sub-goals
    getup_mpc.py      (workhorse)   general region-to-target "imagine" QP solver (UNCHANGED)

Interface contract (documented in plan.md, "Autonomous interpretable get-up strategy search"):

  evaluate_edge(ctx, q_src, src, dst) -> EdgeResult
     input : a certified configuration q_src that satisfies mode `src` (getup_modes.mode_check)
     output: status  FEASIBLE | INFEASIBLE | PRUNED (rule id + reason, never evaluated)
             keyframes  (one per executed sub-goal, each a certified static equilibrium)
             cost breakdown (getup_search.COST doc), attention trace per sub-goal,
             diagnosis on failure (which sub-goal, outcome, residual task terms, unmet mode
             predicates, QP binding families / elastic IIS, constraint-family relaxation ladder)

  The oracle never chooses a mode sequence.  The solver never sees modes, only Phase objects
  (lists of region-to-target Task terms + a done predicate) -- exactly the interface F3 used.

Compilation rule  (mode B definition  ->  sub-goals; identical for every edge, no per-edge script):

  G1 MAKE     (only if some required any-of group of B is not touching)
              land the lowest member of every missing group   [t_region_down]
              keep the trunk orientation of the start          [axis task, low weight]
              done: every required group touches (<= TOUCH_EPS)
  G2 ORIENT   (only if B's orientation / height predicate fails)
              rotate the trunk axis named by B onto B's target [axis task]
              keep required groups on the floor                 [t_region_down]
              level >= 1 : CoM ground projection over the required support centroid [t_com]
              level 0    : CoM ground projection at least as far as the IMAGINED PIVOT edge along
                           the roll direction u (1-D; pivot = floor-near TRUNK shell point furthest
                           along u; u = -a_h if the named axis must rise, +a_h if it must fall;
                           a_h = its horizontal projection) [com_over_pivot]  -> far-side
                           supports unload and are released before the trunk rotates
              feet required : soles flat                        [t_foot_flat]
              STAND only    : CoM height 0.30 m + nominal leg posture (the F3 P4 term set)
              done: orientation + extra predicates hold and required groups still touch
  G3 RELEASE  (only if B's full predicate still fails)
              lift every touching region B forbids              [t_region_up, 2 cm]
              keep CoM over the required support, orientation, required groups down
              done: getup_modes.mode_check(B) == True (incl. statics LP, rho <= 0.8)
  "make before break": new support first, then re-orient / shift load, then release.
  Which contacts actually carry load and which joints move inside each sub-goal is decided by
  the interpretable attention heads in getup_mpc / getup_attention (unchanged).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from getup_schema import Schema, REGIONS
from getup_attention import SchemaAttention
from getup_mpc import (QSMPC, MPCParams, Phase, Task, run_phase, certify, rot_err_axis,
                       t_com, t_region_down, t_region_up, t_foot_flat, t_posture, stand_ref)
from getup_modes import MODE, TOUCH_EPS, RHO_CAP, edge_rules, mode_check
from getup_keyframes import KeyframeImaginer
from model import JOINTS

# --------------------------------------------------------------------------- orientation targets
# (axis column of the base rotation, target elevation [deg]); heading = current horizontal
# heading of that axis.  Targets sit INSIDE each mode's orientation predicate (margin).
ORIENT_TARGET = {
    "SUPINE": (0, 90.0),     # chest up
    "SIDE":   (1, 90.0),     # body +y up (right side down)
    "PRONE":  (0, -90.0),    # chest down
    "SIT":    (2, 90.0),     # head up
    "QUAD":   (0, -90.0),    # chest down (trunk horizontal)
    "HF":     (2, 45.0),     # head axis 45 deg above horizontal (chest pitched down)
    "SQUAT":  (2, 75.0),
    "STAND":  (2, 90.0),
}
AXIS_NAME = {0: "chest(+x)", 1: "left(+y)", 2: "head(+z)"}
SUBGOAL_TICKS = {"G1_make": 300, "G2_orient": 400, "G3_release": 300}
STALL_TICKS, STALL_TOL = 60, 0.01
MAX_RESTARTS, RESTART_GAIN = 4, 0.02
TRACK_DEG, TRACK_Q, TRACK_TICKS = 5.0, 0.2, 250

RELAX_LADDER = {             # constraint-family relaxation ladder (as getup_f3_diag.py)
    "no_selfcol": dict(use_selfcol=False),
    "no_jlim": dict(use_jlim=False),
    "big_trust": dict(tr_joint=0.1, tr_base_lin=0.03, tr_base_ang=0.1),
    "rho_cap_inf": dict(rho_cap=1e3),
    "mu_x2": dict(mu=1.2),
    "no_hold_ramp": dict(F_hold=1e9, F_s=1e9),   # contact state machine off: loaded points may
                                                 # lift / slide, load may jump between points
}
LADDER_TICKS = 15
UNLOCK_FRAC = 0.25           # a relaxation "unlocks" if it removes >= 25 % more task cost than full


def axis_target(k, col, elev_deg):
    a = k.R[:, col]
    h = np.array([a[0], a[1], 0.0])
    n = np.linalg.norm(h)
    h = h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    e = np.radians(elev_deg)
    return np.cos(e) * h + np.sin(e) * np.array([0, 0, 1.0])


def t_axis(k, col, target, w):
    """Region(trunk)-orientation-to-target: rotate base axis `col` onto `target`."""
    e = rot_err_axis(k.R[:, col], target)
    return Task(f"trunk_{AXIS_NAME[col]}", -e, k.Jrot, w)


# --------------------------------------------------------------------------- records
@dataclass
class EdgeResult:
    src: str
    dst: str
    status: str                         # FEASIBLE | INFEASIBLE | PRUNED
    rule: str = ""                      # pruning rule id (PRUNED only)
    reason: str = ""                    # pruning reason / one-line failure reason
    cost: dict = field(default_factory=dict)
    subgoals: list = field(default_factory=list)   # per sub-goal: outcome, ticks, attention, tasks
    keyframes: list = field(default_factory=list)  # (name, q, cert summary)
    diagnosis: dict = field(default_factory=dict)
    q_end: np.ndarray | None = None
    ticks: int = 0
    wall_s: float = 0.0


class OracleContext:
    def __init__(self):
        self.S = Schema()
        self.A = SchemaAttention(self.S)
        self.p = MPCParams()
        self.rules = {(e.src, e.dst): e for e in edge_rules()}


# --------------------------------------------------------------------------- compilation
def compile_subgoals(S, dst, q_src):
    """Mode B -> ordered list of Phase objects (region-to-target sub-goals), by the rule above."""
    m = MODE[dst]
    col, elev = ORIENT_TARGET[dst]
    k0 = S.kin(q_src, with_G=False, with_pairs=False)
    tgt0 = axis_target(k0, col, elev)          # fixed at edge start (heading frozen)
    keep_col = col
    keep_tgt = k0.R[:, col].copy()             # start orientation of that axis
    need = set().union(*m.req)
    qstand = stand_ref(S)
    feet_req = any(g & {"L_foot", "R_foot"} for g in m.req)

    def touching(k):
        d = S.region_height(k)
        return {r for r, v in zip(REGIONS, d) if v <= TOUCH_EPS}

    def missing_groups(k):
        t = touching(k)
        return [g for g in m.req if not (t & g)]

    def lowest(k, grp):
        d = S.region_height(k)
        return min(grp, key=lambda r: d[REGIONS.index(r)])

    def support_centroid(k):
        xy = S.region_xy(k)
        pts = [xy[REGIONS.index(lowest(k, g))] for g in m.req]
        c = np.mean(pts, 0)
        return np.array([c[0], c[1], 0.0])

    def keep_down(k, w=10.0):
        return [t_region_down(S, k, lowest(k, g), w) for g in m.req]

    # imagined pivot (level-0 targets = rotations with the trunk on the floor): if the named axis
    # must RISE, the body rolls over the shell edge on the opposite side (-a_h); if it must FALL,
    # over the edge on its own side (+a_h).  CoM ground projection is driven onto that edge so the
    # far-side contacts unload (held -> released by the MPC state machine) before rotating.
    rise = elev > np.degrees(np.arcsin(np.clip(k0.R[2, col], -1, 1)))
    roll_pts = np.concatenate([S.region_points[r] for r in ("upper_trunk", "lower_back", "pelvis")])

    def pivot(k):
        """-> (pivot point xy, unit roll direction u) or None."""
        a = k.R[:, col]
        u = np.array([a[0], a[1]])
        n = np.linalg.norm(u)
        if n < 1e-6:
            return None
        u = (-u if rise else u) / n
        idx = [i for i in roll_pts if k.z[i] <= 0.03] or list(roll_pts)
        proj = k.xy[idx] @ u
        return k.xy[idx[int(np.argmax(proj))]] - 0.01 * u, u

    def t_com_over_pivot(k, w, scale=10.0):
        """1-D: CoM ground projection at least as far along u as the pivot edge (fore-aft free)."""
        pv, u = pivot(k)
        r = min(0.0, float(u @ k.com[:2] - u @ pv))
        return Task("com_over_pivot", np.array([scale * r]), scale * (u @ k.Jcom[:2])[None, :], w)

    def common(k):
        ts = []
        if m.level >= 1:
            ts.append(t_com(k, 30.0, support_centroid(k), axes=(0, 1)))
        elif pivot(k) is not None:
            ts.append(t_com_over_pivot(k, 30.0))
        if feet_req:
            ts.append(t_foot_flat(k, 5.0))
        return ts

    def orient_ok(k, cert):
        return bool(m.orient(k.R)) and (m.extra is None or bool(m.extra(k.qpos)))

    def cstar(S_, k):
        if m.level >= 1:
            return support_centroid(k)
        pv = pivot(k)
        return np.array([pv[0][0], pv[0][1], 0.0]) if pv is not None else k.com

    # G1 make
    def g1_tasks(S_, k):
        return ([t_region_down(S, k, lowest(k, g), 10.0) for g in missing_groups(k)] +
                [t_axis(k, keep_col, keep_tgt, 2.0)])

    def g1_done(S_, k, cert):
        return not missing_groups(k)

    # G2 orient
    def g2_tasks(S_, k):
        ts = [t_axis(k, col, tgt0, 30.0)] + keep_down(k) + common(k)
        if dst == "STAND":
            ts += [t_com(k, 10.0, [0, 0, 0.30], axes=(2,)), t_posture(S, k, 1.0, qstand, JOINTS[:12])]
        return ts

    def g2_done(S_, k, cert):
        return orient_ok(k, cert) and not missing_groups(k)

    # G3 release
    def g3_tasks(S_, k):
        bad = sorted(touching(k) - m.allowed - need)
        ts = [t_region_up(S, k, r, 0.02, 10.0) for r in bad]
        return ts + [t_axis(k, col, tgt0, 10.0)] + keep_down(k) + common(k)

    def g3_done(S_, k, cert):
        return mode_check(S, k, cert, dst)[0]

    return [Phase("G1_make", g1_tasks, g1_done, cstar, SUBGOAL_TICKS["G1_make"]),
            Phase("G2_orient", g2_tasks, g2_done, cstar, SUBGOAL_TICKS["G2_orient"]),
            Phase("G3_release", g3_tasks, g3_done, cstar, SUBGOAL_TICKS["G3_release"])]


def compile_tracking(S, dst, chain):
    """Imagined keyframe chain -> one region-to-target tracking sub-goal per keyframe, then G3.

    T_i (track keyframe i): trunk axis onto the keyframe axis [axis task, w 30] and a second trunk
    axis onto the keyframe's (full trunk orientation) [axis task, w 10], all 20 joints
    toward the keyframe joints [posture, w 3], CoM ground projection onto the keyframe CoM
    [t_com, w 30], every region touching in the keyframe brought down [t_region_down, w 10],
    soles flat only if B requires feet (the keyframe IK imposes flat soles only then).
    done: axis within TRACK_DEG of the keyframe axis, max joint error <= TRACK_Q, and every
    region loaded in the keyframe touches; the LAST keyframe also requires the full mode
    predicate of B (getup_modes.mode_check incl. statics).
    """
    col = ORIENT_TARGET[dst][0]
    phases = []
    nk = len(chain["keyframes"])
    for i, kf in enumerate(chain["keyframes"]):
        qk = kf["q"]
        kk = S.kin(qk, with_G=False, with_pairs=False)
        ax = kk.R[:, col].copy()
        col2 = (col + 1) % 3                        # second axis -> full trunk orientation
        ax2 = kk.R[:, col2].copy()
        com = kk.com.copy()
        qj = qk[S.r.qadr].copy()
        down = [r for r in kf["touching"]]
        loaded = [r for r, v in kf["loads"].items() if v > 0.02]
        feet = ({"L_foot", "R_foot"} <= set(down)) and any(g & {"L_foot", "R_foot"} for g in MODE[dst].req)
        last = (i == nk - 1)

        def tasks(S_, k, ax=ax, ax2=ax2, col2=col2, com=com, qj=qj, down=down, feet=feet):
            ts = [t_axis(k, col, ax, 30.0), t_axis(k, col2, ax2, 10.0), t_posture(S, k, 3.0, np.asarray(qj), JOINTS),
                  t_com(k, 30.0, com, axes=(0, 1))]
            ts += [t_region_down(S, k, r, 10.0) for r in down]
            if feet:
                ts.append(t_foot_flat(k, 5.0))
            return ts

        def done(S_, k, cert, ax=ax, qj=qj, loaded=loaded, last=last):
            if last:                       # the last keyframe's job: be in mode B
                return mode_check(S, k, cert, dst)[0]
            ang = np.degrees(np.arccos(np.clip(k.R[:, col] @ ax, -1, 1)))
            jerr = float(np.max(np.abs(k.qpos[S.r.qadr] - qj)))
            d = S.region_height(k)
            touch_ok = all(d[REGIONS.index(r)] <= TOUCH_EPS for r in loaded)
            return ang <= TRACK_DEG and jerr <= TRACK_Q and touch_ok

        phases.append(Phase(f"T{i + 1}_track[{kf['stage']}]", tasks, done,
                            lambda S_, k, com=com: np.array([com[0], com[1], 0.0]), TRACK_TICKS))
    g3 = compile_subgoals(S, dst, chain["keyframes"][-1]["q"])[2]
    return phases + [g3]


# --------------------------------------------------------------------------- traces
def attention_summary(seg):
    acur = [L["sup"]["a_cur"] for L in seg if "sup" in L]
    aj = [L["mot"]["a_joint"] for L in seg if "mot" in L]
    al = [np.asarray(L["alpha"]).mean(0) for L in seg if "alpha" in L]
    out = {}
    if acur:
        a = np.mean(acur, 0)
        out["support_top"] = [(REGIONS[i], round(float(a[i]), 3)) for i in np.argsort(-a)[:4]]
    if aj:
        a = np.mean(aj, 0)
        out["motion_top"] = [(JOINTS[i], round(float(a[i]), 3)) for i in np.argsort(-a)[:4]]
    if al:
        a = np.mean(al, 0)
        out["effort_bottleneck"] = (JOINTS[int(np.argmax(a))], round(float(a.max()), 3))
    lands = sorted({r for L in seg for r in L.get("landing", [])})
    out["landing_subtasks"] = lands
    out["gate_opened_ticks"] = sum(1 for L in seg if L.get("gate_opened"))
    return out


def task_terms(S, phase, q):
    k = S.kin(q, with_G=False, with_pairs=False)
    return {t.name: round(float(t.w * np.sum(t.r ** 2)), 4) for t in phase.tasks(S, k)}


def cert_summary(c):
    return dict(rho=round(float(c["rho"]), 4) if np.isfinite(c["rho"]) else None, binding=c.get("binding"),
                feasible=bool(c["feasible"]),
                loads={r: round(float(v), 3) for r, v in c.get("loads", {}).items() if v > 0.005},
                pen_mm=round(1e3 * c["pen"], 2), selfd_mm=round(1e3 * c["selfd"], 2))


def relaxation_ladder(ctx, q, phase):
    """From the failure configuration, run LADDER_TICKS MPC ticks with each constraint family
    relaxed; report task cost reached.  A family whose relaxation unlocks progress is binding."""
    S, A = ctx.S, ctx.A
    res = {}
    for name, kw in [("full", {})] + list(RELAX_LADDER.items()):
        pp = MPCParams(**{**ctx.p.__dict__, **kw})
        mpc = QSMPC(S, A, pp)
        qq = q.copy()
        c0 = sum(task_terms(S, phase, qq).values())
        status = "ok"
        for _ in range(LADDER_TICKS):
            dq, info = mpc.step(qq, phase, {})
            if dq is None:
                status = "QP_INFEASIBLE"
                break
            qq = mpc.project(S.integrate(qq, dq))
        c1 = sum(task_terms(S, phase, qq).values())
        res[name] = dict(cost0=round(c0, 4), cost=round(c1, 4), status=status,
                         done=bool(phase.done(S, S.kin(qq, with_G=False, with_pairs=False),
                                              certify(S, qq, pp.mu))))
    full_drop = res["full"]["cost0"] - res["full"]["cost"]
    binding = [n for n, r in res.items() if n != "full" and
               (r["done"] or (r["cost0"] - r["cost"]) - full_drop >= UNLOCK_FRAC * max(r["cost0"], 1e-9))]
    return res, binding


# --------------------------------------------------------------------------- the oracle
def evaluate_edge(ctx: OracleContext, q_src, src, dst, verbose=True, ladder=True):
    S = ctx.S
    rule = ctx.rules[(src, dst)]
    if rule.status == "pruned":
        return EdgeResult(src, dst, "PRUNED", rule=rule.rule, reason=rule.reason)
    t0 = time.time()
    c_src = certify(S, q_src, ctx.p.mu)
    ok, why = mode_check(S, S.kin(q_src, with_G=False, with_pairs=False), c_src, src)
    if not ok:
        return EdgeResult(src, dst, "INFEASIBLE", reason=f"source config not in mode {src}: {why}")
    # ---- stage 1: EXISTENCE -- imagine a certified keyframe chain (getup_keyframes)
    col, elev = ORIENT_TARGET[dst]
    tk = time.time()
    chain = KeyframeImaginer(S, ctx.p.mu).chain(q_src, dst, col, elev, verbose=verbose)
    chain_rec = dict(ok=chain["ok"], keep=chain["keep"], wall_s=round(time.time() - tk, 1),
                     keyframes=[{k: v for k, v in kf.items() if k != "q"} for kf in chain["keyframes"]],
                     gap=chain["gap"])
    if not chain["ok"]:
        g = chain["gap"]
        diag = dict(kind="EXISTENCE_GAP", stage=g["stage"], frac=g["frac"], attempts=g["attempts"],
                    verdict=(f"no certified quasi-static keyframe of mode-{dst} kind at stage [{g['stage']}] "
                             f"(all CoM candidates fail: " + " | ".join(f"{a['cand']}: {a['why']}" for a in g["attempts"])
                             + ")"))
        r = EdgeResult(src, dst, "INFEASIBLE", diagnosis=diag, q_end=q_src.copy(), wall_s=round(time.time() - t0, 1),
                       reason=f"EXISTENCE_GAP at [{g['stage']}]")
        r.cost = edge_cost([], 0)
        r.subgoals = [dict(name="imagine_keyframe_chain", outcome="GAP", chain=chain_rec)]
        return r
    # ---- stage 2: CONNECTIVITY -- the MPC tracks the chain (region-to-target sub-goals)
    mpc = QSMPC(S, ctx.A, ctx.p)
    q = q_src.copy()
    logs, keyframes = [], []
    subgoals = [dict(name="imagine_keyframe_chain", outcome="OK", chain=chain_rec)]
    status, diag = "FEASIBLE", {}
    for ph in compile_tracking(S, dst, chain):
        k = S.kin(q, with_G=False, with_pairs=False)
        if ph.done(S, k, certify(S, q, ctx.p.mu)):
            subgoals.append(dict(name=ph.name, outcome="SKIPPED (already satisfied)", ticks=0))
            continue
        n0 = len(logs)
        tp = time.time()
        q_start = q.copy()
        mpc.reset_warm()
        q, out = run_phase(S, mpc, q, ph, ctx.p.mu, logs, stall_ticks=STALL_TICKS, stall_tol=STALL_TOL,
                           verbose=verbose)
        # cold-restart rule (warm-start / contact-state artefact guard, deterministic): a STALLED
        # sub-goal is restarted from the stall configuration with a FRESH solver instance (cold
        # warm start, contact state re-seeded from the exact statics LP) as long as the previous
        # attempt reduced the task cost by >= RESTART_GAIN; at most MAX_RESTARTS times.
        restarts = 0
        c_prev = sum(task_terms(S, ph, q_start).values())
        while out in ("STALLED", "QP_INFEASIBLE") and restarts < MAX_RESTARTS:
            c_now = sum(task_terms(S, ph, q).values())
            if (c_prev - c_now) < RESTART_GAIN * max(c_prev, 1e-9) and restarts > 0:
                break
            c_prev = c_now
            restarts += 1
            mpc = QSMPC(S, ctx.A, ctx.p)
            q, out = run_phase(S, mpc, q, ph, ctx.p.mu, logs, stall_ticks=STALL_TICKS, stall_tol=STALL_TOL,
                               verbose=verbose)
        seg = logs[n0:]
        cert = certify(S, q, ctx.p.mu)
        sg = dict(name=ph.name, outcome=out, ticks=len(seg), wall_s=round(time.time() - tp, 1),
                  restarts=restarts,
                  trunk_axes_z_start=[round(float(v), 3) for v in S.kin(q_start, with_G=False, with_pairs=False).R[2]],
                  trunk_axes_z_end=[round(float(v), 3) for v in S.kin(q, with_G=False, with_pairs=False).R[2]],
                  attention=attention_summary(seg), task_terms_end=task_terms(S, ph, q),
                  cert_violations=sum(1 for L in seg if L.get("cert_violation")))
        subgoals.append(sg)
        keyframes.append(dict(name=f"{src}->{dst}:{ph.name}", q=q.copy(), cert=cert_summary(cert)))
        if verbose:
            print(f"   {src}->{dst} {ph.name}: {out} ticks={len(seg)} t={sg['wall_s']}s terms={sg['task_terms_end']}",
                  flush=True)
        if out != "DONE":
            status = "INFEASIBLE"
            last = seg[-1] if seg else {}
            kq = S.kin(q, with_G=False, with_pairs=False)
            diag = dict(kind="CONNECTIVITY", subgoal=ph.name, outcome=out, tick=last.get("tick"),
                        residual_task_terms=sg["task_terms_end"],
                        unmet_mode_predicates=mode_check(S, kq, cert, dst)[1],
                        qp_family_drop=last.get("infeasible_diag"),
                        elastic_iis=[tuple(map(str, e)) for e in (last.get("elastic") or [])],
                        cert_end=cert_summary(cert),
                        peak_rho=max([L["cert"]["rho"] for L in seg if L.get("cert") and np.isfinite(L["cert"]["rho"])],
                                     default=None))
            if ladder:
                lad, binding = relaxation_ladder(ctx, q, ph)
                diag["relaxation_ladder"] = lad
                diag["binding_families"] = binding
                diag["verdict"] = ("constraint-bound: " + ", ".join(binding)) if binding else (
                    "local: certified keyframes exist on both sides, but the MPC finds no quasi-static descent "
                    "direction between them from this configuration and no single constraint-family "
                    "relaxation unlocks it (oracle path-finding limit, NOT a proof of non-existence)")
            break
    res = EdgeResult(src, dst, status, subgoals=subgoals, keyframes=keyframes, diagnosis=diag,
                     q_end=q.copy(), ticks=len(logs), wall_s=round(time.time() - t0, 1))
    res.cost = edge_cost(logs, len(logs))
    if status == "INFEASIBLE":
        res.reason = f"{diag['subgoal']} {diag['outcome']}: " + diag.get("verdict", "")
    return res


# cost terms -- the documented cost is defined in getup_search.COST_DOC; computed here from the
# certified per-tick statics so that the sequencer only consumes numbers.
C_SWITCH, W_EFFORT, W_MARGIN, W_LEN = 1.0, 1.0, 1.0, 0.5


def edge_cost(logs, n):
    ratios = [np.asarray(L["cert"]["ratio"]) for L in logs if L.get("cert") and L["cert"].get("ratio") is not None]
    rhos = [L["cert"]["rho"] for L in logs if L.get("cert") and np.isfinite(L["cert"]["rho"])]
    effort = float(sum(np.mean(r ** 2) for r in ratios))
    peak = float(max(rhos)) if rhos else 0.0
    c = dict(switch=C_SWITCH, effort=round(W_EFFORT * effort, 4),
             margin=round(W_MARGIN * peak / RHO_CAP, 4), length=round(W_LEN * n / 100.0, 4),
             peak_rho=round(peak, 4), ticks=n)
    c["total"] = round(c["switch"] + c["effort"] + c["margin"] + c["length"], 4)
    return c
