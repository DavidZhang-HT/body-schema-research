"""Phases 2-4 + curriculum + 3-way comparison harness (headless, CPU).

Usage:  .venv/bin/python experiment.py <cond> <seed>      cond in {fixed, passive, active}
        .venv/bin/python experiment.py phase2              nominal-vs-real exploration report
Writes results/<cond>_<seed>.json.

Curriculum per episode (identical time / safety budget for all three conditions):
  stand 1.0 s -> double-support calibration 6.0 s -> stepping in place (6 steps)
  -> reference-guided forward walking (10 steps, stride from Phase-3 projection)
  perturbations: floor mu drop (true mu * 0.75) at walk start, lateral push mid-walk.

Conditions (only the body-model source differs; controller is identical):
  fixed   : URDF-nominal theta, never updated.
  passive : quiet standing during calibration; ungated online ID on ALL samples
            (incl. impacts / single support) during the whole episode.
  active  : multi-frequency sway + squat + single-foot unloading during calibration,
            batch ID, then surprise-gated slow adaptation (Phase 4).
"""
import os, sys, json, time
import numpy as np
from model import RealRobot, BodyParams, JOINTS, TAU_MAX
from wbc import WBC, BodyModel, GaitParams, SOLE_PT, G
from scenarios import true_params, true_params_q0, Q0_PRIOR_STD_DEG
from zero_id import ZeroEstimator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
REF = np.load(os.path.join(HERE, "..", "data", "ref_walk.npz"))["q_cycle"]


# ----------------------------------------------------------------------------- identification
class SchemaEstimator:
    """Identifies theta_id = (trunk_dm, trunk_dx, trunk_dy, delay) from proprioception only.

    Foot wrenches are reconstructed from leg joint torques (pi+ has no F/T sensors):
        F_i = J_leg,i^-T (h_leg + M_leg qdd - tau_leg)
    which depends on the leg links only (URDF-reliable), NOT on the uncertain trunk.
    Quasi-static whole-body balance then gives total weight (-> trunk_dm) and the true
    CoM xy (-> trunk_dx, trunk_dy, linear LS in the base frame).  trunk_dz, mu,
    contact stiffness, backlash are NOT observable in double-support statics -> prior.
    Delay: lag between commanded torque and motor-current torque (needs a rich torque
    signal, i.e. active excitation).
    """

    def __init__(self, bm: BodyModel, wbc: WBC):
        self.bm, self.wbc = bm, wbc
        self.A, self.b, self.W = [], [], []
        self.tc, self.tm = [], []

    def add(self, o, tau_cmd, gate=True):
        self.tc.append(tau_cmd.copy()); self.tm.append(o.tau.copy())
        if gate and (np.linalg.norm(o.acc - o.acc / np.linalg.norm(o.acc) * G) > 0.6 or np.abs(o.qd[:12]).max() > 1.0):
            return
        bm, wbc = self.bm, self.wbc
        bm.load_state(o)
        d = bm.d
        M, h = bm.dyn()
        Fz = 0.0; Mx = 0.0; My = 0.0
        for i, s in enumerate(("l", "r")):
            cols = wbc.vadr[i * 6:(i + 1) * 6]
            p, J, _ = bm.jac6(wbc.feet[s], SOLE_PT)
            try:
                F = np.linalg.solve(J[:, cols].T, h[cols] - o.tau[i * 6:(i + 1) * 6])
            except np.linalg.LinAlgError:
                return
            f, tq = F[:3], F[3:]
            mom = np.cross(p, f) + tq
            Fz += f[2]; Mx += mom[0]; My += mom[1]
        if Fz < 30:
            return
        Wt = Fz
        cx, cy = -My / Wt, Mx / Wt          # true CoM xy (static moment balance)
        # nominal-geometry CoM with the CURRENT dm at nominal ipos, then LS for offset
        R = d.xmat[bm.r.base].reshape(3, 3)
        self.A.append(R[:2, :2]); self.b.append(np.array([cx, cy]))
        self.W.append((Wt, d.subtree_com[0][:2].copy(), float(bm.m.body_subtreemass[0]),
                       float(bm.m.body_mass[bm.r.base]), bm.p.trunk_dx, bm.p.trunk_dy))

    def solve(self, prior: BodyParams, min_n=50):
        p = BodyParams(**prior.as_dict())
        n = len(self.W)
        out = dict(n_static=n)
        if n >= min_n:
            Wt = np.array([w[0] for w in self.W])
            m_tot_model = self.W[-1][2]
            dm = float(np.median(Wt) / G - (m_tot_model - p.trunk_dm))
            p.trunk_dm = float(np.clip(dm, -0.5, 2.0))
            # m_true c_true = m_model c_model + m_base_true R (d_true - d_model)
            m_true = m_tot_model - self.W[-1][3] + (self.W[-1][3] - prior.trunk_dm + p.trunk_dm)
            mb = self.W[-1][3] - prior.trunk_dm + p.trunk_dm
            AA, bb = [], []
            for Ri, ci, w in zip(self.A, self.b, self.W):
                c_model = w[1] * (w[2]) / m_true       # approx: same geometry, mass-corrected
                AA.append(mb / m_true * Ri); bb.append(ci - c_model + mb / m_true * Ri @ np.array([w[4], w[5]]))
            x = np.linalg.lstsq(np.vstack(AA), np.concatenate(bb), rcond=None)[0]
            res = np.concatenate(bb) - np.vstack(AA) @ x
            p.trunk_dx, p.trunk_dy = float(np.clip(x[0], -0.06, 0.06)), float(np.clip(x[1], -0.06, 0.06))
            out.update(dm_std=float(np.std(Wt / G) / np.sqrt(n)), off_res_std=float(res.std()))
        # delay via torque-signal cross-correlation (2 ms tick; sub-tick by parabola)
        if len(self.tc) > 200:
            tc = np.array(self.tc)[:, :12]; tm = np.array(self.tm)[:, :12]
            tc = tc - tc.mean(0); tm = tm - tm.mean(0)
            errs = [np.mean((tm[L:] - tc[:len(tc) - L]) ** 2) for L in range(0, 15)]
            L = int(np.argmin(errs))
            if 0 < L < 14:
                a, b_, c = errs[L - 1], errs[L], errs[L + 1]
                L = L + 0.5 * (a - c) / max(a - 2 * b_ + c, 1e-12)
            p.delay = float(np.clip(L * 0.002, 0, 0.03))
            out["delay_contrast"] = float((max(errs) - min(errs)) / max(min(errs), 1e-9))
        return p, out


# ----------------------------------------------------------------------------- Phase 2 exploration
def exploration_targets(t, T, foot_mid, active=True, yaw_twist=0.0, yaw_f=0.35, squat_cyc=0.0):
    """multi-sine trunk sway + slight squat + single-foot light unloading (safe: CoM kept
    >= 2 cm inside the support polygon, |dz| <= 2.5 cm, trunk pitch <= 4 deg).
    Optional (Phase 2b, zero_motion_scan.py): yaw_twist = amplitude (rad) of a trunk yaw twist
    about the planted feet (feet stay flat, no step) -> breaks the L/R sagittal symmetry that
    makes the common-mode intra-leg zero allocations unobservable; squat_cyc = amplitude of an
    extra slow 0.2 Hz squat cycle (m).  Defaults (0) reproduce the original Phase-2 motion."""
    if not active:
        return dict(xy=foot_mid, vxy=np.zeros(2), axy=np.zeros(2), dz=0.0, vz=0.0, az=0.0, pitch=0.0)
    f = np.array([0.3, 0.7, 1.3]); A = np.array([0.012, 0.008, 0.004])
    ph = np.array([0.0, 1.1, 2.3])
    w = 2 * np.pi * f
    x = np.sum(A * np.sin(w * t + ph)); vx = np.sum(A * w * np.cos(w * t + ph)); ax = -np.sum(A * w**2 * np.sin(w * t + ph))
    # lateral: slow unloading cycle (0.25 Hz, +-4.5 cm -> ~80/20 load split) + small sines
    yl = 0.045 * np.sin(2 * np.pi * 0.25 * t); vyl = 0.045 * 2 * np.pi * 0.25 * np.cos(2 * np.pi * 0.25 * t)
    ayl = -0.045 * (2 * np.pi * 0.25) ** 2 * np.sin(2 * np.pi * 0.25 * t)
    y = yl + 0.004 * np.sin(2 * np.pi * 1.1 * t); vy = vyl + 0.004 * 2 * np.pi * 1.1 * np.cos(2 * np.pi * 1.1 * t)
    dz = -0.012 * (1 - np.cos(2 * np.pi * 0.4 * t)); vz = -0.012 * 2 * np.pi * 0.4 * np.sin(2 * np.pi * 0.4 * t)
    az = -0.012 * (2 * np.pi * 0.4) ** 2 * np.cos(2 * np.pi * 0.4 * t)
    if squat_cyc:
        wq = 2 * np.pi * 0.2
        dz += -squat_cyc * (1 - np.cos(wq * t)); vz += -squat_cyc * wq * np.sin(wq * t)
        az += -squat_cyc * wq ** 2 * np.cos(wq * t)
    ramp = min(1.0, t / 1.0) * min(1.0, max(0.0, (T - t) / 1.0))
    out = dict(xy=foot_mid + ramp * np.array([x, y]), vxy=ramp * np.array([vx, vy]),
               axy=ramp * np.array([ax, ayl]), dz=ramp * dz, vz=ramp * vz, az=ramp * az,
               pitch=ramp * 0.06 * np.sin(2 * np.pi * 0.55 * t))
    if yaw_twist:
        out["yaw"] = ramp * yaw_twist * np.sin(2 * np.pi * yaw_f * t + 0.3)
    return out


# ----------------------------------------------------------------------------- Phase 4 slow layer
class SurpriseGate:
    """Layer 3: commit a model update only if (i) the sample is >=0.15 s after the last
    touchdown and in planned double support (no transient impact errors), (ii) at least
    N gated samples were collected since last commit, (iii) the new estimate differs from
    the current one by > k sigma (sufficient evidence)."""

    def __init__(self, k=2.0, N=150):
        self.k, self.N = k, N
        self.last_td = -1.0; self.prev_stance = None; self.n_new = 0; self.commits = 0; self.rejected = 0

    def admissible(self, t, stance):
        if self.prev_stance is not None and len(stance) > len(self.prev_stance):
            self.last_td = t
        self.prev_stance = stance
        return len(stance) == 2 and t - self.last_td > 0.15


# ----------------------------------------------------------------------------- episode
def episode(cond, seed, gp=None, stride=0.06, T_cal=6.0, push=(0, 18.0, 0), push_dur=0.12, verbose=False,
            scenario="base"):
    """cond = <fixed|passive|active>[_nodc][_z|_zt].  '_z' adds joint zero-offset ID (zero_id.py):
    gated loop-closure samples during calibration -> gated commit of q0_hat at calibration end
    -> buffered calibration samples are replayed into SchemaEstimator with corrected kinematics.
    '_zt' = the same with the joint-torque residual channel (ZeroEstimator(torque=True), Phase 2b
    torque channel; four gates incl. torque consistency + channel agreement).
    scenario 'q0' = true_params_q0(seed) (plant with leg joint zero offsets)."""
    parts = cond.split("_")
    flags = set(parts[1:])
    abl = "nodc" if "nodc" in flags else ""
    zid = "z" in flags or "zt" in flags
    ztq = "zt" in flags                     # '_zt' = zero-ID with loop closure + joint-torque channel
    cond_name = cond
    cond = parts[0]
    tp = true_params_q0(seed) if scenario == "q0" else true_params(seed)
    rob = RealRobot(tp, seed=seed)
    prior = BodyParams()
    bm = BodyModel(prior)
    gp = gp or GaitParams(stride=stride)
    wbc = WBC(bm, gp, mu_ctrl=0.6, ref=REF)
    est = SchemaEstimator(BodyModel(prior), wbc)
    est.wbc = wbc  # feet ids/vadr identical across BodyModel instances
    gate = SurpriseGate()
    zest = ZeroEstimator(prior_std=np.deg2rad(Q0_PRIOR_STD_DEG), torque=ztq) if zid else None
    cal_buf = []; zero_info = {}
    o = rob.observe()
    log = dict(t=[], fid=[], fid_true=[], viol=[], dcm_err=[], phase=[])
    T0 = 1.0; t_cal_end = T0 + T_cal
    n_sip, n_fwd = 6, 10
    stage = "stand"; ident_info = {}; p_hat = prior
    t_push = None; mu_changed = False
    foot_prev = None
    tend = t_cal_end + 2 * 0.6 + (n_sip + n_fwd) * gp.T_step + 2.5
    while rob.t < tend:
        t = rob.t
        # ---------------- curriculum / stage machine
        if stage == "stand" and t >= T0:
            stage = "calib"; bm.load_state(o)
            mid = 0.5 * (bm.point(wbc.feet["l"], SOLE_PT)[:2] + bm.point(wbc.feet["r"], SOLE_PT)[:2])
        if stage == "calib":
            wbc.planner.stand = exploration_targets(t - T0, T_cal, mid, active=(cond == "active"))
            if t >= t_cal_end:
                if cond in ("active", "passive"):
                    if zid:
                        # gated zero-offset commit, then replay calibration into SchemaEstimator
                        q0_hat, zero_info = zest.solve()
                        prior_z = BodyParams(**prior.as_dict()); prior_z.q0 = [float(x) for x in q0_hat] + [0.0] * 8
                        est.bm.set_params(prior_z)
                        for oo, tt in cal_buf:
                            est.add(oo, tt, gate=True)
                        p_hat, ident_info = est.solve(prior_z)
                    else:
                        p_hat, ident_info = est.solve(prior)
                    p_hat.frictionloss = prior.frictionloss
                    bm.set_params(p_hat); est.bm.set_params(p_hat)
                    wbc.delay_comp = 0.0 if abl == "nodc" else p_hat.delay
                wbc.planner.stand = {}
                stage = "sip"; bm.load_state(o)
                fp = {s: bm.point(b, SOLE_PT)[:2] for s, b in wbc.feet.items()}
                wbc.planner.start_walk(t, (fp["l"], fp["r"]), bm.d.subtree_com[0][:2].copy(), n_sip, 0.0)
                wbc.ref_on = False
        elif stage == "sip" and wbc.planner.k > wbc.planner.n and t - wbc.planner.step_t0 > 0.6:
            stage = "fwd"; bm.load_state(o)
            fp = {s: bm.point(b, SOLE_PT)[:2] for s, b in wbc.feet.items()}
            wbc.planner.start_walk(t, (fp["l"], fp["r"]), bm.d.subtree_com[0][:2].copy(), n_fwd, gp.stride)
            wbc.ref_on = True
            rob.r.set_mu(tp.mu * 0.75); mu_changed = True        # friction change
        if stage == "fwd" and t_push is None and wbc.planner.k == 5:
            rob.apply_push(push, push_dur); t_push = t
        # ---------------- control
        tau, info = wbc.control(o, t)
        o_next = rob.step(tau)
        # ---------------- online ID (passive: ungated; active: surprise-gated)
        if stage == "calib" and cond in ("active", "passive") and t > T0 + 0.3:
            if zid:
                zest.add(o, info, wbc, t)          # zero-ID sample gate (double support, quasi-static)
                cal_buf.append((o, tau.copy()))
            else:
                est.add(o, tau, gate=True)
        if stage in ("sip", "fwd"):
            if cond == "passive":
                est.add(o, tau, gate=False); gate.n_new += 1
            elif cond == "active" and gate.admissible(t, info["stance"]):
                est.add(o, tau, gate=True); gate.n_new += 1
            if cond in ("passive", "active") and gate.n_new >= gate.N:
                cand, ci = est.solve(p_hat)
                dlt = np.array([cand.trunk_dx - p_hat.trunk_dx, cand.trunk_dy - p_hat.trunk_dy])
                sig = max(ci.get("off_res_std", 0.01) / np.sqrt(max(ci.get("n_static", 1), 1)), 1e-3)
                if cond == "passive" or np.linalg.norm(dlt) > gate.k * sig:
                    cand.frictionloss = prior.frictionloss; cand.delay = p_hat.delay
                    p_hat = cand; bm.set_params(p_hat); gate.commits += 1
                else:
                    gate.rejected += 1
                gate.n_new = 0
                est.A, est.b, est.W = est.A[-1500:], est.b[-1500:], est.W[-1500:]
        o = o_next
        # ---------------- metrics
        if stage in ("sip", "fwd"):
            qr, _ = wbc.posture_ref(t)
            fid = np.sqrt(np.mean((o.q[:12] - qr[:12]) ** 2)) if wbc.ref_on else np.nan
            # ground-truth fidelity (true joint angles; differs from 'fid' only by encoder offsets/noise)
            qt = rob.d.qpos[rob.r.qadr][:12]
            fid_t = np.sqrt(np.mean((qt - qr[:12]) ** 2)) if wbc.ref_on else np.nan
            slip = False
            fpos = np.array([rob.d.xpos[b] for b in rob.r.foot_bid])
            if foot_prev is not None:
                v = (fpos - foot_prev)[:, :2] / rob.ctrl_dt
                slip = bool(np.any((fpos[:, 2] < 0.06) & (np.linalg.norm(v, axis=1) > 0.15) &
                                   (np.array([s in info["stance"] for s in ("l", "r")]))))
            foot_prev = fpos
            viol = (info["tau_ratio"] >= 0.999) or (not info["ok"]) or slip
            xe = np.linalg.norm(info["xi"] - info["xi_ref"]) if "xi_ref" in info else 0.0
            log["t"].append(t); log["fid"].append(fid); log["fid_true"].append(fid_t); log["viol"].append(viol)
            log["dcm_err"].append(xe); log["phase"].append(stage)
        if rob.fallen():
            break
    fell = rob.fallen()
    tt = np.array(log["t"]); de = np.array(log["dcm_err"]); fid = np.array(log["fid"], float)
    rec = None
    if t_push is not None and not fell:
        idx = np.where(tt > t_push + push_dur)[0]
        ok = de[idx] < 0.03
        for j in range(len(idx)):
            if ok[j:j + 50].all():
                rec = float(tt[idx[j]] - t_push); break
    elif t_push is not None and fell:
        rec = None
    def perr(p):
        return dict(dm=p.trunk_dm - tp.trunk_dm, dx=p.trunk_dx - tp.trunk_dx, dy=p.trunk_dy - tp.trunk_dy,
                    delay=p.delay - tp.delay)
    fidt = np.array(log["fid_true"], float)
    q0err = np.rad2deg(np.array(p_hat.q0[:12]) - np.array(tp.q0[:12]))
    res = dict(cond=cond_name, seed=seed, scenario=scenario, fell=bool(fell), t_end=float(rob.t), stage_reached=log["phase"][-1] if log["phase"] else "stand",
               fidelity_deg=float(np.rad2deg(np.nanmean(fid))) if np.isfinite(fid).any() else None,
               fidelity_true_deg=float(np.rad2deg(np.nanmean(fidt))) if np.isfinite(fidt).any() else None,
               viol_rate=float(np.mean(log["viol"])) if log["viol"] else None,
               recovery_s=rec, pushed=t_push is not None,
               fwd_dist=float(rob.d.qpos[0]), lat_drift=float(rob.d.qpos[1]),
               theta_true=tp.as_dict(), theta_hat=p_hat.as_dict(), theta_err=perr(p_hat), theta_err_prior=perr(prior),
               q0_err_deg=q0err.round(3).tolist(), q0_err_rms_deg=float(np.sqrt((q0err ** 2).mean())),
               zero_ident=zero_info,
               ident=ident_info, commits=gate.commits, rejected=gate.rejected, gp=gp.as_dict())
    return res


# ----------------------------------------------------------------------------- Phase 2 report
def phase2(seed=0, T=6.0):
    """Run the SAME active exploration on the nominal plant and on the 'real' plant,
    compare observation channels -> which deviations are task-relevant."""
    out = {}
    for name, tp in (("nominal", BodyParams()), ("real", true_params(seed))):
        rob = RealRobot(tp, seed=seed); bm = BodyModel(BodyParams())
        wbc = WBC(bm, GaitParams(), ref=REF)
        o = rob.observe(); L = dict(com=[], tau=[], grf=[], q=[])
        mid = None
        while rob.t < 1.0 + T:
            t = rob.t
            if t >= 1.0:
                if mid is None:
                    bm.load_state(o)
                    mid = 0.5 * (bm.point(wbc.feet["l"], SOLE_PT)[:2] + bm.point(wbc.feet["r"], SOLE_PT)[:2])
                wbc.planner.stand = exploration_targets(t - 1.0, T, mid)
            tau, info = wbc.control(o, t)
            o = rob.step(tau)
            if t >= 1.0:
                L["com"].append(rob.true_com() - info["com"]); L["tau"].append(o.tau[:12].copy())
                L["grf"].append([info["grf"]["l"], info["grf"]["r"]]); L["q"].append(o.q[:12].copy())
        out[name] = {k: np.array(v) for k, v in L.items()}
        out[name + "_fell"] = rob.fallen()
    n, r = out["nominal"], out["real"]
    m = min(len(n["tau"]), len(r["tau"]))
    rep = dict(
        fell=dict(nominal=bool(out["nominal_fell"]), real=bool(out["real_fell"])),
        com_model_error_mm_nominal=(1e3 * np.abs(n["com"]).mean(0)).round(2).tolist(),
        com_model_error_mm_real=(1e3 * np.abs(r["com"]).mean(0)).round(2).tolist(),
        leg_tau_rms_diff_Nm=np.sqrt(((r["tau"][:m] - n["tau"][:m]) ** 2).mean(0)).round(3).tolist(),
        leg_tau_mean_diff_Nm=(r["tau"][:m] - n["tau"][:m]).mean(0).round(3).tolist(),
        grf_sum_mean_N=dict(nominal=float(n["grf"].sum(1).mean()), real=float(r["grf"].sum(1).mean())),
        min_foot_load_frac=dict(nominal=float((n["grf"].min(1) / n["grf"].sum(1)).min()),
                                real=float((r["grf"].min(1) / r["grf"].sum(1)).min())),
        joint_q_rms_diff_deg=np.rad2deg(np.sqrt(((r["q"][:m] - n["q"][:m]) ** 2).mean(0))).round(3).tolist(),
    )
    return rep


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "phase2":
        rep = phase2()
        json.dump(rep, open(os.path.join(OUT, "phase2.json"), "w"), indent=1)
        print(json.dumps(rep, indent=1))
    else:
        cond, seed = sys.argv[1], int(sys.argv[2])
        stride = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06
        scenario = sys.argv[4] if len(sys.argv) > 4 else "base"
        t0 = time.time()
        r = episode(cond, seed, stride=stride, scenario=scenario)
        r["wall_s"] = time.time() - t0
        odir = OUT if scenario == "base" else os.path.join(OUT, scenario)
        os.makedirs(odir, exist_ok=True)
        json.dump(r, open(os.path.join(odir, f"{cond}_{seed}_s{stride}.json"), "w"), indent=1)
        print(cond, seed, "fell" if r["fell"] else "ok", r["stage_reached"], "t_end", round(r["t_end"], 2),
              "fid", r["fidelity_deg"], "fid_true", r["fidelity_true_deg"],
              "viol", r["viol_rate"], "rec", r["recovery_s"], "err", {k: round(v, 4) for k, v in r["theta_err"].items()},
              "q0_rms", round(r["q0_err_rms_deg"], 3), "zcommit", r["zero_ident"].get("committed"),
              "ident", r["ident"], "commits", r["commits"], r["rejected"], "wall", round(r["wall_s"], 1), flush=True)
