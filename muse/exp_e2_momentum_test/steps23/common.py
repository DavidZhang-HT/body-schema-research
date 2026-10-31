"""E2 Steps 2-3 shared infrastructure: MuJoCo forward dynamics on scheme A's plant.

Plant: model.PiPlus(NOMINAL, timestep=0.001, shell="full") -- the same MJCF XML,
the same G0 primitive contact shell, and the same nominal BodyParams that
scheme A's Schema uses (getup_schema.Schema builds PiPlus(params=BodyParams(),
shell="full")).  model.py's builder never calls mj_step; we create our own
MjData stepping loop here (clean nominal physics: no sensor noise, no delay,
no q0 offsets -- unlike RealRobot, which is the "physical" noisy plant).

Mode evaluation reuses the committed vocabulary: Schema.kin + statics_lp +
mode_check from sim/getup_modes.py, with SIDE_L registered harness-locally
(exact mirror of SIDE, as in step1/run_step1.py D1).  Success for the dynamic
edge = the geometric mode predicate (req-touch + allowed/forbidden +
orientation), i.e. mode_check with a trivially-feasible cert -- the task's
"mode predicate, not the quasi-static certifier".  The real statics LP is
still computed and reported for every success.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = "/home/hatch/workspace/research/body-schema-gait/muse/sim"
STEP1 = "/home/hatch/workspace/research/body-schema-gait/muse/exp_e2_momentum_test/step1"
for p in (SIM, STEP1):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import mujoco

from model import PiPlus, NOMINAL, JOINTS, TAU_MAX
from getup_schema import Schema
from getup_modes import mode_check, Mode, LYING_SIDE
import getup_modes as gm
from getup_mpc import certify
from mirror import mirror_q

MU_PLAN = 1.0  # same as Schema default

# ------------------------------------------------------------------ SIDE_L (harness-local, mirrors step1 D1)
def _f(*r):
    return frozenset(r)

if "SIDE_L" not in gm.MODE:
    SIDE_L = Mode(
        "SIDE_L", "side-lying {left side}",
        req=(_f("upper_trunk", "lower_back", "pelvis", "L_upper_arm"),), allowed=gm.ALL,
        orient_txt="left_z <= -0.7 (body +y down, i.e. left side down)",
        orient=lambda R: R[2, 1] <= -0.7,
        oclass=LYING_SIDE, level=0,
        meaning="harness-local exact mirror of SIDE (E2 steps 2-3); tests left roll")
    gm.MODES.append(SIDE_L)
    gm.MODE["SIDE_L"] = SIDE_L

TRIVIAL_CERT = {"feasible": True, "rho": 0.0}

# ------------------------------------------------------------------ plant
class Plant:
    """Thin wrapper: PiPlus model + our own MjData stepping loop."""

    def __init__(self, timestep=0.001):
        self.r = PiPlus(NOMINAL, timestep=timestep, shell="full")
        self.m = self.r.m
        self.d = mujoco.MjData(self.m)
        self.qadr = self.r.qadr
        self.vadr = self.r.vadr
        self.act = self.r.act
        self.nv = self.m.nv
        self._M = np.zeros((self.nv, self.nv))
        self.floor_gid = self.r.floor_gid
        # PD gains (legs 12, arms 8) in JOINTS order
        self.kp = np.array([50.0] * 12 + [15.0] * 8)
        self.kd = np.array([2.5] * 12 + [0.8] * 8)
        self.lo = self.r.lo.copy()
        self.hi = self.r.hi.copy()

    def spawn(self, q):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:] = np.asarray(q, float)
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, d)

    def snapshot(self):
        d = self.d
        return (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.time)

    def restore(self, snap):
        qpos, qvel, ctrl, t = snap
        d = self.d
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.ctrl[:] = ctrl
        d.time = t
        mujoco.mj_forward(self.m, d)

    def step_pd(self, q_tgt):
        """One physics step with PD torque toward q_tgt (JOINTS order). Returns tau."""
        d = self.d
        q = d.qpos[self.qadr]
        qd = d.qvel[self.vadr]
        tau = self.kp * (np.asarray(q_tgt, float) - q) - self.kd * qd
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        d.ctrl[self.act] = tau
        mujoco.mj_step(self.m, d)
        return tau

    def step_torque(self, tau):
        d = self.d
        tau = np.clip(np.asarray(tau, float), -TAU_MAX, TAU_MAX)
        d.ctrl[self.act] = tau
        mujoco.mj_step(self.m, d)
        return tau

    # ------------------------------------------------------------- metrics
    def kinetic_energy(self):
        mujoco.mj_fullM(self.m, self.d, self._M)
        v = self.d.qvel
        return 0.5 * float(v @ (self._M @ v))

    def base_R(self):
        return self.d.xmat[self.r.base].reshape(3, 3).copy()

    def com(self):
        return self.d.subtree_com[0].copy()

    def min_self_dist(self):
        """Most negative non-floor contact distance (self-penetration probe)."""
        d = self.d
        out = 1.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_gid in (c.geom1, c.geom2):
                continue
            if c.dist < out:
                out = c.dist
        return float(out)

    def min_floor_dist(self):
        d = self.d
        out = 1.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_gid in (c.geom1, c.geom2):
                if c.dist < out:
                    out = c.dist
        return float(out)

    def finite(self):
        return bool(np.all(np.isfinite(self.d.qpos)) and np.all(np.isfinite(self.d.qvel)))


# ------------------------------------------------------------------ mode evaluation
class ModeEval:
    """Schema-based mode checking on MuJoCo states (kinematics only)."""

    def __init__(self, mu=MU_PLAN):
        self.S = Schema(mu=mu)
        self.mu = mu

    def kin(self, qpos):
        return self.S.kin(np.asarray(qpos, float), with_G=False, with_pairs=False)

    def predicate(self, qpos, key):
        """Geometric mode predicate only (trivially-feasible cert)."""
        k = self.kin(qpos)
        ok, why = mode_check(self.S, k, TRIVIAL_CERT, key)
        return ok, why, k

    def full_check(self, qpos, key):
        """Full mode_check incl. the quasi-static certifier (for reporting)."""
        k = self.S.kin(qpos, with_G=False, with_pairs=False)
        cert = self.S.statics_lp(k)
        ok, why = mode_check(self.S, k, cert, key)
        return ok, why, cert, k

    def certify(self, qpos):
        return certify(self.S, np.asarray(qpos, float), self.mu)

    def roll_progress(self, qpos, sign):
        k = self.kin(qpos)
        return float(sign * k.R[2, 1])


# ------------------------------------------------------------------ starts
def load_clean_start():
    return np.load(os.path.join(STEP1, "q_supine_clean.npy"))


def make_starts(mev, joint_noise_seed=101):
    """6 starts: clean/mirrored/perturbed x right/left roll.  Each verified SUPINE."""
    q_clean = load_clean_start()
    q_mirr = mirror_q(q_clean, JOINTS, mev.S.r.qadr)
    rng = np.random.default_rng(joint_noise_seed)
    starts = []
    # 1-2: clean -> right
    starts.append(("clean_R_a", q_clean.copy(), "SIDE", +1, 11))
    starts.append(("clean_R_b", q_clean.copy(), "SIDE", +1, 22))
    # 3-4: mirrored -> left
    starts.append(("mirr_L_a", q_mirr.copy(), "SIDE_L", -1, 11))
    starts.append(("mirr_L_b", q_mirr.copy(), "SIDE_L", -1, 22))
    # 5: clean + joint noise -> right
    qn = q_clean.copy()
    dq = rng.normal(0, 0.04, 20)
    qn[mev.S.r.qadr] = np.clip(qn[mev.S.r.qadr] + dq, mev.S.r.lo, mev.S.r.hi)
    starts.append(("noisy_R", qn, "SIDE", +1, 33))
    # 6: mirrored + base yaw 0.15 rad -> left
    qy = q_mirr.copy()
    qz = np.array([np.cos(0.15 / 2), 0, 0, np.sin(0.15 / 2)])
    qq = np.zeros(4)
    mujoco.mju_mulQuat(qq, qz, qy[3:7])
    qy[3:7] = qq
    starts.append(("mirr_yaw_L", qy, "SIDE_L", -1, 33))

    verified = []
    for tag, q, dst, sign, seed in starts:
        ok, why, cert, _ = mev.full_check(q, "SUPINE")
        c = mev.certify(q)
        verified.append({
            "tag": tag, "dst": dst, "sign": sign, "seed": seed,
            "supine_ok": bool(ok), "supine_why": why,
            "rho0": float(cert["rho"]), "selfd0_mm": float(1e3 * c["selfd"]),
        })
        assert ok, f"start {tag} not SUPINE: {why}"
        assert c["selfd"] >= -0.001, f"start {tag} self-penetrating: {c['selfd']}"
    return starts, verified


# ------------------------------------------------------------------ rollout with PD-tracked waypoints
def rollout_block(plant, q0, waypoints, T_block, ctrl_dt=0.02,
                  log_every=1, w_selfpen=300.0):
    """Execute PD-tracked waypoint trajectory from plant state; returns trace dict.

    waypoints: (n_wp, 20) absolute joint targets in JOINTS order.
    trace: per-control-step t, KE, com, R, min_self_dist, tau.
    """
    d = plant.d
    sub = int(round(ctrl_dt / plant.m.opt.timestep))
    n_wp = len(waypoints)
    knots = np.vstack([d.qpos[plant.qadr][None, :], waypoints])
    times = np.linspace(0, T_block, n_wp + 1)
    n_steps = int(round(T_block / ctrl_dt))
    trace = {"t": [], "ke": [], "com": [], "R21": [], "selfd": [],
             "tau": [], "com_speed": [], "qpos": [], "qvel": []}
    com_prev = plant.com()
    peak_ke = 0.0
    peak_com_speed = 0.0
    max_selfpen = 0.0  # most negative self dist -> penetration depth positive
    ok = True
    for s in range(n_steps):
        t = s * ctrl_dt
        # linear interp between knots
        f = t / T_block * n_wp
        i = min(int(f), n_wp - 1)
        a = f - i
        q_t = (1 - a) * knots[i] + a * knots[i + 1]
        for _ in range(sub):
            tau = plant.step_pd(q_t)
        if not plant.finite():
            ok = False
            break
        ke = plant.kinetic_energy()
        peak_ke = max(peak_ke, ke)
        com = plant.com()
        cs = float(np.linalg.norm(com - com_prev) / ctrl_dt)
        com_prev = com
        peak_com_speed = max(peak_com_speed, cs)
        sd = plant.min_self_dist()
        max_selfpen = max(max_selfpen, -min(sd, 0.0))
        R = plant.base_R()
        if s % log_every == 0:
            trace["t"].append(float(d.time))
            trace["ke"].append(float(ke))
            trace["com"].append(com.tolist())
            trace["R21"].append(float(R[2, 1]))
            trace["selfd"].append(float(sd))
            trace["tau"].append(tau.tolist())
            trace["com_speed"].append(float(cs))
            trace["qpos"].append(d.qpos.copy().tolist())
            trace["qvel"].append(d.qvel.copy().tolist())
    trace["peak_ke"] = float(peak_ke)
    trace["peak_com_speed"] = float(peak_com_speed)
    trace["max_selfpen_m"] = float(max_selfpen)
    trace["ok"] = bool(ok)
    trace["ke_end"] = float(plant.kinetic_energy())
    return trace
