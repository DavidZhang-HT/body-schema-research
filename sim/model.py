"""pi+ (HighTorque Mini Pi Plus) MuJoCo model construction + "real robot" plant.

Source of truth for the static kinematic schema: official HighTorque files
  assets/pi_plus_24dof/xml/pi_plus_20dof.xml   (MJCF, 20 actuated DoF)
  assets/pi_plus_24dof/urdf/pi_plus_24dof.urdf (URDF, 24 DoF incl. wrists/head)
fetched from github.com/HighTorque-Robotics/HT_Robot_URDF (main).

Modifications (documented in plan.md):
  * mesh geoms removed (headless, CPU) -> primitive collision geoms
    (trunk boxes kept from the official MJCF, capsules for limbs, box soles
    sized from the ankle_roll_link STL bounding box: 172 x 72 mm).
  * wrist + head joints frozen (as in the official 20-DoF MJCF).
  * arm torque limits = 10 Nm (URDF value; the MJCF says 20 -> inconsistency).
  * inertials are kept exactly as in the official MJCF (= "URDF-nominal").
"""
from __future__ import annotations

import os
import re
import copy
from dataclasses import dataclass, field, asdict

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFFICIAL_XML = os.path.join(ROOT, "assets", "pi_plus_24dof", "xml", "pi_plus_20dof.xml")

LEG_JOINTS = [f"{s}_{j}_joint" for s in ("l", "r") for j in
              ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")]
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("l", "r") for j in
              ("shoulder_pitch", "shoulder_roll", "upper_arm", "elbow")]
JOINTS = LEG_JOINTS + ARM_JOINTS          # canonical controller order (20)
TAU_MAX = np.array([20.0] * 12 + [10.0] * 8)
FEET = ("l_ankle_roll_link", "r_ankle_roll_link")
# sole box in ankle_roll_link frame (from STL bbox x[-0.112,0.06] y[-0.036,0.036] z_min -0.05)
SOLE_CENTER = np.array([-0.026, 0.0, -0.045])
SOLE_HALF = np.array([0.086, 0.036, 0.005])
SOLE_Z = -0.05
FOOT_CORNERS = np.array([[SOLE_CENTER[0] + sx * SOLE_HALF[0], sy * SOLE_HALF[1], SOLE_Z]
                         for sx in (1, -1) for sy in (1, -1)])

# Limb capsules for self-collision (fromto in body frame), group 3.
_CAPS = {
    "l_hip_roll_link": ("0 0 -0.02 0 0 -0.09", 0.032),
    "r_hip_roll_link": ("0 0 -0.02 0 0 -0.09", 0.032),
    "l_calf_link": ("0 0 -0.01 0 0 -0.12", 0.030),
    "r_calf_link": ("0 0 -0.01 0 0 -0.12", 0.030),
    "l_elbow_link": ("0 0 -0.01 0 0 -0.09", 0.022),
    "r_elbow_link": ("0 0 -0.01 0 0 -0.09", 0.022),
    "l_shoulder_roll_link": ("0 0 -0.01 0 0 -0.06", 0.024),
    "r_shoulder_roll_link": ("0 0 -0.01 0 0 -0.06", 0.024),
}


@dataclass
class BodyParams:
    """Dynamic + contact schema parameters theta (what the URDF does not tell you)."""
    trunk_dm: float = 0.0        # extra trunk mass [kg] (battery/compute/cables not in URDF)
    trunk_dx: float = 0.0        # trunk CoM offset [m]
    trunk_dy: float = 0.0
    trunk_dz: float = 0.0
    delay: float = 0.0           # actuation latency [s]
    frictionloss: float = 0.2    # leg joint Coulomb friction / stiction [Nm] (MJCF default 0.2)
    contact_tc: float = 0.02     # ground contact solref time constant [s] (stiffness proxy)
    mu: float = 1.0              # foot-ground friction coefficient
    backlash: float = 0.0        # ankle gearbox backlash [rad] (plant only, kept as prior)
    # joint zero offsets (encoder / assembly bias) in JOINTS order [rad]:
    #   q_meas = q_true + q0.  Plant: injected in RealRobot.observe().  Controller model:
    #   BodyModel.load_state() uses q_true_hat = q_meas - q0_hat (q-correction before planning).
    q0: list = field(default_factory=lambda: [0.0] * 20)

    def vec(self, keys):
        return np.array([getattr(self, k) for k in keys])

    def with_vec(self, keys, v):
        p = copy.deepcopy(self)
        for k, x in zip(keys, v):
            setattr(p, k, float(x))
        return p

    def as_dict(self):
        return asdict(self)


NOMINAL = BodyParams()


# ---------------------------------------------------------------------------------------
# G0 (get-up) contact shell.  The official meshes are not available for the limbs (only
# base_link / ankle_roll STLs are vendored), so the limb shell is a *simplified primitive
# shell* sized from the link offsets in the MJCF.  Per analysis-getup.md G0, the gap between
# the real (mesh) shell and this primitive shell is by definition the "contact-schema error";
# the internal model and the plant share this shell here (nominal-model F3 test).
# Added only when shell="full" so that every legacy (walking / q0) run stays bit-identical.
_GETUP_SHELL = {
    # body: list of (name-suffix, type, spec, radius)
    "l_thigh_link": [("thigh", "capsule", "0 0 0.0 0 0 -0.045", 0.032)],
    "r_thigh_link": [("thigh", "capsule", "0 0 0.0 0 0 -0.045", 0.032)],
    "l_calf_link": [("knee", "sphere", "0.012 0 0", 0.030)],
    "r_calf_link": [("knee", "sphere", "0.012 0 0", 0.030)],
    "l_upper_arm_link": [("uarm", "capsule", "0 0 0 0 0 -0.038", 0.022)],
    "r_upper_arm_link": [("uarm", "capsule", "0 0 0 0 0 -0.038", 0.022)],
    "l_elbow_link": [("hand", "sphere", "0 0 -0.110", 0.025)],
    "r_elbow_link": [("hand", "sphere", "0 0 -0.110", 0.025)],
}


def build_xml(timestep=0.001, shell: str = "feet") -> str:
    """Load the official MJCF and turn it into a headless, primitive-geometry model.

    shell="feet" (default, legacy): only soles + trunk/head boxes touch the floor; limb
    capsules are self-collision only.  shell="full" (get-up, G0): limb capsules + added
    thigh / knee / upper-arm / hand primitives also collide with the floor, and the trunk
    boxes collide with the limb shell (thigh-to-chest etc.)."""
    xml = open(OFFICIAL_XML).read()
    xml = re.sub(r'<compiler[^>]*/>', '<compiler angle="radian" autolimits="true"/>', xml)
    xml = re.sub(r'\s*<mesh [^>]*/>', '', xml)
    xml = re.sub(r'\s*<geom [^>]*mesh="[^"]*"[^>]*/>', '', xml)
    # official head geoms were meshes -> add head box on trunk for collision/visual mass-less
    xml = xml.replace('<site name="imu"', '<geom name="head" type="box" size="0.05 0.05 0.05" pos="0.01 0 0.27" '
                      'contype="2" conaffinity="0" rgba="1 1 1 1" density="0"/>\n      <site name="imu"', 1)
    # trunk boxes: name them, collide only for fall detection
    xml = xml.replace('rgba="0.75294 0.75294 0.75294 1"/>', 'rgba="0.75294 0.75294 0.75294 1" density="0" contype="2" conaffinity="1"/>')
    # remove the 'sensor' block (we compute observations in python) and visual extras
    xml = re.sub(r'<sensor>.*?</sensor>', '', xml, flags=re.S)
    xml = re.sub(r'<statistic[^>]*/>', '', xml)
    # arm actuator limits: URDF says 10 Nm
    for j in ARM_JOINTS:
        xml = xml.replace(f'joint="{j}" gear="1" ctrllimited="true" ctrlrange="-20 20"',
                          f'joint="{j}" gear="1" ctrllimited="true" ctrlrange="-10 10"')
        xml = re.sub(rf'(name="{j}"[^>]*actuatorfrcrange=)"-20 20"', r'\1"-10 10"', xml)
    # feet soles + limb capsules
    for foot in FEET:
        sole = (f'<geom name="{foot[0]}_sole" type="box" size="{SOLE_HALF[0]} {SOLE_HALF[1]} {SOLE_HALF[2]}" '
                f'pos="{SOLE_CENTER[0]} {SOLE_CENTER[1]} {SOLE_CENTER[2]}" contype="1" conaffinity="1" '
                f'friction="0.01 0.005 0.0001" condim="3" rgba="0.2 0.2 0.2 1" density="0"/>')
        xml = re.sub(rf'(<body name="{foot}"[^>]*>\s*<inertial[^>]*/>\s*<joint[^>]*/>)', r'\1\n' + sole, xml)
    for body, (ft, r) in _CAPS.items():
        cap = (f'<geom name="{body}_cap" type="capsule" fromto="{ft}" size="{r}" contype="4" conaffinity="4" '
               f'group="3" density="0" rgba="0.5 0.5 0.8 0.3"/>')
        xml = re.sub(rf'(<body name="{body}"[^>]*>\s*<inertial[^>]*/>\s*<joint[^>]*/>)', r'\1\n' + cap, xml)
    if shell == "full":
        for body, geoms in _GETUP_SHELL.items():
            for suf, typ, spec, r in geoms:
                where = f'fromto="{spec}"' if typ == "capsule" else f'pos="{spec}"'
                g = (f'<geom name="{body}_{suf}" type="{typ}" {where} size="{r}" contype="4" conaffinity="4" '
                     f'group="3" density="0" rgba="0.8 0.5 0.5 0.4"/>')
                xml = re.sub(rf'(<body name="{body}"[^>]*>\s*<inertial[^>]*/>\s*<joint[^>]*/>)', r'\1\n' + g, xml)
        # trunk boxes also collide with the limb shell (bit 4) -> thigh/arm vs chest
        xml = xml.replace('density="0" contype="2" conaffinity="1"/>', 'density="0" contype="2" conaffinity="5"/>')
    elif shell != "feet":
        raise ValueError(shell)
    # floor: only collides with soles (contype 1) and trunk (2) [+ limb shell (4) if "full"]
    floor_ca = "7" if shell == "full" else "3"
    xml = xml.replace('<geom contype="15" conaffinity="15"  name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
                      f'<geom contype="3" conaffinity="{floor_ca}" name="floor" size="0 0 0.05" type="plane" '
                      'friction="1 0.005 0.0001" condim="3"/>')
    xml = re.sub(r'<texture[^>]*/>|<material[^>]*/>', '', xml)
    excl = "".join(f'<exclude body1="{s}_{a}" body2="{s}_{b}"/>' for s in "lr" for a, b in
                   (("hip_roll_link", "calf_link"), ("shoulder_roll_link", "elbow_link"),
                    ("calf_link", "ankle_roll_link"), ("hip_pitch_link", "calf_link")))
    xml = xml.replace('</mujoco>', f'<contact>{excl}</contact>\n</mujoco>')
    xml = xml.replace('<mujoco model="pi_plus_24dof">',
                      f'<mujoco model="pi_plus_20dof_bodyschema">\n  <option timestep="{timestep}" integrator="implicitfast" '
                      f'cone="pyramidal"/>\n  <size memory="4M"/>', 1)
    return xml


class PiPlus:
    """Index helper around an MjModel built from the official pi+ MJCF."""

    def __init__(self, params: BodyParams = NOMINAL, timestep=0.001, shell: str = "feet"):
        self.shell = shell
        self.xml = build_xml(timestep, shell)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.base = m.body("base_link").id
        self.jid = np.array([m.joint(j).id for j in JOINTS])
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        self.foot_bid = np.array([m.body(f).id for f in FEET])
        self.sole_gid = np.array([m.geom(f"{f[0]}_sole").id for f in FEET])
        self.floor_gid = m.geom("floor").id
        self.leg_v = self.vadr[:12]
        self.nv = m.nv
        self.total_mass0 = float(m.body_subtreemass[0])
        self._m0 = (m.body_mass[self.base], m.body_ipos[self.base].copy())
        # BUGFIX of the official MJCF: the free joint inherits class "pi_motor", which puts
        # damping/armature/frictionloss on the floating base (non-physical). Zero them.
        m.dof_frictionloss[:6] = 0
        m.dof_damping[:6] = 0
        m.dof_armature[:6] = 0
        self._floss0 = m.dof_frictionloss.copy()
        self.lo = m.jnt_range[self.jid, 0].copy()
        self.hi = m.jnt_range[self.jid, 1].copy()
        self.params = None
        self.set_params(params)

    # ---------------------------------------------------------------- params
    def set_params(self, p: BodyParams):
        m = self.m
        m0, c0 = self._m0
        m.body_mass[self.base] = m0 + p.trunk_dm
        m.body_ipos[self.base] = c0 + np.array([p.trunk_dx, p.trunk_dy, p.trunk_dz])
        m.dof_frictionloss[:] = self._floss0
        m.dof_frictionloss[self.leg_v] = p.frictionloss
        m.geom_friction[self.floor_gid, 0] = p.mu
        for g in list(self.sole_gid) + [self.floor_gid]:
            m.geom_solref[g] = [p.contact_tc, 1.0]
        mujoco.mj_setConst(m, self.d)
        self.params = copy.deepcopy(p)

    def set_mu(self, mu):
        self.m.geom_friction[self.floor_gid, 0] = mu
        self.params.mu = mu

    # ---------------------------------------------------------------- state
    def default_q(self, knee=0.35):
        """Slightly crouched standing posture (hip pitch, knee, ankle) in JOINTS order."""
        q = np.zeros(20)
        for s, sign in (("l", 1), ("r", 1)):
            q[JOINTS.index(f"{s}_hip_pitch_joint")] = -knee / 2
            q[JOINTS.index(f"{s}_calf_joint")] = knee
            q[JOINTS.index(f"{s}_ankle_pitch_joint")] = -knee / 2
        q[JOINTS.index("l_shoulder_roll_joint")] = 0.12
        q[JOINTS.index("r_shoulder_roll_joint")] = -0.12
        return q

    def reset(self, qj=None, d=None):
        d = d or self.d
        mujoco.mj_resetData(self.m, d)
        qj = self.default_q() if qj is None else qj
        d.qpos[self.qadr] = qj
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(self.m, d)
        zmin = min(self.sole_points(d)[:, 2])
        d.qpos[2] -= zmin + 0.0005
        mujoco.mj_forward(self.m, d)
        return d

    def supine_q(self):
        """G0: supine rest posture (legs straight, arms alongside the trunk), JOINTS order."""
        q = np.zeros(20)
        q[JOINTS.index("l_shoulder_roll_joint")] = 0.12
        q[JOINTS.index("r_shoulder_roll_joint")] = -0.12
        return q

    def reset_supine(self, qj=None, d=None, yaw=0.0):
        """G0: spawn lying on the back (chest up, head toward -x), lowest shell point on z=0.

        Base orientation = Rz(yaw) * Ry(-pi/2): body +x (chest) -> world +z, body +z (head)
        -> world -x.  Requires shell="full" for the limbs to rest on the floor physically."""
        qy = np.array([np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0])
        qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        qq = np.zeros(4); mujoco.mju_mulQuat(qq, qz, qy)
        return self.reset_pose(self.supine_q() if qj is None else qj, qq, d)

    def reset_pose(self, qj, quat, d=None):
        """Place the robot with joint posture qj and base orientation quat (wxyz) so that its
        lowest floor-colliding shell point touches z=0 (+0.5 mm clearance)."""
        d = d or self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.qadr] = qj
        d.qpos[3:7] = np.asarray(quat, float) / np.linalg.norm(quat)
        d.qpos[0:3] = [0, 0, 1.0]
        mujoco.mj_kinematics(self.m, d)
        d.qpos[2] -= self.lowest_shell_z(d) + 0.0005
        mujoco.mj_forward(self.m, d)
        return d

    def lowest_shell_z(self, d=None):
        """Lowest point of every floor-colliding geom (box corners, capsule/sphere bottoms)."""
        d = d or self.d
        m = self.m
        fl = m.geom_conaffinity[self.floor_gid]
        zmin = np.inf
        for g in range(m.ngeom):
            if g == self.floor_gid or not ((m.geom_contype[g] & fl) or (m.geom_conaffinity[g] & m.geom_contype[self.floor_gid])):
                continue
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                z = c[2] - np.abs(R[2, :]) @ s
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = c[2] - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = c[2] - abs(R[2, 2]) * s[1] - s[0]
            else:
                continue
            zmin = min(zmin, z)
        return zmin

    def sole_points(self, d=None):
        d = d or self.d
        pts = []
        for b in self.foot_bid:
            R = d.xmat[b].reshape(3, 3)
            pts.append(d.xpos[b] + FOOT_CORNERS @ R.T)
        return np.vstack(pts)

    def com(self, d=None):
        d = d or self.d
        return d.subtree_com[0].copy()


# ============================================================================
@dataclass
class Obs:
    t: float
    q: np.ndarray          # joint encoder positions (20)
    qd: np.ndarray         # joint velocities (20)
    tau: np.ndarray        # measured joint torque (motor current based, 20)
    base_pos: np.ndarray   # state-estimator output (3)
    base_quat: np.ndarray  # IMU orientation (4, wxyz)
    base_linvel: np.ndarray  # world-frame base velocity (estimator, 3)
    gyro: np.ndarray       # body-frame angular velocity (3)
    acc: np.ndarray        # body-frame specific force (3)


class RealRobot:
    """The 'physical' pi+: true parameters theta*, unknown to the controller.

    Effects that are NOT in the URDF: payload mass + CoM offset, actuation delay,
    joint stiction/Coulomb friction, ankle backlash, softer foot pads, lower mu,
    joint zero offsets (encoder/assembly bias, params.q0: q_meas = q_true + q0).
    Sensors: joint encoders (+zero offset, +backlash hysteresis on ankles, 2e-4 rad noise),
    motor-current torque (2% + 0.05 Nm noise), IMU (quat 1e-3, gyro 5e-3, acc 5e-2),
    and a base state estimator (pos 2 mm, vel 0.02 m/s noise).
    NO foot force/torque sensors (pi+ has none) -> contact is estimated.
    """

    def __init__(self, params: BodyParams, seed=0, ctrl_dt=0.002, shell: str = "feet"):
        self.r = PiPlus(params, shell=shell)
        self.m, self.d = self.r.m, self.r.d
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.dt = self.m.opt.timestep
        self.substeps = int(round(ctrl_dt / self.dt))
        self.ctrl_dt = ctrl_dt
        nbuf = int(round(params.delay / self.dt))      # latency in physics steps
        self.buf = [np.zeros(20)] * nbuf
        self.tau_applied = np.zeros(20)
        self.push = None
        self.ankle_idx = [JOINTS.index(j) for j in JOINTS if "ankle" in j]
        self.r.reset()
        self._last_vel = np.zeros(3)

    def reset_supine(self, qj=None, quat=None):
        if quat is None:
            self.r.reset_supine(qj)
        else:
            self.r.reset_pose(qj, quat)
        self.buf = [np.zeros(20)] * len(self.buf)
        self.tau_applied = np.zeros(20)

    def reset(self, qj=None):
        self.r.reset(qj)
        self.buf = [np.zeros(20)] * len(self.buf)
        self.tau_applied = np.zeros(20)

    @property
    def t(self):
        return self.d.time

    def apply_push(self, force_xyz, duration):
        self.push = (np.asarray(force_xyz, float), self.d.time + duration)

    def step(self, tau_cmd):
        tau_cmd = np.clip(tau_cmd, -TAU_MAX, TAU_MAX)
        for _ in range(self.substeps):
            self.buf.append(tau_cmd.copy())
            self.tau_applied = self.buf.pop(0)
            self.d.ctrl[self.r.act] = self.tau_applied
            if self.push is not None:
                if self.d.time < self.push[1]:
                    self.d.xfrc_applied[self.r.base, :3] = self.push[0]
                else:
                    self.d.xfrc_applied[self.r.base, :3] = 0
                    self.push = None
            mujoco.mj_step(self.m, self.d)
        return self.observe()

    def observe(self) -> Obs:
        d, r, rng = self.d, self.r, self.rng
        q = d.qpos[r.qadr].copy()
        qd = d.qvel[r.vadr].copy()
        # ankle backlash: motor-side encoder leads the link by b/2 in the loaded direction
        b = self.p.backlash
        if b > 0:
            q[self.ankle_idx] += 0.5 * b * np.tanh(self.tau_applied[self.ankle_idx] / 0.3)
        # joint zero offsets (deterministic bias, consumes no RNG -> old runs bit-identical)
        q += np.asarray(self.p.q0, float)
        q += rng.normal(0, 2e-4, 20)
        qd += rng.normal(0, 5e-3, 20)
        tau = self.tau_applied * (1 + rng.normal(0, 0.02, 20)) + rng.normal(0, 0.05, 20)
        quat = d.qpos[3:7] + rng.normal(0, 1e-3, 4)
        quat /= np.linalg.norm(quat)
        R = d.xmat[r.base].reshape(3, 3)
        gyro = d.qvel[3:6] + rng.normal(0, 5e-3, 3)        # free joint angvel is body frame
        lin_acc_w = d.qacc[0:3].copy()
        acc = R.T @ (lin_acc_w + np.array([0, 0, 9.81])) + rng.normal(0, 5e-2, 3)
        return Obs(d.time, q, qd, tau, d.qpos[0:3] + rng.normal(0, 2e-3, 3), quat,
                   d.qvel[0:3] + rng.normal(0, 0.02, 3), gyro, acc)

    # ground truth helpers (for evaluation only, never used by the controller)
    def fallen(self):
        """Standing-context failure (walking experiments).  NOT valid for get-up, where lying
        is the start state: use getup_state() / GetupMonitor instead (G0)."""
        R = self.d.xmat[self.r.base].reshape(3, 3)
        return self.d.qpos[2] < 0.22 or R[2, 2] < np.cos(np.deg2rad(50))

    def contact_links(self):
        """Ground truth: set of body names currently touching the floor (evaluation only)."""
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.r.floor_gid in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == self.r.floor_gid else c.geom1
                out.add(self.m.body(self.m.geom_bodyid[g]).name)
        return out

    def true_com(self):
        return self.d.subtree_com[0].copy()


# ============================================================================
# G0: get-up success / failure criterion.  For get-up, lying is the START, not a failure, so
# the standing-context fallen() is meaningless.  Classification of a configuration:
#   STAND      base z >= 0.85 * z_stand_nominal, trunk tilt <= 20 deg, only soles touch floor
#   LYING      trunk axis within 30 deg of horizontal and base z < 0.15 m
#   TRANSITION otherwise
# Get-up FAILURE (evaluated over an episode by GetupMonitor):
#   (a) regression: an unplanned return to LYING after the base rose above the sitting height,
#   (b) self-collision penetration > 5 mm, (c) timeout without reaching STAND (held 1 s).
Z_STAND_NOM = 0.376          # base z when standing at zero posture (soles on floor)


def getup_state(r: "PiPlus", d=None, contact_bodies=None):
    d = d or r.d
    R = d.xmat[r.base].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    z = d.qpos[2]
    feet_only = contact_bodies is None or (len(contact_bodies) > 0 and contact_bodies <= set(FEET))
    if z >= 0.85 * Z_STAND_NOM and tilt <= 20 and feet_only:
        return "STAND"
    if abs(90 - tilt) <= 30 and z < 0.15:
        return "LYING"
    return "TRANSITION"


class GetupMonitor:
    """Episode-level get-up success/failure (replaces fallen() in the get-up context)."""

    def __init__(self, z_sit=0.15, hold=1.0, timeout=60.0, pen_tol=0.005):
        self.z_sit, self.hold, self.timeout, self.pen_tol = z_sit, hold, timeout, pen_tol
        self.max_z, self.t_stand, self.status = 0.0, None, "RUNNING"

    def update(self, rob: "RealRobot"):
        if self.status != "RUNNING":
            return self.status
        d = rob.d
        st = getup_state(rob.r, d, rob.contact_links())
        self.max_z = max(self.max_z, d.qpos[2])
        selfpen = min([d.contact[i].dist for i in range(d.ncon)
                       if rob.r.floor_gid not in (d.contact[i].geom1, d.contact[i].geom2)], default=0.0)
        if selfpen < -self.pen_tol:
            self.status = "FAIL_SELFCOLLISION"
        elif st == "LYING" and self.max_z > self.z_sit:
            self.status = "FAIL_REGRESSION"
        elif st == "STAND":
            self.t_stand = d.time if self.t_stand is None else self.t_stand
            if d.time - self.t_stand >= self.hold:
                self.status = "SUCCESS"
        else:
            self.t_stand = None
        if self.status == "RUNNING" and d.time > self.timeout:
            self.status = "FAIL_TIMEOUT"
        return self.status


if __name__ == "__main__":
    r = PiPlus()
    print("nq nv nu", r.m.nq, r.m.nv, r.m.nu, "mass", r.total_mass0)
    r.reset()
    print("base z", r.d.qpos[2], "com", r.com())
    # passive drop test with PD hold
    rob = RealRobot(BodyParams(), seed=0)
    q0 = rob.r.default_q()
    for i in range(1000):
        o = rob.observe()
        tau = 60 * (q0 - o.q) - 1.5 * o.qd
        rob.step(tau)
    print("after 2s PD hold: base z %.3f fallen %s" % (rob.d.qpos[2], rob.fallen()), "ncon", rob.d.ncon)
