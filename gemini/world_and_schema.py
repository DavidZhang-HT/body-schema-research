"""Explicit Body Schema & True Physical Simulation World for Pi Plus (22-DOF URDF).

Separates:
1. Static Kinematic Prior (given by URDF: link lengths, joint axes, meshes).
2. True Physical World (m_true: unmodeled payload mass & CoM shift, joint damping/Coulomb
   friction, motor response delay, and compliant sole-ground contact).
3. Explicit Internal Body Schema (m_schema: initialized strictly from Nominal URDF,
   calibrated online via contact-nullspace proprioceptive prediction error + mj_setConst).
"""
from dataclasses import dataclass, field
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
SCENE_XML = ASSETS / 'scene.xml'
NOMINAL_QPOS_PATH = ASSETS / 'nominal_qpos.npy'
REF_WALK_NPZ = ASSETS / 'reference_walk' / 'motion.npz'
REF_WALK_META = ASSETS / 'reference_walk' / 'meta.json'

FEET = ['l_ankle_roll_link', 'r_ankle_roll_link']
SOLE_SITES = ['l_ankle_roll_link_sole', 'r_ankle_roll_link_sole']
LEG_JOINTS = [
    f'{s}_{j}_joint'
    for s in ['l', 'r']
    for j in ['hip_pitch', 'hip_roll', 'thigh', 'calf', 'ankle_pitch', 'ankle_roll']
]
UPPER_JOINTS = [
    f'{s}_{j}_joint'
    for s in ['l', 'r']
    for j in ['shoulder_pitch', 'shoulder_roll', 'upper_arm', 'elbow']
] + ['head_yaw_joint', 'head_pitch_joint']


@dataclass
class TrueWorldConfig:
    """Unmodeled real-world dynamics & contact discrepancies unknown to Nominal URDF."""
    delta_torso_mass_kg: float = 2.60
    delta_torso_com_m: tuple[float, float, float] = (-0.056, 0.026, 0.025)
    true_joint_damping: float = 0.32          # Nominal URDF scene assumes 0.04 Nms/rad
    true_joint_frictionloss: float = 0.14     # Nominal URDF scene assumes 0.00 Nm
    motor_lag_tau_s: float = 0.006            # 6 ms actuator torque response lag
    contact_solref: tuple[float, float] = (0.015, 0.95)  # Compliant contact vs nominal (0.008, 1.0)
    # Mid-walk unexpected external push impulse (for Stage 4 Surprise test)
    enable_surprise_push: bool = True
    push_duration_s: float = 0.18
    push_force_world_N: tuple[float, float, float] = (16.0, -15.0, 0.0)


def build_true_world(cfg: TrueWorldConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Instantiate the ground-truth MuJoCo physical simulation model."""
    m_true = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d_true = mujoco.MjData(m_true)
    torso_id = m_true.body('torso_link').id
    m_true.body_mass[torso_id] += cfg.delta_torso_mass_kg
    m_true.body_ipos[torso_id] += np.asarray(cfg.delta_torso_com_m, dtype=float)
    m_true.dof_damping[6:] = cfg.true_joint_damping
    m_true.dof_frictionloss[6:] = cfg.true_joint_frictionloss
    for gid in range(m_true.ngeom):
        if m_true.geom_contype[gid] != 0:
            m_true.geom_solref[gid, 0] = cfg.contact_solref[0]
            m_true.geom_solref[gid, 1] = cfg.contact_solref[1]
    mujoco.mj_setConst(m_true, d_true)
    return m_true, d_true


class ExplicitBodySchema:
    """Robot's internal forward/inverse dynamic & contact body schema.

    Initialized strictly from Nominal URDF. Updated online via proprioceptive
    prediction errors without ever reading simulator ground-truth parameters.
    """

    def __init__(self, scene_xml: Path = SCENE_XML):
        self.scene_xml = Path(scene_xml)
        self.m = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.d = mujoco.MjData(self.m)

        # Frozen nominal URDF prior & scratch regressor model
        self.m_prior = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.d_prior = mujoco.MjData(self.m_prior)
        self.m_reg = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.d_reg = mujoco.MjData(self.m_reg)

        self.base_id = self.m.body('base_link').id
        self.torso_id = self.m.body('torso_link').id
        self.sole_sites = [self.m.site(s).id for s in SOLE_SITES]

        self.nom_torso_mass = float(self.m_prior.body_mass[self.torso_id])
        self.nom_torso_com = self.m_prior.body_ipos[self.torso_id].copy()
        self.nom_damping = float(self.m_prior.dof_damping[6])

        # Leg DOF indices for decoupled proprioceptive contact wrench estimation
        self.left_leg_dofs = np.array([
            self.m.joint(f'l_{j}_joint').dofadr[0]
            for j in ['hip_pitch', 'hip_roll', 'thigh', 'calf', 'ankle_pitch', 'ankle_roll']
        ])
        self.right_leg_dofs = np.array([
            self.m.joint(f'r_{j}_joint').dofadr[0]
            for j in ['hip_pitch', 'hip_roll', 'thigh', 'calf', 'ankle_pitch', 'ankle_roll']
        ])

        # RLS state: theta = [dM_torso, dp_x, dp_y, dp_z, d_damping, f_coulomb]
        self.theta = np.zeros(6, dtype=float)
        self.P = np.diag([16.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        # Contact & single-support schema parameters calibrated across Stage 2 & Phase B
        self.ankle_roll_compliance = 0.0012   # rad / Nm (updated online)
        self.ankle_pitch_compliance = 0.0008  # rad / Nm (updated online)
        self.ss_z_droop_comp_m = 0.0          # swing-foot clearance / stance-sink compensation
        self.ss_lat_bias_comp_m = 0.0         # lateral CoM margin boost learned in Phase B
        self.swing_ff_gain = 1.0              # swing-leg inertia/damping feedforward scale

        # Causal low-pass filter buffers for synchronized proprioceptive SysID
        self.qvel_filt = np.zeros(self.m.nv)
        self.qacc_filt = np.zeros(self.m.nv)
        self.tau_filt = np.zeros(self.m.nu)
        self._filter_initialized = False

    @property
    def est_delta_mass(self) -> float:
        return float(np.clip(self.theta[0], -3.0, 6.0))

    @property
    def est_torso_mass(self) -> float:
        return self.nom_torso_mass + self.est_delta_mass

    @property
    def est_delta_torso_com(self) -> np.ndarray:
        est_c = (self.nom_torso_mass * self.nom_torso_com + self.theta[1:4]) / max(self.est_torso_mass, 1.0)
        return np.clip(est_c - self.nom_torso_com, -0.15, 0.15)

    @property
    def est_joint_damping(self) -> float:
        return self.nom_damping + float(np.clip(self.theta[4], 0.0, 1.5))

    @property
    def est_coulomb_friction(self) -> float:
        return float(np.clip(self.theta[5], 0.0, 0.8))

    @property
    def info_gain_trace(self) -> float:
        """ uncertainty trace Tr(P) -> decreases monotonically as information gain accumulates."""
        return float(np.trace(self.P))

    def sync_internal_mujoco_model(self, slew_alpha: float = 0.08) -> None:
        """Smoothly push calibrated inertial & friction parameters into internal MjModel m_schema."""
        cur_m = float(self.m.body_mass[self.torso_id])
        cur_c = self.m.body_ipos[self.torso_id].copy()
        cur_b = float(self.m.dof_damping[6])

        tgt_m = self.est_torso_mass
        tgt_c = self.nom_torso_com + self.est_delta_torso_com
        tgt_b = self.est_joint_damping

        self.m.body_mass[self.torso_id] = (1.0 - slew_alpha) * cur_m + slew_alpha * tgt_m
        self.m.body_ipos[self.torso_id] = (1.0 - slew_alpha) * cur_c + slew_alpha * tgt_c
        self.m.dof_damping[6:] = (1.0 - slew_alpha) * cur_b + slew_alpha * tgt_b
        # Note: Coulomb friction is compensated smoothly via tanh feedforward in controller
        # rather than rigid solver constraint in m_schema so QP remains smooth & fast.
        mujoco.mj_setConst(self.m, self.d)

    def whole_body_com(self, qpos: np.ndarray) -> np.ndarray:
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        return self.d.subtree_com[self.base_id].copy()

    def whole_body_com_and_vel(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)
        jc = np.zeros((3, self.m.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jc, self.base_id)
        return self.d.subtree_com[self.base_id].copy(), (jc @ qvel).copy()

    def update_proprioceptive_filters(self, qvel: np.ndarray, qacc_raw: np.ndarray, tau_cmd: np.ndarray, alpha: float = 0.18) -> None:
        if not self._filter_initialized:
            self.qvel_filt[:] = qvel
            self.qacc_filt[:] = qacc_raw
            self.tau_filt[:] = tau_cmd
            self._filter_initialized = True
            return
        self.qvel_filt = alpha * qvel + (1.0 - alpha) * self.qvel_filt
        self.qacc_filt = alpha * qacc_raw + (1.0 - alpha) * self.qacc_filt
        self.tau_filt = alpha * tau_cmd + (1.0 - alpha) * self.tau_filt

    def rls_update_double_support(self, qpos: np.ndarray, forgetting: float = 0.997) -> float:
        """Update [dM, dp_x, dp_y, dp_z, d_damping, f_coulomb] in the 16D contact-free nullspace."""
        qvel = self.qvel_filt
        qacc = self.qacc_filt
        tau = self.tau_filt

        self.d_prior.qpos[:] = qpos
        self.d_prior.qvel[:] = qvel
        self.d_prior.qacc[:] = qacc
        mujoco.mj_inverse(self.m_prior, self.d_prior)
        id_nom = self.d_prior.qfrc_inverse.copy()

        Jc_list = []
        for si in self.sole_sites:
            jp = np.zeros((3, self.m_prior.nv))
            jr = np.zeros((3, self.m_prior.nv))
            mujoco.mj_jacSite(self.m_prior, self.d_prior, jp, jr, si)
            Jc_list.extend([jp, jr])
        Jc = np.vstack(Jc_list)  # 12 x 28
        _, _, Vt = np.linalg.svd(Jc, full_matrices=True)
        Nc = Vt[12:, :]  # 16 x 28 orthonormal contact-free nullspace

        gen_tau = np.zeros(self.m_prior.nv)
        gen_tau[6:] = tau
        y = Nc @ (gen_tau - id_nom)

        cols = []
        eps_m = 1.0
        self.m_reg.body_mass[self.torso_id] = self.nom_torso_mass + eps_m
        self.m_reg.body_ipos[self.torso_id] = (self.nom_torso_mass * self.nom_torso_com) / (self.nom_torso_mass + eps_m)
        mujoco.mj_setConst(self.m_reg, self.d_reg)
        self.d_reg.qpos[:] = qpos; self.d_reg.qvel[:] = qvel; self.d_reg.qacc[:] = qacc
        mujoco.mj_inverse(self.m_reg, self.d_reg)
        cols.append(Nc @ ((self.d_reg.qfrc_inverse - id_nom) / eps_m))

        eps_p = 0.1
        for axis in range(3):
            dp = np.zeros(3); dp[axis] = eps_p
            self.m_reg.body_mass[self.torso_id] = self.nom_torso_mass
            self.m_reg.body_ipos[self.torso_id] = self.nom_torso_com + dp / self.nom_torso_mass
            mujoco.mj_setConst(self.m_reg, self.d_reg)
            self.d_reg.qpos[:] = qpos; self.d_reg.qvel[:] = qvel; self.d_reg.qacc[:] = qacc
            mujoco.mj_inverse(self.m_reg, self.d_reg)
            cols.append(Nc @ ((self.d_reg.qfrc_inverse - id_nom) / eps_p))

        v_j = np.zeros(self.m_prior.nv); v_j[6:] = qvel[6:]
        cols.append(Nc @ v_j)

        s_j = np.zeros(self.m_prior.nv); s_j[6:] = np.tanh(qvel[6:] / 0.05)
        cols.append(Nc @ s_j)

        Y = np.column_stack(cols)
        pred_err = y - Y @ self.theta
        R_cov = 0.25 * np.eye(Y.shape[0])
        K = self.P @ Y.T @ np.linalg.inv(Y @ self.P @ Y.T + R_cov)
        self.theta = self.theta + K @ pred_err
        self.P = (np.eye(6) - K @ Y) @ self.P / forgetting
        self.sync_internal_mujoco_model()

        # Also estimate ankle contact compliance from IMU base tilt vs FK sole orientation
        # In double support on flat ground, actual sole tilt in world frame reflects contact compliance:
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        sole_rolls = []
        sole_pitches = []
        for si in self.sole_sites:
            rpy_s = Rotation.from_matrix(self.d.site_xmat[si].reshape(3, 3)).as_euler('xyz')
            sole_rolls.append(abs(rpy_s[0]))
            sole_pitches.append(abs(rpy_s[1]))
        mean_roll_defl = float(np.mean(sole_rolls))
        if mean_roll_defl > 1e-4:
            # Smoothly track effective ankle/contact roll compliance scale
            self.ankle_roll_compliance = 0.98 * self.ankle_roll_compliance + 0.02 * np.clip(mean_roll_defl / 3.5, 0.0005, 0.006)

        return float(np.linalg.norm(pred_err))

    def infer_proprioceptive_foot_wrenches(self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray, tau_cmd: np.ndarray) -> np.ndarray:
        """Infer 6D world contact wrenches [w_L, w_R] purely from joint encoders, IMU, and motor torques."""
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        self.d.qacc[:] = qacc
        mujoco.mj_inverse(self.m, self.d)
        # qfrc_inverse = M*qacc + h - qfrc_passive (note damping is in qfrc_passive with minus sign in MuJoCo)
        # Add estimated Coulomb friction torque opposing motion:
        fric_c = np.zeros(self.m.nv)
        fric_c[6:] = self.est_coulomb_friction * np.tanh(qvel[6:] / 0.05)
        gen_tau = np.zeros(self.m.nv)
        gen_tau[6:] = tau_cmd
        # J_c^T w = M*qacc + h_bias + d*qvel + fric_c - S^T tau
        residual = self.d.qfrc_inverse + fric_c - gen_tau

        wrenches = np.zeros((2, 6))
        l_char = 0.045  # characteristic foot moment lever arm (m) for unit-consistent conditioning
        for ee, (si, dofs) in enumerate(zip(self.sole_sites, [self.left_leg_dofs, self.right_leg_dofs])):
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, si)
            # Scale angular Jacobian rows by l_char so all 6 rows have units of meters
            J_scaled = np.vstack([jp[:, dofs], l_char * jr[:, dofs]])  # 6 x 6
            JT = J_scaled.T
            w_s = np.linalg.solve(JT.T @ JT + 1.2e-3 * np.eye(6), JT.T @ residual[dofs])
            w = np.r_[w_s[:3], l_char * w_s[3:]]
            # Enforce unilateral non-adhesive ground contact & Coulomb friction cone:
            # an airborne swing foot (Fz < 6 N) cannot transmit phantom ground shear forces!
            fz = float(max(0.0, w[2]))
            if fz < 6.0:
                w[:] = 0.0
            else:
                w[2] = fz
                w[0] = float(np.clip(w[0], -0.85 * fz, 0.85 * fz))
                w[1] = float(np.clip(w[1], -0.85 * fz, 0.85 * fz))
                w[3:] = np.clip(w[3:], -0.07 * fz, 0.07 * fz)
            wrenches[ee] = w
        return wrenches

    def infer_proprioceptive_external_base_force(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        tau_cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate foot contact wrenches from unexpected external body push using
        floating-base Newton-Euler residual vs leg-joint-inferred contact wrenches!
        Returns: (foot_wrenches_2x6, ext_body_force_world_3d).
        """
        wrenches = self.infer_proprioceptive_foot_wrenches(qpos, qvel, qacc, tau_cmd)
        # Note: self.d already ran mj_inverse(self.m, self.d) inside infer_proprioceptive_foot_wrenches
        # In MuJoCo, free-joint rows 0:3 of qfrc_inverse are linear generalized forces in WORLD frame!
        J_feet_base_T = np.zeros((3, 12))
        for ee, si in enumerate(self.sole_sites):
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, si)
            J_feet_base_T[:, 6 * ee : 6 * (ee + 1)] = np.vstack([jp[:, :3], jr[:, :3]]).T
        f_feet_on_base = J_feet_base_T @ wrenches.ravel()
        f_ext_world = self.d.qfrc_inverse[:3] - f_feet_on_base
        return wrenches, f_ext_world
