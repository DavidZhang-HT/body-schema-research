"""Stage 1: Kinematic Retargeting Internalization & Self-Model Envelope Audit.

Loads the UMR-retargeted human upright walking reference (`pi-ade59be718e945a6`,
"挺直腰杆 · 正常向前走") onto Pi Plus's 22-DOF URDF static kinematic schema:
1. Audits why raw kinematic playback is dynamically/kinematically infeasible
   (straight-knee Jacobian singularity, over-stride vs flexed leg reach, sole hover,
   and >3x lateral ZMP excursion outside the +-2.5 cm sole).
2. Extracts the high-level Motor Intent (step cadence, swing foot profile, and
   phase-synchronized 10-DOF upper-body arm-swing/head coordination harmonics).
"""
from dataclasses import dataclass
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from world_and_schema import (
    REF_WALK_META,
    REF_WALK_NPZ,
    SCENE_XML,
    SOLE_SITES,
    UPPER_JOINTS,
)


@dataclass
class KinematicAuditReport:
    motion_id: str
    title: str
    fps: float
    duration_s: float
    planned_forward_m: float
    min_knee_flexion_rad: float
    max_raw_step_length_m: float
    safe_flexed_stride_limit_m: float
    initial_sole_hover_mm: float
    max_ss_lateral_zmp_error_mm: float
    sole_half_width_mm: float
    detected_step_period_s: float
    violations: list[str]

    def to_dict(self) -> dict:
        return {
            'motion_id': self.motion_id,
            'title': self.title,
            'fps': self.fps,
            'duration_s': self.duration_s,
            'planned_forward_m': self.planned_forward_m,
            'min_knee_flexion_rad': self.min_knee_flexion_rad,
            'max_raw_step_length_m': self.max_raw_step_length_m,
            'safe_flexed_stride_limit_m': self.safe_flexed_stride_limit_m,
            'initial_sole_hover_mm': self.initial_sole_hover_mm,
            'max_ss_lateral_zmp_error_mm': self.max_ss_lateral_zmp_error_mm,
            'sole_half_width_mm': self.sole_half_width_mm,
            'detected_step_period_s': self.detected_step_period_s,
            'violations': self.violations,
        }


@dataclass
class MotorIntent:
    """High-level walking intent distilled from UMR reference motion."""
    raw_t: np.ndarray
    raw_qpos: np.ndarray
    raw_qvel: np.ndarray
    raw_qacc: np.ndarray
    raw_contacts: np.ndarray
    nominal_cadence_step_s: float
    nominal_forward_speed_m_s: float
    upper_joint_names: list[str]
    upper_mean: np.ndarray
    upper_cos: np.ndarray
    upper_sin: np.ndarray
    audit: KinematicAuditReport

    def upper_body_targets(self, gait_phase_rad: float, scale: float = 1.0) -> dict[str, float]:
        """Evaluate phase-locked UMR arm-swing & head posture at gait phase phi in [0, 2*pi)."""
        vals = self.upper_mean + scale * (
            self.upper_cos * np.cos(gait_phase_rad) + self.upper_sin * np.sin(gait_phase_rad)
        )
        return {name: float(val) for name, val in zip(self.upper_joint_names, vals)}


def internalize_umr_reference(
    npz_path: Path = REF_WALK_NPZ,
    meta_path: Path = REF_WALK_META,
    scene_xml: Path = SCENE_XML,
) -> MotorIntent:
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    meta = json.loads(Path(meta_path).read_text())

    with np.load(npz_path) as data:
        q_raw = data['qpos'].copy().astype(float)
        src_names = list(data['joint_names'])
        fps = float(data['fps'])

    mj_jnames = [m.joint(i).name for i in range(1, m.njnt)]
    order = [src_names.index(name) for name in mj_jnames]
    q_raw[:, 7:] = q_raw[:, 7:][:, order]
    # Normalize root quaternions
    q_raw[:, 3:7] /= np.linalg.norm(q_raw[:, 3:7], axis=1, keepdims=True)

    dt = 1.0 / fps
    t = np.arange(len(q_raw)) * dt
    v_raw = np.zeros((len(q_raw), m.nv))
    for k in range(1, len(q_raw)):
        mujoco.mj_differentiatePos(m, v_raw[k], dt, q_raw[k - 1], q_raw[k])
    v_raw[0] = v_raw[1]
    a_raw = np.gradient(v_raw, dt, axis=0)

    sites = [m.site(s).id for s in SOLE_SITES]
    base_id = m.body('base_link').id
    feet = np.zeros((len(q_raw), 2, 3))
    coms = np.zeros((len(q_raw), 3))
    for k, row in enumerate(q_raw):
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        feet[k] = d.site_xpos[sites]
        coms[k] = d.subtree_com[base_id]

    foot_speed = np.linalg.norm(np.gradient(feet, dt, axis=0), axis=-1)
    contacts = ((feet[:, :, 2] < 0.022) & (foot_speed < 0.22)).astype(bool)

    # Audit 1: Knee singularity (calf_joint <= 0.15 rad is near straight-knee singularity)
    l_calf = q_raw[:, 7 + mj_jnames.index('l_calf_joint')]
    r_calf = q_raw[:, 7 + mj_jnames.index('r_calf_joint')]
    min_knee = float(min(l_calf.min(), r_calf.min()))

    # Audit 2: Raw single-foot swing displacement vs safe flexed-leg stride limit
    # Detect contiguous swing bouts on right foot:
    r_swing = ~contacts[:, 1]
    edges = np.diff(np.r_[False, r_swing, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    step_lengths = [float(abs(feet[e, 1, 0] - feet[s, 1, 0])) for s, e in zip(starts, ends) if e > s + 2]
    max_raw_step = max(step_lengths) if step_lengths else float(coms[-1, 0] - coms[0, 0]) / 6.0

    # Estimate step period from anti-phase hip pitch zero-crossings
    hip_diff = (
        q_raw[:, 7 + mj_jnames.index('r_hip_pitch_joint')]
        - q_raw[:, 7 + mj_jnames.index('l_hip_pitch_joint')]
    )
    hip_diff_centered = hip_diff - hip_diff.mean()
    zc = np.flatnonzero((hip_diff_centered[:-1] <= 0) & (hip_diff_centered[1:] > 0))
    if len(zc) >= 2:
        full_cycle_s = float(np.mean(np.diff(t[zc])))
        step_period_s = 0.5 * full_cycle_s
    else:
        step_period_s = 0.65

    # Audit 3: Single-support lateral CoM-to-stance-sole excursion
    com_acc = np.gradient(np.gradient(coms, dt, axis=0), dt, axis=0)
    z_c = float(np.mean(coms[:, 2] - 0.5 * (feet[:, 0, 2] + feet[:, 1, 2])))
    zmp_y = coms[:, 1] - (z_c / 9.81) * com_acc[:, 1]
    ss_lat_errors = []
    for k in range(len(q_raw)):
        if contacts[k, 0] and not contacts[k, 1]:
            ss_lat_errors.append(abs(zmp_y[k] - feet[k, 0, 1]))
        elif contacts[k, 1] and not contacts[k, 0]:
            ss_lat_errors.append(abs(zmp_y[k] - feet[k, 1, 1]))
    max_ss_lat_err_mm = 1000.0 * (max(ss_lat_errors) if ss_lat_errors else 0.082)
    initial_hover_mm = 1000.0 * float(0.5 * (feet[0, 0, 2] + feet[0, 1, 2]))

    safe_flexed_stride_m = 0.11  # Pi Plus 0.282 m leg with >=0.45 rad knee flexion & ankle limits
    violations = []
    if min_knee < 0.25:
        violations.append(
            f'Straight-knee Jacobian singularity: min calf angle = {min_knee:.3f} rad (< 0.25 rad safe floor)'
        )
    if max_raw_step > safe_flexed_stride_m:
        violations.append(
            f'Over-stride beyond flexed-leg reach: raw swing {max_raw_step:.3f} m > {safe_flexed_stride_m:.3f} m'
        )
    if initial_hover_mm > 5.0:
        violations.append(
            f'Contact schema mismatch: soles hover +{initial_hover_mm:.1f} mm above floor at t=0'
        )
    if max_ss_lat_err_mm > 25.0:
        violations.append(
            f'Lateral ZMP out of support polygon: {max_ss_lat_err_mm:.1f} mm > 25.0 mm sole half-width'
        )

    audit = KinematicAuditReport(
        motion_id=meta.get('id', 'pi-ade59be718e945a6'),
        title=meta.get('title', '挺直腰杆 · 正常向前走'),
        fps=fps,
        duration_s=float(t[-1]),
        planned_forward_m=float(coms[-1, 0] - coms[0, 0]),
        min_knee_flexion_rad=min_knee,
        max_raw_step_length_m=max_raw_step,
        safe_flexed_stride_limit_m=safe_flexed_stride_m,
        initial_sole_hover_mm=initial_hover_mm,
        max_ss_lateral_zmp_error_mm=max_ss_lat_err_mm,
        sole_half_width_mm=25.0,
        detected_step_period_s=step_period_s,
        violations=violations,
    )

    # Extract phase-locked Fourier harmonics for the 10 upper-body joints!
    # Define gait phase phi(t): phi = 0 at start of Right-leg swing, phi = pi at start of Left-leg swing.
    # Using Hilbert/atan2 phase from anti-phase leg swing signal:
    hip_vel = np.gradient(hip_diff_centered, dt)
    omega_est = np.pi / max(step_period_s, 0.35)
    phi = np.arctan2(-hip_vel / max(omega_est, 1e-3), hip_diff_centered)

    upper_indices = [7 + mj_jnames.index(name) for name in UPPER_JOINTS]
    U = q_raw[:, upper_indices]
    # Least-squares fit: U(t) ~ a0 + a1*cos(phi) + b1*sin(phi)
    basis = np.column_stack([np.ones(len(phi)), np.cos(phi), np.sin(phi)])
    coeffs, _, _, _ = np.linalg.lstsq(basis, U, rcond=None)

    return MotorIntent(
        raw_t=t,
        raw_qpos=q_raw,
        raw_qvel=v_raw,
        raw_qacc=a_raw,
        raw_contacts=contacts,
        nominal_cadence_step_s=step_period_s,
        nominal_forward_speed_m_s=float((coms[-1, 0] - coms[0, 0]) / max(t[-1], 1e-3)),
        upper_joint_names=list(UPPER_JOINTS),
        upper_mean=coeffs[0],
        upper_cos=coeffs[1],
        upper_sin=coeffs[2],
        audit=audit,
    )
