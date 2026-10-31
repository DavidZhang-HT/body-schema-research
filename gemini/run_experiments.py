"""Run controlled MuJoCo physical simulation experiments on Pi Plus 22-DOF URDF:

- Exp 1 (`exp1_raw_umr_replay`):
  Direct torque-controlled tracking of the UMR human walking reference
  (`pi-ade59be718e945a6`, "挺直腰杆 · 正常向前走") without Body Schema projection.
- Exp 2 (`exp2_uncalibrated_schema`):
  Ablation — Stage 3 Dynamic Feasibility Projection using ONLY the Nominal URDF
  (skipping Stage 2 Active Postural Exploration & Stage 4 Surprise adaptation)
  inside the True Physical World (+2.4 kg off-center torso payload, higher joint
  damping/Coulomb friction, 6 ms motor lag, compliant contact, mid-walk push).
- Exp 3 (`exp3_full_body_schema`):
  Proposed 4-Stage Curriculum Body Schema Self-Exploration in ONE continuous,
  uninterrupted MuJoCo episode (zero qpos resets, zero base external assistance):
    [0.0 - 3.0 s] Stage 2 Double-Support Active Postural Exploration (ASID)
    [3.0 - 6.1 s] Curriculum Phase B: Stepping in Place (single-support calibration)
    [6.1 - 13.4 s] Stage 3 + 4: UMR-Guided Forward Walking + Surprise-Driven Reflex
"""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from stage1_kinematic_internalizer import MotorIntent, internalize_umr_reference
from stage3_dynamic_projector import DynamicFeasibilityProjector, ProjectedSegmentPlan, quintic_blend
from stage4_surprise_controller import SurpriseDrivenMonitor, WholeBodySchemaQPController
from world_and_schema import (
    ASSETS,
    ExplicitBodySchema,
    FEET,
    NOMINAL_QPOS_PATH,
    ROOT,
    TrueWorldConfig,
    build_true_world,
)

OUT_DIR = ROOT / 'outputs'


def measure_ground_contacts(m_true: mujoco.MjModel, d_true: mujoco.MjData) -> tuple[bool, np.ndarray]:
    """Check non-foot ground collision and sum true normal contact forces on [L, R] feet."""
    floor_id = m_true.geom('floor').id
    nonfoot = False
    foot_normals = np.zeros(2)
    for ci in range(d_true.ncon):
        c = d_true.contact[ci]
        if floor_id not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor_id else c.geom1
        bname = m_true.body(m_true.geom_bodyid[other]).name
        if bname not in FEET:
            nonfoot = True
        else:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m_true, d_true, ci, wrench)
            foot_normals[FEET.index(bname)] += max(0.0, float(wrench[0]))
    return nonfoot, foot_normals


def summarize_run(
    name: str,
    cfg: TrueWorldConfig,
    m_true: mujoco.MjModel,
    schema: ExplicitBodySchema,
    log: dict[str, np.ndarray],
    failed_reason: str | None,
    qp_fallbacks: int,
    extra_meta: dict,
) -> dict:
    qpos = log['qpos']
    torques = log['torque']
    speeds = log['joint_speed']
    tilt = log['tilt']
    t = log['t']

    model_info = json.loads((ASSETS / 'model.json').read_text())
    vel_limits = np.asarray(model_info['joint_velocity_limits'], dtype=float)
    ctrl_limits = m_true.actuator_ctrlrange[:, 1]

    forward_m = float(qpos[-1, 0] - qpos[0, 0])
    lateral_drift_m = float(qpos[-1, 1] - qpos[0, 1])
    max_tilt = float(np.abs(tilt).max()) if len(tilt) else 0.0
    min_height = float(qpos[:, 2].min()) if len(qpos) else 0.0
    max_torque_Nm = float(np.abs(torques).max()) if len(torques) else 0.0
    max_torque_ratio = float((np.abs(torques) / ctrl_limits).max()) if len(torques) else 0.0
    max_vel_ratio = float((speeds / vel_limits).max()) if len(speeds) else 0.0
    min_jnt_margin = float(
        np.minimum(
            qpos[:, 7:] - m_true.jnt_range[1:, 0],
            m_true.jnt_range[1:, 1] - qpos[:, 7:],
        ).min()
    )
    nonfoot = bool(np.any(log['nonfoot_contact']))

    # Allow 8 mrad (0.45 deg) for MuJoCo's soft joint-limit solver compliance
    passed = bool(
        failed_reason is None
        and not nonfoot
        and min_height > 0.22
        and max_tilt < 0.40
        and max_torque_ratio <= 1.0001
        and max_vel_ratio < 1.0
        and min_jnt_margin > -0.008
    )

    return {
        'experiment': name,
        'passed': passed,
        'completed_without_fall': failed_reason is None and not nonfoot,
        'failure_reason': failed_reason,
        'duration_s': round(float(t[-1]), 3),
        'forward_distance_m': round(forward_m, 4),
        'lateral_drift_m': round(lateral_drift_m, 4),
        'min_base_height_m': round(min_height, 4),
        'max_roll_pitch_rad': round(max_tilt, 4),
        'max_roll_pitch_deg': round(float(np.degrees(max_tilt)), 2),
        'max_motor_torque_Nm': round(max_torque_Nm, 3),
        'max_motor_torque_limit_ratio': round(max_torque_ratio, 4),
        'max_velocity_limit_ratio': round(max_vel_ratio, 4),
        'min_joint_limit_margin_rad': round(min_jnt_margin, 4),
        'nonfoot_ground_contact': nonfoot,
        'qp_fallback_steps': int(qp_fallbacks),
        'base_externally_stabilized': False,
        'true_world_discrepancies': {
            'delta_torso_mass_kg': cfg.delta_torso_mass_kg,
            'delta_torso_com_m': list(cfg.delta_torso_com_m),
            'true_joint_damping': cfg.true_joint_damping,
            'true_joint_frictionloss': cfg.true_joint_frictionloss,
            'motor_lag_tau_s': cfg.motor_lag_tau_s,
            'surprise_push_enabled': cfg.enable_surprise_push,
            'surprise_push_force_N': list(cfg.push_force_world_N) if cfg.enable_surprise_push else [0, 0, 0],
        },
        'final_body_schema_estimates': {
            'est_delta_mass_kg': round(schema.est_delta_mass, 4),
            'est_delta_torso_com_m': [round(float(x), 4) for x in schema.est_delta_torso_com],
            'est_joint_damping': round(schema.est_joint_damping, 4),
            'est_coulomb_friction': round(schema.est_coulomb_friction, 4),
            'ss_z_droop_comp_mm': round(1000.0 * schema.ss_z_droop_comp_m, 2),
            'ss_lat_bias_comp_mm': round(1000.0 * schema.ss_lat_bias_comp_m, 2),
            'swing_ff_gain': round(schema.swing_ff_gain, 4),
        },
        **extra_meta,
    }


def run_exp1_raw_umr_replay(intent: MotorIntent, cfg: TrueWorldConfig) -> dict:
    """Exp 1: Direct torque-driven replay of unmodified UMR kinematic walking motion."""
    m_true, d_true = build_true_world(cfg)
    schema = ExplicitBodySchema()
    ctrl = WholeBodySchemaQPController(schema)

    q_umr = intent.raw_qpos.copy()
    v_umr = intent.raw_qvel.copy()
    a_umr = intent.raw_qacc.copy()
    t_umr = intent.raw_t.copy()
    c_umr = intent.raw_contacts.copy()

    d_true.qpos[:] = q_umr[0]
    d_true.qvel[:] = 0.0
    mujoco.mj_forward(m_true, d_true)

    dt = m_true.opt.timestep
    lead_s = 0.40
    total_s = lead_s + float(t_umr[-1]) + 0.60
    steps = int(total_s / dt)

    tau_applied = np.zeros(m_true.nu)
    logs = {k: [] for k in [
        't', 'qpos', 'ref_qpos', 'torque', 'joint_speed', 'tilt',
        'foot_force_true', 'nonfoot_contact', 'true_com', 'schema_com', 'phase_id'
    ]}
    failed = None
    base_id = m_true.body('base_link').id

    for k in range(steps):
        t_cur = d_true.time
        phase_t = float(np.clip(t_cur - lead_s, 0.0, t_umr[-1]))
        q_ref = np.array([np.interp(phase_t, t_umr, q_umr[:, i]) for i in range(m_true.nq)])
        q_ref[3:7] /= np.linalg.norm(q_ref[3:7])
        v_ref = np.array([np.interp(phase_t, t_umr, v_umr[:, i]) for i in range(m_true.nv)])
        a_ref = np.array([np.interp(phase_t, t_umr, a_umr[:, i]) for i in range(m_true.nv)])
        idx = min(int(np.searchsorted(t_umr, phase_t, side='right')) - 1, len(t_umr) - 1)
        idx = max(idx, 0)
        contacts = [bool(c_umr[idx, 0]), bool(c_umr[idx, 1])]
        if t_cur < lead_s or t_cur > lead_s + t_umr[-1]:
            v_ref[:] = 0.0
            a_ref[:] = 0.0
            contacts = [True, True]

        tau_cmd = ctrl.compute_torque(d_true.qpos, d_true.qvel, q_ref, v_ref, a_ref, contacts)
        tau_applied += (dt / cfg.motor_lag_tau_s) * (tau_cmd - tau_applied)
        d_true.ctrl[:] = tau_applied

        nonfoot, f_true = measure_ground_contacts(m_true, d_true)
        rpy = Rotation.from_quat(d_true.qpos[[4, 5, 6, 3]]).as_euler('xyz')

        logs['t'].append(t_cur)
        logs['qpos'].append(d_true.qpos.copy())
        logs['ref_qpos'].append(q_ref.copy())
        logs['torque'].append(tau_cmd.copy())
        logs['joint_speed'].append(np.abs(d_true.qvel[6:]).copy())
        logs['tilt'].append(rpy[:2].copy())
        logs['foot_force_true'].append(f_true)
        logs['nonfoot_contact'].append(nonfoot)
        logs['true_com'].append(d_true.subtree_com[base_id].copy())
        logs['schema_com'].append(schema.whole_body_com(d_true.qpos))
        logs['phase_id'].append(0)

        mujoco.mj_step(m_true, d_true)
        if d_true.qpos[2] < 0.20 or nonfoot or not np.isfinite(d_true.qpos).all():
            failed = f'Fall / non-foot ground impact at t={d_true.time:.3f}s'
            # Record 0.25s post-fall impact frames so video clearly shows the fall
            for _ in range(int(0.25 / dt)):
                d_true.ctrl[:] = 0.0
                mujoco.mj_step(m_true, d_true)
                _, f_post = measure_ground_contacts(m_true, d_true)
                rpy_p = Rotation.from_quat(d_true.qpos[[4, 5, 6, 3]]).as_euler('xyz')
                logs['t'].append(d_true.time)
                logs['qpos'].append(d_true.qpos.copy())
                logs['ref_qpos'].append(q_ref.copy())
                logs['torque'].append(np.zeros(m_true.nu))
                logs['joint_speed'].append(np.abs(d_true.qvel[6:]).copy())
                logs['tilt'].append(rpy_p[:2].copy())
                logs['foot_force_true'].append(f_post)
                logs['nonfoot_contact'].append(True)
                logs['true_com'].append(d_true.subtree_com[base_id].copy())
                logs['schema_com'].append(schema.whole_body_com(d_true.qpos))
                logs['phase_id'].append(0)
            break

    np_logs = {k: np.asarray(v) for k, v in logs.items()}
    np.savez_compressed(OUT_DIR / 'exp1_raw_umr_replay.npz', **np_logs)
    report = summarize_run(
        'exp1_raw_umr_replay', cfg, m_true, schema, np_logs, failed, ctrl.qp_fallbacks,
        {'description': 'Direct torque-limited tracking of UMR kinematic walk reference'}
    )
    (OUT_DIR / 'exp1_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def evaluate_plan_at_time(
    plan: ProjectedSegmentPlan,
    t_local: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], np.ndarray, float, float, int | None, float]:
    """Interpolate CoM, ZMP, feet, contacts, torso RPY, and UMR phase at t_local."""
    tg = plan.t
    t_clamped = float(np.clip(t_local, tg[0], tg[-1]))
    idx = min(int(np.searchsorted(tg, t_clamped, side='right')) - 1, len(tg) - 1)
    idx = max(idx, 0)

    def interp_arr(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(len(tg), -1)
        out = np.array([np.interp(t_clamped, tg, flat[:, c]) for c in range(flat.shape[1])])
        return out.reshape(arr.shape[1:])

    com_p = interp_arr(plan.com_pos)
    com_v = interp_arr(plan.com_vel)
    com_a = interp_arr(plan.com_acc)
    zmp_p = interp_arr(plan.zmp_ref)
    feet_p = interp_arr(plan.foot_pos)
    contacts = [bool(plan.contacts[idx, 0]), bool(plan.contacts[idx, 1])]
    rpy_p = interp_arr(plan.torso_rpy)
    phi = float(np.interp(t_clamped, tg, plan.gait_phase))
    ascale = float(np.interp(t_clamped, tg, plan.arm_scale))

    swing_leg = None
    swing_prog = 0.0
    for st in plan.steps:
        if st.t_start <= t_clamped <= st.t_end:
            swing_leg = st.swing_leg
            swing_prog = float((t_clamped - st.t_start) / max(st.t_end - st.t_start, 1e-6))
            break

    if t_local >= tg[-1]:
        com_v[:] = 0.0
        com_a[:] = 0.0
        contacts = [True, True]
        swing_leg = None

    return com_p, com_v, com_a, zmp_p, feet_p, contacts, rpy_p, phi, ascale, swing_leg, swing_prog


def run_curriculum_or_ablation(
    intent: MotorIntent,
    cfg: TrueWorldConfig,
    enable_body_schema_learning: bool,
) -> dict:
    """Run either Full 4-Stage Body Schema Curriculum (True) or Uncalibrated Nominal Ablation (False)."""
    exp_name = 'exp3_full_body_schema' if enable_body_schema_learning else 'exp2_uncalibrated_schema'
    m_true, d_true = build_true_world(cfg)
    schema = ExplicitBodySchema()
    q_nom = np.load(NOMINAL_QPOS_PATH)

    projector = DynamicFeasibilityProjector(schema, q_nom)
    ctrl = WholeBodySchemaQPController(schema)
    monitor = SurpriseDrivenMonitor(schema)

    d_true.qpos[:] = q_nom
    d_true.qvel[:] = 0.0
    mujoco.mj_forward(m_true, d_true)

    dt = m_true.opt.timestep
    torso_id = m_true.body('torso_link').id
    base_id = m_true.body('base_link').id

    # Curriculum schedule:
    # Note: dx is swing foot target ahead of STANCE foot, so total foot swing travel is
    # dx_prev + dx_cur. Setting dx = 0.050 m keeps total swing travel = 0.100 m <= 0.110 m
    # flexed-leg workspace limit (10 steps => ~0.46 m clean forward walk!).
    T_stage2 = 3.00 if enable_body_schema_learning else 0.40
    inplace_steps = [0.0, 0.0, 0.0, 0.0] if enable_body_schema_learning else []
    forward_steps = [0.034, 0.047, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.047, 0.00]

    # Initial double-support geometry in schema
    schema.d.qpos[:] = q_nom
    mujoco.mj_forward(schema.m, schema.d)
    init_feet = schema.d.site_xpos[schema.sole_sites].copy()
    init_feet[:, 2] = 0.0
    init_com = np.array([init_feet[:, 0].mean(), 0.0, schema.d.subtree_com[base_id][2]])

    q_ref = q_nom.copy()
    prev_q_ref = q_nom.copy()
    v_ref_filt = np.zeros(m_true.nv)
    a_ref_filt = np.zeros(m_true.nv)
    prev_qvel_meas = np.zeros(m_true.nv)
    tau_applied = np.zeros(m_true.nu)

    phase_b_plan: ProjectedSegmentPlan | None = None
    phase_b_start_t = T_stage2
    phase_c_plan: ProjectedSegmentPlan | None = None
    phase_c_start_t: float | None = None

    # Determine push window relative to forward walk start so both Exp 2 and Exp 3
    # experience the exact same physical push at mid-walk (2.98s into forward walking, during Right swing Step 4)!
    push_offset_in_walk_s = 2.98

    logs = {k: [] for k in [
        't', 'qpos', 'ref_qpos', 'torque', 'joint_speed', 'tilt',
        'foot_force_true', 'foot_force_inferred', 'nonfoot_contact',
        'true_com', 'schema_com', 'com_ref', 'zmp_ref',
        'est_dm', 'est_dc', 'est_damping', 'est_friction', 'info_trace',
        'dcm_surprise', 'swing_droop_mm', 'reflex_offset_xy', 'push_active', 'phase_id'
    ]}

    failed = None
    max_total_time = 15.0
    max_steps = int(max_total_time / dt)

    for k in range(max_steps):
        t_cur = d_true.time
        prev_q_ref[:] = q_ref
        push_active = False

        # =====================================================================
        # PHASE A: Stage 2 Active Postural Exploration (or brief 0.4s hold in Exp 2)
        # =====================================================================
        if t_cur < T_stage2:
            phase_id = 1
            phase_name = 'STAGE2_ACTIVE_SYSID'
            if enable_body_schema_learning:
                # Smooth probing window finishing cleanly by t = 2.65s, then 0.35s smooth
                # settle into calibrated balanced posture!
                T_probe = 2.65
                if t_cur <= T_probe:
                    w = np.sin(np.pi * (t_cur / T_probe)) ** 2
                else:
                    w = 0.0
                # Recompute balanced CoM center over planted feet as schema updates online!
                # Probe around calibrated standing CoM height, then smoothly settle to athletic
                # flexed-knee walking CoM height (0.310 m) for wide lateral push-recovery reach!
                auto_pitch, auto_roll, _ = projector.compute_schema_posture_adaptation()
                settle_s = quintic_blend(np.clip((t_cur - 1.5) / 1.4, 0.0, 1.0))
                cal_stand_z = float(schema.whole_body_com(q_nom)[2])
                stage2_base_z = (1.0 - settle_s) * cal_stand_z + settle_s * 0.310

                target_com = np.array([init_com[0], init_com[1], stage2_base_z]) + w * np.array([
                    0.012 * np.sin(2.0 * np.pi * 0.72 * t_cur),
                    0.022 * np.sin(2.0 * np.pi * 0.56 * t_cur),
                    -0.012 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 1.45 * t_cur)),
                ])
                target_rpy = w * np.array([
                    0.10 * np.sin(2.0 * np.pi * 0.86 * t_cur + 0.3),
                    0.12 * (np.sin(2.0 * np.pi * 0.66 * t_cur) + 0.45 * np.sin(2.0 * np.pi * 1.52 * t_cur)),
                    0.07 * np.sin(2.0 * np.pi * 0.95 * t_cur),
                ]) + settle_s * np.array([auto_roll, auto_pitch, 0.0])

                upper_probe = intent.upper_body_targets(0.0, scale=0.0)
                upper_probe['l_shoulder_pitch_joint'] += w * 0.55 * np.sin(2.0 * np.pi * 2.1 * t_cur)
                upper_probe['r_shoulder_pitch_joint'] -= w * 0.55 * np.sin(2.0 * np.pi * 2.1 * t_cur)
                upper_probe['l_elbow_joint'] += w * 0.42 * np.sin(2.0 * np.pi * 2.6 * t_cur)
                upper_probe['r_elbow_joint'] += w * 0.42 * np.cos(2.0 * np.pi * 2.6 * t_cur)
            else:
                target_com = init_com.copy()
                target_rpy = np.zeros(3)
                upper_probe = intent.upper_body_targets(0.0, scale=0.0)

            q_ref = projector.solve_ik_frame(q_ref, target_com, init_feet, target_rpy, upper_probe, n_iters=4)
            com_ref = target_com
            com_vel_ref = None
            com_acc_ref = None
            zmp_plan = init_com[:2]
            contacts = [True, True]
            snap_dcm = 0.0
            snap_droop = 0.0
            snap_fz = np.zeros(2)
            snap_reflex = np.zeros(2)

        # =====================================================================
        # PHASE B: Curriculum Stepping in Place (Single-Support Contact Calibration)
        # =====================================================================
        elif enable_body_schema_learning and (
            phase_c_start_t is None or t_cur < phase_c_start_t
        ):
            if phase_b_plan is None:
                phase_b_plan = projector.build_step_plan(
                    q_start=q_ref,
                    intent=intent,
                    step_displacements_x=inplace_steps,
                    first_swing_leg=1,
                    t_lead_ds=0.30,
                    t_swing=0.34,
                    t_ds=0.22,
                    t_tail_ds=0.34,
                    base_clearance_m=0.020,
                )
                phase_c_start_t = phase_b_start_t + float(phase_b_plan.t[-1])

            t_loc = t_cur - phase_b_start_t
            if t_cur < phase_c_start_t:
                phase_id = 2
                phase_name = 'STEP_IN_PLACE'
                (
                    com_ref, com_vel_ref, com_acc_ref, zmp_plan, feet_plan,
                    contacts_plan, rpy_plan, phi, ascale, sw_leg, sw_prog
                ) = evaluate_plan_at_time(phase_b_plan, t_loc)

                contacts, adapted_feet, snap = monitor.update(
                    t_cur, d_true.qpos, d_true.qvel, tau_applied, phase_b_plan.omega0,
                    com_ref, com_vel_ref, zmp_plan, feet_plan, contacts_plan,
                    sw_leg, sw_prog, phase_name,
                )
                upper_t = intent.upper_body_targets(phi, scale=0.30)
                q_ref = projector.solve_ik_frame(q_ref, com_ref, adapted_feet, rpy_plan, upper_t, n_iters=5)
                snap_dcm = snap.dcm_surprise_m_s
                snap_droop = snap.swing_z_droop_mm
                snap_fz = np.array([snap.inferred_fz_L_N, snap.inferred_fz_R_N])
                snap_reflex = snap.reflex_foot_offset_xy
            else:
                phase_id = 3

        # =====================================================================
        # PHASE C: Stage 3 + Stage 4 UMR-Guided Forward Walk + Surprise Reflex
        # =====================================================================
        if (not enable_body_schema_learning and t_cur >= T_stage2) or (
            enable_body_schema_learning and phase_c_start_t is not None and t_cur >= phase_c_start_t
        ):
            phase_id = 3
            phase_name = 'WALK_FORWARD'
            if phase_c_plan is None:
                if phase_c_start_t is None:
                    phase_c_start_t = t_cur
                phase_c_plan = projector.build_step_plan(
                    q_start=q_ref,
                    intent=intent,
                    step_displacements_x=forward_steps,
                    first_swing_leg=1,
                    t_lead_ds=0.32,
                    t_swing=0.38,
                    t_ds=0.28,
                    t_tail_ds=0.75,
                    base_clearance_m=0.022,
                )

            t_loc = t_cur - phase_c_start_t
            if t_loc > float(phase_c_plan.t[-1]) + 0.35:
                break

            (
                com_ref, com_vel_ref, com_acc_ref, zmp_plan, feet_plan,
                contacts_plan, rpy_plan, phi, ascale, sw_leg, sw_prog
            ) = evaluate_plan_at_time(phase_c_plan, t_loc)

            if enable_body_schema_learning:
                contacts, adapted_feet, snap = monitor.update(
                    t_cur, d_true.qpos, d_true.qvel, tau_applied, phase_c_plan.omega0,
                    com_ref, com_vel_ref, zmp_plan, feet_plan, contacts_plan,
                    sw_leg, sw_prog, phase_name,
                )
                if snap.reproject_delta_xy is not None:
                    obs_com = schema.whole_body_com(d_true.qpos)
                    projector.apply_capture_point_reprojection(
                        phase_c_plan, t_loc, obs_com[:2], snap.reproject_delta_xy
                    )
                    (
                        com_ref, com_vel_ref, com_acc_ref, zmp_plan, adapted_feet,
                        contacts_plan, rpy_plan, phi, ascale, sw_leg, sw_prog
                    ) = evaluate_plan_at_time(phase_c_plan, t_loc)
                snap_dcm = snap.dcm_surprise_m_s
                snap_droop = snap.swing_z_droop_mm
                snap_fz = np.array([snap.inferred_fz_L_N, snap.inferred_fz_R_N])
                snap_reflex = snap.reflex_foot_offset_xy
            else:
                contacts = contacts_plan
                adapted_feet = feet_plan
                snap_dcm = 0.0
                snap_droop = 0.0
                snap_fz = np.zeros(2)
                snap_reflex = np.zeros(2)

            upper_t = intent.upper_body_targets(phi, scale=ascale)
            q_ref = projector.solve_ik_frame(q_ref, com_ref, adapted_feet, rpy_plan, upper_t, n_iters=5)

            # Check mid-walk external push perturbation window
            if cfg.enable_surprise_push and (push_offset_in_walk_s <= t_loc <= push_offset_in_walk_s + cfg.push_duration_s):
                push_active = True
                d_true.xfrc_applied[torso_id, :3] = np.asarray(cfg.push_force_world_N, dtype=float)
            else:
                d_true.xfrc_applied[torso_id, :] = 0.0

        # Smooth causal velocity & acceleration of q_ref
        v_raw = np.zeros(m_true.nv)
        if k > 0:
            mujoco.mj_differentiatePos(schema.m, v_raw, dt, prev_q_ref, q_ref)
        v_prev = v_ref_filt.copy()
        v_ref_filt = 0.45 * v_raw + 0.55 * v_ref_filt
        a_raw = np.clip((v_ref_filt - v_prev) / dt, -60.0, 60.0)
        a_ref_filt = 0.25 * a_raw + 0.75 * a_ref_filt

        # Compute motor torques strictly from internal Body Schema QP!
        tau_cmd = ctrl.compute_torque(
            d_true.qpos, d_true.qvel, q_ref, v_ref_filt, a_ref_filt, contacts,
            com_ref=com_ref, com_vel_ref=com_vel_ref, com_acc_ref=com_acc_ref,
        )

        # Apply true actuator lag & step physical MuJoCo world
        tau_applied += (dt / cfg.motor_lag_tau_s) * (tau_cmd - tau_applied)
        d_true.ctrl[:] = tau_applied

        nonfoot, f_true = measure_ground_contacts(m_true, d_true)
        rpy_cur = Rotation.from_quat(d_true.qpos[[4, 5, 6, 3]]).as_euler('xyz')

        logs['t'].append(t_cur)
        logs['qpos'].append(d_true.qpos.copy())
        logs['ref_qpos'].append(q_ref.copy())
        logs['torque'].append(tau_cmd.copy())
        logs['joint_speed'].append(np.abs(d_true.qvel[6:]).copy())
        logs['tilt'].append(rpy_cur[:2].copy())
        logs['foot_force_true'].append(f_true)
        logs['foot_force_inferred'].append(snap_fz)
        logs['nonfoot_contact'].append(nonfoot)
        logs['true_com'].append(d_true.subtree_com[base_id].copy())
        logs['schema_com'].append(schema.whole_body_com(d_true.qpos))
        logs['com_ref'].append(com_ref.copy())
        logs['zmp_ref'].append(zmp_plan.copy())
        logs['est_dm'].append(schema.est_delta_mass)
        logs['est_dc'].append(schema.est_delta_torso_com.copy())
        logs['est_damping'].append(schema.est_joint_damping)
        logs['est_friction'].append(schema.est_coulomb_friction)
        logs['info_trace'].append(schema.info_gain_trace)
        logs['dcm_surprise'].append(snap_dcm)
        logs['swing_droop_mm'].append(snap_droop)
        logs['reflex_offset_xy'].append(snap_reflex.copy())
        logs['push_active'].append(push_active)
        logs['phase_id'].append(phase_id)

        prev_qvel_meas[:] = d_true.qvel
        mujoco.mj_step(m_true, d_true)
        d_true.xfrc_applied[torso_id, :] = 0.0

        # Proprioceptive acceleration filter & Stage 2 RLS online update
        qacc_meas = (d_true.qvel - prev_qvel_meas) / dt
        schema.update_proprioceptive_filters(d_true.qvel, qacc_meas, tau_cmd, alpha=0.18)

        if enable_body_schema_learning and phase_id == 1 and 80 < k < int(2.70 / dt) and (k % 5 == 0):
            schema.rls_update_double_support(d_true.qpos)

        if d_true.qpos[2] < 0.20 or nonfoot or not np.isfinite(d_true.qpos).all():
            failed = f'Fall / non-foot ground impact at t={d_true.time:.3f}s (phase={phase_id})'
            for _ in range(int(0.25 / dt)):
                d_true.ctrl[:] = 0.0
                mujoco.mj_step(m_true, d_true)
                _, f_post = measure_ground_contacts(m_true, d_true)
                rpy_p = Rotation.from_quat(d_true.qpos[[4, 5, 6, 3]]).as_euler('xyz')
                logs['t'].append(d_true.time)
                logs['qpos'].append(d_true.qpos.copy())
                logs['ref_qpos'].append(q_ref.copy())
                logs['torque'].append(np.zeros(m_true.nu))
                logs['joint_speed'].append(np.abs(d_true.qvel[6:]).copy())
                logs['tilt'].append(rpy_p[:2].copy())
                logs['foot_force_true'].append(f_post)
                logs['foot_force_inferred'].append(np.zeros(2))
                logs['nonfoot_contact'].append(True)
                logs['true_com'].append(d_true.subtree_com[base_id].copy())
                logs['schema_com'].append(schema.whole_body_com(d_true.qpos))
                logs['com_ref'].append(com_ref.copy())
                logs['zmp_ref'].append(zmp_plan.copy())
                logs['est_dm'].append(schema.est_delta_mass)
                logs['est_dc'].append(schema.est_delta_torso_com.copy())
                logs['est_damping'].append(schema.est_joint_damping)
                logs['est_friction'].append(schema.est_coulomb_friction)
                logs['info_trace'].append(schema.info_gain_trace)
                logs['dcm_surprise'].append(snap_dcm)
                logs['swing_droop_mm'].append(snap_droop)
                logs['reflex_offset_xy'].append(snap_reflex.copy())
                logs['push_active'].append(False)
                logs['phase_id'].append(phase_id)
            break

    np_logs = {k: np.asarray(v) for k, v in logs.items()}
    np.savez_compressed(OUT_DIR / f'{exp_name}.npz', **np_logs)

    extra = {
        'description': (
            'Full 4-stage curriculum Body Schema Self-Exploration (ASID -> Stepping-in-Place -> UMR Walk + Push Reflex)'
            if enable_body_schema_learning
            else 'Ablation: Stage 3 UMR Dynamic Projection using uncalibrated Nominal URDF only'
        ),
        'stage3_adapted_posture': {
            'auto_pitch_bias_deg': round(float(np.degrees(phase_c_plan.auto_pitch_bias_rad)), 2) if phase_c_plan else 0.0,
            'auto_roll_bias_deg': round(float(np.degrees(phase_c_plan.auto_roll_bias_rad)), 2) if phase_c_plan else 0.0,
            'adapted_stance_half_width_m': round(float(phase_c_plan.adapted_half_width_m), 4) if phase_c_plan else 0.0815,
        },
        'surprise_events': monitor.surprise_events,
    }
    report = summarize_run(exp_name, cfg, m_true, schema, np_logs, failed, ctrl.qp_fallbacks, extra)
    report_file = 'exp3_report.json' if enable_body_schema_learning else 'exp2_report.json'
    (OUT_DIR / report_file).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('=== [Stage 1] Internalizing UMR Walking Reference on Pi Plus 22-DOF URDF ===')
    intent = internalize_umr_reference()
    (OUT_DIR / 'stage1_kinematic_audit.json').write_text(
        json.dumps(intent.audit.to_dict(), indent=2, ensure_ascii=False)
    )
    print(json.dumps(intent.audit.to_dict(), indent=2, ensure_ascii=False))

    cfg = TrueWorldConfig()

    print('\n=== [Exp 1] Running Raw UMR Kinematic Reference Replay (Torque-Only MuJoCo) ===')
    r1 = run_exp1_raw_umr_replay(intent, cfg)
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    print('\n=== [Exp 2] Running Ablation: Uncalibrated Nominal URDF Dynamic Projection ===')
    r2 = run_curriculum_or_ablation(intent, cfg, enable_body_schema_learning=False)
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print('\n=== [Exp 3] Running Full 4-Stage Body Schema Self-Exploration Curriculum ===')
    r3 = run_curriculum_or_ablation(intent, cfg, enable_body_schema_learning=True)
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    summary = {
        'elapsed_wall_s': round(time.time() - t0, 2),
        'stage1_audit': intent.audit.to_dict(),
        'exp1_raw_umr_replay': r1,
        'exp2_uncalibrated_schema': r2,
        'exp3_full_body_schema': r3,
    }
    (OUT_DIR / 'experiment_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nAll 3 MuJoCo experiments completed in {summary["elapsed_wall_s"]}s!')


if __name__ == '__main__':
    main()
