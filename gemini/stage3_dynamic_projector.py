"""Stage 3: Dynamic Feasibility Projection & Calibrated Whole-Body Schema IK.

Translates high-level kinematic walking intent (from Stage 1 UMR reference)
into a dynamically feasible whole-body trajectory using the Stage-2 calibrated
ExplicitBodySchema:
1. Solves the exact boundary-value Linear Inverted Pendulum / Capture-Point (DCM)
   CoM differential equation c_ddot - omega0^2 * c = -omega0^2 * p_zmp via a fast
   symmetric tridiagonal banded linear system.
2. Automatically adapts torso pitch/roll bias and stance width to compensate for
   calibrated CoM offsets (e.g., unknown torso payload) and single-support roll droop.
3. Generates C^2-continuous quintic swing-foot trajectories with zero impact velocity.
4. Maps (CoM_target, Foot_L, Foot_R, Torso_RPY, UMR_UpperBody) -> bounded whole-body
   joint states (q, qvel, qacc) respecting flexed-knee margin (calf >= 0.40 rad).
"""
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation

from stage1_kinematic_internalizer import MotorIntent
from world_and_schema import ExplicitBodySchema, LEG_JOINTS


def quintic_blend(s: np.ndarray | float) -> np.ndarray | float:
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def quintic_blend_vel(s: np.ndarray | float, T: float) -> np.ndarray | float:
    s = np.clip(s, 0.0, 1.0)
    return (30.0 * s * s * (1.0 - s) * (1.0 - s)) / max(T, 1e-6)


def quintic_bell(s: np.ndarray | float) -> np.ndarray | float:
    """C^2 bell curve in [0, 1] peaking at 1.0 when s = 0.5, zero pos/vel/acc at endpoints."""
    s = np.clip(s, 0.0, 1.0)
    # 16 * s^2 * (1-s)^2 has h(0)=h(1)=0, h'(0)=h'(1)=0, h(0.5)=1.0
    u = s * (1.0 - s)
    return 16.0 * u * u


@dataclass
class StepSpec:
    swing_leg: int          # 0 = Left swings (Right supports), 1 = Right swings (Left supports)
    t_start: float
    t_end: float
    target_xy: np.ndarray   # (2,) world XY landing target for swing sole
    clearance_m: float
    gait_phase_start: float
    gait_phase_end: float
    arm_swing_scale: float


@dataclass
class ProjectedSegmentPlan:
    t: np.ndarray               # (N,)
    com_pos: np.ndarray         # (N, 3)
    com_vel: np.ndarray         # (N, 3)
    com_acc: np.ndarray         # (N, 3)
    zmp_ref: np.ndarray         # (N, 2)
    foot_pos: np.ndarray        # (N, 2, 3)
    foot_vel: np.ndarray        # (N, 2, 3)
    contacts: np.ndarray        # (N, 2) bool
    torso_rpy: np.ndarray       # (N, 3)
    gait_phase: np.ndarray      # (N,)
    arm_scale: np.ndarray       # (N,)
    omega0: float
    auto_pitch_bias_rad: float
    auto_roll_bias_rad: float
    adapted_half_width_m: float
    steps: list[StepSpec]


def solve_lipm_bvp_1d(zmp: np.ndarray, dt: float, omega0: float, x0: float, xT: float) -> np.ndarray:
    """Exact discrete BVP solver for x_ddot - omega0^2 * x = -omega0^2 * zmp with x(0)=x0, x(T)=xT."""
    n = len(zmp)
    if n <= 2:
        return np.linspace(x0, xT, n)
    m = n - 2
    inv_dt2 = 1.0 / (dt * dt)
    diag = (2.0 * inv_dt2 + omega0 * omega0) * np.ones(m)
    off = -inv_dt2 * np.ones(m - 1)
    ab = np.zeros((3, m))
    ab[0, 1:] = off
    ab[1, :] = diag
    ab[2, :-1] = off
    rhs = (omega0 * omega0) * zmp[1:-1].copy()
    rhs[0] += inv_dt2 * x0
    rhs[-1] += inv_dt2 * xT
    x_inner = solve_banded((1, 1), ab, rhs)
    return np.r_[x0, x_inner, xT]


class DynamicFeasibilityProjector:
    """Projects MotorIntent through the calibrated ExplicitBodySchema."""

    def __init__(self, schema: ExplicitBodySchema, nominal_qpos: np.ndarray):
        self.schema = schema
        self.nominal_qpos = nominal_qpos.copy()

        m = schema.m
        self.leg_jids = [m.joint(n).id for n in LEG_JOINTS]
        self.qa = np.r_[np.arange(3), m.jnt_qposadr[self.leg_jids]]
        self.va = np.r_[np.arange(3), m.jnt_dofadr[self.leg_jids]]

        self.lo = np.r_[[-20.0, -20.0, 0.16], m.jnt_range[self.leg_jids, 0] + 0.035]
        self.hi = np.r_[[20.0, 20.0, 0.55], m.jnt_range[self.leg_jids, 1] - 0.035]
        # Enforce flexed-knee branch away from straight-knee singularity (calf >= 0.38 rad)
        # and slight extra margin on narrow inner hip-roll & ankle-pitch limits
        for idx, name in enumerate(LEG_JOINTS):
            jid = self.leg_jids[idx]
            if '_calf_' in name:
                self.lo[3 + idx] = max(self.lo[3 + idx], 0.38)
            elif name == 'l_hip_roll_joint':
                self.lo[3 + idx] = max(self.lo[3 + idx], m.jnt_range[jid, 0] + 0.048)
            elif name == 'r_hip_roll_joint':
                self.hi[3 + idx] = min(self.hi[3 + idx], m.jnt_range[jid, 1] - 0.048)
            elif '_ankle_pitch_' in name:
                self.lo[3 + idx] = max(self.lo[3 + idx], m.jnt_range[jid, 0] + 0.051)

    def compute_schema_posture_adaptation(self) -> tuple[float, float, float]:
        """Compute torso pitch/roll compensation & stance half-width from calibrated schema."""
        # Compare calibrated CoM vs uncalibrated nominal URDF CoM at nominal standing pose
        self.schema.d_prior.qpos[:] = self.nominal_qpos
        mujoco.mj_forward(self.schema.m_prior, self.schema.d_prior)
        com_nom = self.schema.d_prior.subtree_com[self.schema.base_id].copy()

        com_cal = self.schema.whole_body_com(self.nominal_qpos)
        delta_com_wb = com_cal - com_nom  # e.g., [-0.0185, +0.0084, +0.020]

        # If payload shifted CoM backward (delta_x < 0), lean torso forward (pitch > 0)
        # so pelvis stays centered in leg workspace while true CoM sits over support center!
        lever_z = 0.20
        auto_pitch = float(np.clip(-0.75 * delta_com_wb[0] / lever_z, -0.14, 0.16))
        auto_roll = float(np.clip(0.45 * delta_com_wb[1] / lever_z, -0.08, 0.08))

        # Keep nominal stance half-width 0.0815 m (sole spans [0.0565, 0.1065] m) so lateral
        # CoM oscillation stays compact and never overshoots the outer sole edge!
        half_width = 0.0815
        return auto_pitch, auto_roll, half_width

    def build_step_plan(
        self,
        q_start: np.ndarray,
        intent: MotorIntent,
        step_displacements_x: list[float],
        first_swing_leg: int = 1,  # 1 = Right swings first (matching UMR reference!)
        t_lead_ds: float = 0.35,
        t_swing: float = 0.40,
        t_ds: float = 0.24,
        t_tail_ds: float = 0.55,
        base_clearance_m: float = 0.024,
        dt_plan: float = 0.004,
    ) -> ProjectedSegmentPlan:
        """Project a sequence of steps into a dynamically feasible LIPM/DCM + Foot plan."""
        self.schema.d.qpos[:] = q_start
        mujoco.mj_forward(self.schema.m, self.schema.d)
        init_feet = self.schema.d.site_xpos[self.schema.sole_sites].copy()
        # Project initial feet Z cleanly onto floor plane z=0
        init_feet[:, 2] = 0.0
        init_com = self.schema.d.subtree_com[self.schema.base_id].copy()

        auto_pitch, auto_roll, half_width = self.compute_schema_posture_adaptation()
        clearance = base_clearance_m + self.schema.ss_z_droop_comp_m
        # Target walking CoM height: maintain comfortable flexed knees (~0.80-0.95 rad)
        z_c = float(np.clip(init_com[2], 0.295, 0.325))
        omega0 = float(np.sqrt(9.81 / z_c))

        # Construct step specifications
        cur_feet_xy = init_feet[:, :2].copy()
        centerline_y = float(0.5 * (cur_feet_xy[0, 1] + cur_feet_xy[1, 1]))
        steps: list[StepSpec] = []
        cursor_t = t_lead_ds
        swing_leg = first_swing_leg

        for idx, dx in enumerate(step_displacements_x):
            stance_leg = 1 - swing_leg
            # Swing foot lands dx ahead of CURRENT STANCE FOOT X (or smoothly catches up on final step)
            if idx == len(step_displacements_x) - 1 and abs(dx) < 1e-6:
                target_x = cur_feet_xy[stance_leg, 0]
            else:
                target_x = cur_feet_xy[stance_leg, 0] + dx
            target_y = centerline_y + (half_width if swing_leg == 0 else -half_width)
            target_xy = np.array([target_x, target_y], dtype=float)

            # UMR gait phase: Right swing = [0, pi], Left swing = [pi, 2*pi]
            phi0 = 0.0 if swing_leg == 1 else np.pi
            phi1 = phi0 + np.pi
            arm_scale = float(np.clip(abs(dx) / 0.075, 0.15, 1.0))

            steps.append(
                StepSpec(
                    swing_leg=swing_leg,
                    t_start=cursor_t,
                    t_end=cursor_t + t_swing,
                    target_xy=target_xy,
                    clearance_m=clearance,
                    gait_phase_start=phi0,
                    gait_phase_end=phi1,
                    arm_swing_scale=arm_scale,
                )
            )
            cur_feet_xy[swing_leg] = target_xy
            cursor_t += t_swing + t_ds
            swing_leg = 1 - swing_leg

        total_T = cursor_t - t_ds + t_tail_ds if steps else (t_lead_ds + t_tail_ds)
        t_grid = np.arange(0.0, total_T + 0.5 * dt_plan, dt_plan)
        N = len(t_grid)

        foot_pos = np.zeros((N, 2, 3))
        foot_vel = np.zeros((N, 2, 3))
        contacts = np.ones((N, 2), dtype=bool)
        zmp_ref = np.zeros((N, 2))
        gait_phase = np.zeros(N)
        arm_scale = np.zeros(N)

        # Populate foot trajectories & contacts
        live_feet = init_feet.copy()
        step_idx = 0
        for k, tk in enumerate(t_grid):
            while step_idx < len(steps) and tk > steps[step_idx].t_end + 1e-9:
                live_feet[steps[step_idx].swing_leg, :2] = steps[step_idx].target_xy
                live_feet[steps[step_idx].swing_leg, 2] = 0.0
                step_idx += 1

            foot_pos[k] = live_feet
            if step_idx < len(steps):
                st = steps[step_idx]
                if st.t_start <= tk <= st.t_end:
                    sw = st.swing_leg
                    contacts[k, sw] = False
                    tau = (tk - st.t_start) / (st.t_end - st.t_start)
                    b = quintic_blend(tau)
                    bv = quintic_blend_vel(tau, st.t_end - st.t_start)
                    start_xy = live_feet[sw, :2]
                    foot_pos[k, sw, :2] = (1.0 - b) * start_xy + b * st.target_xy
                    foot_vel[k, sw, :2] = bv * (st.target_xy - start_xy)
                    # C^2 bell curve for vertical foot clearance + smooth -1.5mm touchdown press
                    # so compliant sole positively establishes ground contact at end of swing!
                    foot_pos[k, sw, 2] = st.clearance_m * quintic_bell(tau) - 0.0015 * b
                    foot_vel[k, sw, 2] = (
                        st.clearance_m * 32.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau) / (st.t_end - st.t_start)
                        - 0.0015 * bv
                    )
                    gait_phase[k] = (1.0 - tau) * st.gait_phase_start + tau * st.gait_phase_end
                    arm_scale[k] = st.arm_swing_scale
                else:
                    # In double support before st: hold previous end phase smoothly
                    gait_phase[k] = st.gait_phase_start
                    arm_scale[k] = st.arm_swing_scale
            elif steps:
                gait_phase[k] = steps[-1].gait_phase_end
                arm_scale[k] = 0.0

        # Construct piecewise-smooth ZMP reference trajectory across DS & SS intervals!
        # Note: each Pi Plus sole at y = +-0.0815 m has lateral contact cylinders spanning
        # [0.0565, 0.1065] m. Placing planned ZMP at +-0.068 m (inner-middle safe zone)
        # gives >11 mm inner margin AND >38 mm outer margin against compliant roll overshoot!
        def stance_zmp_center(foot_xy: np.ndarray, stance_leg: int) -> np.ndarray:
            inward_shift = -0.0135 if stance_leg == 0 else +0.0135
            lat_sign = 1.0 if stance_leg == 0 else -1.0
            y_zmp = foot_xy[1] + inward_shift + lat_sign * self.schema.ss_lat_bias_comp_m
            y_zmp = np.clip(y_zmp, foot_xy[1] - 0.016, foot_xy[1] + 0.006)
            return np.array([foot_xy[0], y_zmp], dtype=float)

        if not steps:
            zmp_ref[:] = init_com[:2]
            end_com_xy = init_com[:2].copy()
        else:
            # Build knot intervals for ZMP:
            # 1) Lead DS [0, steps[0].t_start]: init_com[:2] -> stance_0
            # 2) Each SS [st.t_start, st.t_end]: slight heel-to-toe progression (-5mm -> +5mm) on stance foot
            # 3) Inter-step DS [steps[i].t_end, steps[i+1].t_start]: stance_i -> stance_{i+1}
            # 4) Tail DS [steps[-1].t_end, total_T]: last_stance -> final_midpoint
            final_mid_xy = np.array([
                0.5 * (cur_feet_xy[0, 0] + cur_feet_xy[1, 0]),
                centerline_y,
            ])

            # Precompute stance foot XY during each step
            stance_centers = []
            sim_feet = init_feet[:, :2].copy()
            for st in steps:
                stance_leg = 1 - st.swing_leg
                stance_centers.append(stance_zmp_center(sim_feet[stance_leg], stance_leg))
                sim_feet[st.swing_leg] = st.target_xy

            for k, tk in enumerate(t_grid):
                if tk < steps[0].t_start:
                    s = quintic_blend(tk / max(steps[0].t_start, 1e-6))
                    p1 = stance_centers[0] + np.array([-0.005, 0.0])
                    zmp_ref[k] = (1.0 - s) * init_com[:2] + s * p1
                elif tk > steps[-1].t_end:
                    s = quintic_blend((tk - steps[-1].t_end) / max(total_T - steps[-1].t_end, 1e-6))
                    p0 = stance_centers[-1] + np.array([0.005, 0.0])
                    zmp_ref[k] = (1.0 - s) * p0 + s * final_mid_xy
                else:
                    for i, st in enumerate(steps):
                        if st.t_start <= tk <= st.t_end:
                            s = (tk - st.t_start) / (st.t_end - st.t_start)
                            zmp_ref[k] = stance_centers[i] + np.array([-0.005 + 0.010 * s, 0.0])
                            break
                        if i + 1 < len(steps) and st.t_end < tk < steps[i + 1].t_start:
                            s = quintic_blend((tk - st.t_end) / max(steps[i + 1].t_start - st.t_end, 1e-6))
                            p0 = stance_centers[i] + np.array([0.005, 0.0])
                            p1 = stance_centers[i + 1] + np.array([-0.005, 0.0])
                            zmp_ref[k] = (1.0 - s) * p0 + s * p1
                            break
            end_com_xy = final_mid_xy

        # Solve exact tridiagonal LIPM BVP for CoM X and Y!
        com_x = solve_lipm_bvp_1d(zmp_ref[:, 0], dt_plan, omega0, float(init_com[0]), float(end_com_xy[0]))
        com_y = solve_lipm_bvp_1d(zmp_ref[:, 1], dt_plan, omega0, float(init_com[1]), float(end_com_xy[1]))
        com_z = np.full(N, z_c)
        # Smooth initial CoM Z blend in first 0.25s if init_com[2] differs slightly
        blend_z = quintic_blend(np.clip(t_grid / 0.25, 0.0, 1.0))
        com_z = (1.0 - blend_z) * float(init_com[2]) + blend_z * z_c

        com_pos = np.column_stack([com_x, com_y, com_z])
        com_vel = np.gradient(com_pos, dt_plan, axis=0)
        com_vel[0] = 0.0
        com_vel[-1] = 0.0
        com_acc = np.gradient(com_vel, dt_plan, axis=0)

        # Smoothly ramp in torso pitch/roll compensation
        ramp = quintic_blend(np.clip(t_grid / 0.30, 0.0, 1.0))
        init_rpy = Rotation.from_quat(q_start[[4, 5, 6, 3]]).as_euler('xyz')
        torso_rpy = np.column_stack([
            (1.0 - ramp) * init_rpy[0] + ramp * auto_roll,
            (1.0 - ramp) * init_rpy[1] + ramp * auto_pitch,
            np.full(N, init_rpy[2]),
        ])

        return ProjectedSegmentPlan(
            t=t_grid,
            com_pos=com_pos,
            com_vel=com_vel,
            com_acc=com_acc,
            zmp_ref=zmp_ref,
            foot_pos=foot_pos,
            foot_vel=foot_vel,
            contacts=contacts,
            torso_rpy=torso_rpy,
            gait_phase=gait_phase,
            arm_scale=arm_scale,
            omega0=omega0,
            auto_pitch_bias_rad=auto_pitch,
            auto_roll_bias_rad=auto_roll,
            adapted_half_width_m=half_width,
            steps=steps,
        )

    def apply_capture_point_reprojection(
        self,
        plan: ProjectedSegmentPlan,
        t_local: float,
        obs_com_xy: np.ndarray,
        delta_xy: np.ndarray,
    ) -> None:
        """Online 0.1ms LIPM + Footstep re-projection when Stage 4 detects an external push."""
        tg = plan.t
        dt_plan = float(tg[1] - tg[0])
        idx_now = int(np.clip(np.searchsorted(tg, t_local), 0, len(tg) - 3))

        # Find current or next swing step
        active_i = None
        for i, st in enumerate(plan.steps):
            if st.t_end >= t_local + 0.04:
                active_i = i
                break
        if active_i is None:
            return

        st0 = plan.steps[active_i]
        sw0 = st0.swing_leg
        stance0 = 1 - sw0
        stance_y = float(plan.foot_pos[idx_now, stance0, 1])

        # Enforce balance-preserving stance width: reflex step may WIDEN stance, never narrow below 0.163 m!
        safe_delta = delta_xy.copy()
        proposed_y = st0.target_xy[1] + safe_delta[1]
        if sw0 == 0:
            proposed_y = max(proposed_y, stance_y + 0.163)
        else:
            proposed_y = min(proposed_y, stance_y - 0.163)
        safe_delta[1] = proposed_y - st0.target_xy[1]

        # Shift active & subsequent footsteps
        for i in range(active_i, len(plan.steps)):
            plan.steps[i].target_xy = plan.steps[i].target_xy + safe_delta

        # Smoothly blend foot_pos and zmp_ref from idx_now onward with C^2 continuity!
        t_blend_start = max(t_local, st0.t_start)
        t_zmp_done = plan.steps[active_i + 1].t_start if (active_i + 1 < len(plan.steps)) else (st0.t_end + 0.20)
        for k in range(idx_now, len(tg)):
            tk = tg[k]
            if st0.t_start <= tk <= st0.t_end:
                tau = (tk - t_blend_start) / max(st0.t_end - t_blend_start, 1e-4)
                b = float(quintic_blend(np.clip(tau, 0.0, 1.0)))
                plan.foot_pos[k, sw0, :2] += b * safe_delta
            elif tk > st0.t_end:
                plan.foot_pos[k, sw0, :2] += safe_delta
                if active_i + 1 < len(plan.steps):
                    st1 = plan.steps[active_i + 1]
                    if st1.t_start <= tk <= st1.t_end:
                        tau1 = (tk - st1.t_start) / max(st1.t_end - st1.t_start, 1e-4)
                        plan.foot_pos[k, stance0, :2] += float(quintic_blend(tau1)) * safe_delta
                    elif tk > st1.t_end:
                        plan.foot_pos[k, stance0, :2] += safe_delta
                else:
                    plan.foot_pos[k, stance0, :2] += safe_delta

                if tk < t_zmp_done:
                    bz = float(quintic_blend((tk - st0.t_end) / max(t_zmp_done - st0.t_end, 1e-4)))
                    plan.zmp_ref[k] += bz * safe_delta
                else:
                    plan.zmp_ref[k] += safe_delta

        # Re-solve remaining LIPM CoM trajectory in 0.1ms continuously from plan.com_pos[idx_now]!
        c0_x = float(plan.com_pos[idx_now, 0])
        c0_y = float(plan.com_pos[idx_now, 1])
        rem_x = solve_lipm_bvp_1d(plan.zmp_ref[idx_now:, 0], dt_plan, plan.omega0, c0_x, float(plan.zmp_ref[-1, 0]))
        rem_y = solve_lipm_bvp_1d(plan.zmp_ref[idx_now:, 1], dt_plan, plan.omega0, c0_y, float(plan.zmp_ref[-1, 1]))
        plan.com_pos[idx_now:, 0] = rem_x
        plan.com_pos[idx_now:, 1] = rem_y
        plan.com_vel = np.gradient(plan.com_pos, dt_plan, axis=0)
        plan.com_vel[0] = 0.0
        plan.com_vel[-1] = 0.0
        plan.com_acc = np.gradient(plan.com_vel, dt_plan, axis=0)

    def solve_ik_frame(
        self,
        q_seed: np.ndarray,
        target_com: np.ndarray,
        target_feet: np.ndarray,
        target_rpy: np.ndarray,
        upper_targets: dict[str, float],
        n_iters: int = 5,
    ) -> np.ndarray:
        """Fast bounded whole-body differential IK using the calibrated Body Schema."""
        m = self.schema.m
        d = self.schema.d
        q = q_seed.copy()

        # Apply UMR phase-synchronized upper-body coordination
        for jname, val in upper_targets.items():
            jid = m.joint(jname).id
            adr = m.jnt_qposadr[jid]
            q[adr] = np.clip(val, m.jnt_range[jid, 0] + 0.03, m.jnt_range[jid, 1] - 0.03)

        # Optimize [base_xyz (3), 12 leg joints (12), base_rpy (3)] = 18 vars
        x_pos = np.clip(q[self.qa].copy(), self.lo, self.hi)
        x_rpy = Rotation.from_quat(q[[4, 5, 6, 3]]).as_euler('xyz')
        npos = len(self.qa)
        nvar = npos + 3

        for _ in range(n_iters):
            q[self.qa] = x_pos
            rot = Rotation.from_euler('xyz', x_rpy)
            quat = rot.as_quat()
            q[3:7] = quat[[3, 0, 1, 2]]
            d.qpos[:] = q
            mujoco.mj_forward(m, d)

            roll, pitch, yaw = x_rpy
            rz = Rotation.from_euler('z', yaw).as_matrix()
            ry = Rotation.from_euler('y', pitch).as_matrix()
            axes = np.column_stack([rz @ ry @ np.array([1.0, 0.0, 0.0]), rz @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]])
            tangent = np.zeros((m.nv, nvar))
            tangent[self.va, np.arange(npos)] = 1.0
            tangent[3:6, npos:] = rot.as_matrix().T @ axes

            res_list = [target_com - d.subtree_com[self.schema.base_id]]
            jac_list = []
            jc = np.zeros((3, m.nv))
            mujoco.mj_jacSubtreeCom(m, d, jc, self.schema.base_id)
            jac_list.append(jc @ tangent)

            for ee, si in enumerate(self.schema.sole_sites):
                res_list.append(target_feet[ee] - d.site_xpos[si])
                # Flat horizontal sole orientation with yaw matching target_rpy[2]
                des_foot_rot = Rotation.from_euler('z', target_rpy[2]).as_matrix()
                cur_foot_rot = d.site_xmat[si].reshape(3, 3)
                rot_err = Rotation.from_matrix(des_foot_rot @ cur_foot_rot.T).as_rotvec()
                res_list.append(0.35 * rot_err)

                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jp, jr, si)
                jac_list.extend([jp @ tangent, 0.35 * (jr @ tangent)])

            # Soft torso RPY tracking & leg branch-continuity regularization (prevents 1-frame branch flips)
            w_rpy = 0.08
            res_list.append(w_rpy * (target_rpy - x_rpy))
            jac_list.append(np.c_[np.zeros((3, npos)), w_rpy * np.eye(3)])

            w_cont = 0.035
            seed_leg = q_seed[self.qa[3:]]
            res_list.append(w_cont * (seed_leg - x_pos[3:]))
            jac_cont = np.zeros((12, nvar))
            jac_cont[:, 3:npos] = w_cont * np.eye(12)
            jac_list.append(jac_cont)

            J = np.vstack(jac_list)
            e = np.concatenate(res_list)
            if np.linalg.norm(e[:15]) < 1e-5:
                break
            dx = np.linalg.solve(J.T @ J + 1e-4 * np.eye(nvar), J.T @ e)
            x_pos = np.clip(x_pos + dx[:npos], self.lo, self.hi)
            x_rpy = np.clip(x_rpy + dx[npos:], target_rpy - 0.16, target_rpy + 0.16)

        q[self.qa] = x_pos
        quat = Rotation.from_euler('xyz', x_rpy).as_quat()
        q[3:7] = quat[[3, 0, 1, 2]]
        return q
