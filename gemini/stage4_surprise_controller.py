"""Stage 4: Whole-Body Schema Torque QP Controller & Surprise-Driven Online Loop.

1. `WholeBodySchemaQPController`:
   Torque-only floating-base whole-body inverse dynamics QP operating strictly
   on the robot's internal `ExplicitBodySchema` (`schema.m`, `schema.d`).
   No external base forces, no mocap welds, no qpos resets, strict URDF effort limits.
2. `SurpriseDrivenLoop`:
   At 500 Hz, predicts Capture-Point (DCM) evolution, swing-foot trajectory, and
   contact timing from proprioception, attributing prediction errors (Surprise) into:
   - Systematic internal compliance/lag -> updates swing clearance & stance bias;
   - Transient external momentum surprise -> freezes internal mass ID and triggers
     reflexive Capture-Point footstep placement adaptation.
"""
from dataclasses import dataclass, field
import json
import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from world_and_schema import ASSETS, ExplicitBodySchema


def jac_site_6d(m: mujoco.MjModel, d: mujoco.MjData, site_id: int) -> np.ndarray:
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, site_id)
    return np.vstack([jp, jr])


def jacdot_site_v(m: mujoco.MjModel, d: mujoco.MjData, site_id: int) -> np.ndarray:
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacDot(m, d, jp, jr, d.site_xpos[site_id], m.site_bodyid[site_id])
    return np.vstack([jp, jr]) @ d.qvel


def rot_err_vec(target_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target_mat @ current_mat.T).as_rotvec()


class WholeBodySchemaQPController:
    """Constrained Whole-Body Inverse Dynamics QP using only `schema.m`."""

    def __init__(
        self,
        schema: ExplicitBodySchema,
        leg_kp: float = 85.0,
        leg_kd: float = 3.0,
        base_kp: float = 155.0,
        base_kd: float = 26.0,
        com_kp: float = 165.0,
        com_kd: float = 28.0,
    ):
        self.schema = schema
        self.leg_kp = leg_kp
        self.leg_kd = leg_kd
        self.base_kp = base_kp
        self.base_kd = base_kd
        self.com_kp = com_kp
        self.com_kd = com_kd

        m = schema.m
        self.ref_data = mujoco.MjData(m)
        self.scratch_data = mujoco.MjData(m)
        self.prev_sol: np.ndarray | None = None
        self.qp_fallbacks = 0
        self.leg_motors = np.array([
            i for i in range(m.nu)
            if any(tag in m.actuator(i).name for tag in ['hip', 'thigh', 'calf', 'ankle'])
        ])
        self.upper_dofs = np.array([
            i for i in range(6, m.nv)
            if (i - 6) not in self.leg_motors
        ])
        model_meta = json.loads((ASSETS / 'model.json').read_text())
        self.vel_limits = np.asarray(model_meta['joint_velocity_limits'], dtype=float)

    def compute_torque(
        self,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        q_ref: np.ndarray,
        v_ref: np.ndarray,
        a_ref: np.ndarray,
        contacts: list[bool],
        com_ref: np.ndarray | None = None,
        com_vel_ref: np.ndarray | None = None,
        com_acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        m = self.schema.m
        d = self.schema.d
        ref = self.ref_data

        d.qpos[:] = q_meas
        d.qvel[:] = v_meas
        mujoco.mj_forward(m, d)

        v_ref_safe = v_ref.copy()
        v_ref_safe[6:] = np.clip(v_ref_safe[6:], -0.76 * self.vel_limits, 0.76 * self.vel_limits)
        v_ref = v_ref_safe

        ref.qpos[:] = q_ref
        ref.qvel[:] = v_ref
        mujoco.mj_forward(m, ref)

        nv = m.nv
        n_vars = nv + 12  # [qacc (28), wrench_L (6), wrench_R (6)]

        mass_mat = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, mass_mat)
        # h includes Coriolis + gravity + calibrated viscous damping (since qfrc_passive = -b*qvel)
        h = d.qfrc_bias - d.qfrc_passive
        # Add calibrated Coulomb friction feedforward on actuated joints
        tau_coulomb = self.schema.est_coulomb_friction * np.tanh(v_meas[6:] / 0.06)
        h_total = h.copy()
        h_total[6:] += tau_coulomb

        jacs = [jac_site_6d(m, d, si) for si in self.schema.sole_sites]
        ext_jac_T = np.hstack([j.T for j in jacs])
        dyn = np.c_[mass_mat, -ext_jac_T]

        costs = []
        rhs = []
        constraints = [dyn[:6]]
        low = [-h_total[:6]]
        high = [-h_total[:6]]

        def add_task(mat: np.ndarray, target: np.ndarray, weight: float) -> None:
            w_sqrt = np.sqrt(weight)
            costs.append(w_sqrt * mat)
            rhs.append(w_sqrt * target)

        qerr = np.zeros(nv)
        mujoco.mj_differentiatePos(m, qerr, 1.0, q_meas, q_ref)

        # 1. Whole-body configuration acceleration regularization
        swing_scale = self.schema.swing_ff_gain
        add_task(
            np.c_[np.eye(nv), np.zeros((nv, 12))],
            swing_scale * a_ref + 80.0 * qerr + 18.0 * (v_ref - v_meas),
            2.5,
        )

        # 1b. Smooth URDF velocity-limit safety barrier (active only above 72% speed limit)
        v_act = v_meas[6:]
        over_v = np.abs(v_act) - 0.72 * self.vel_limits
        fast_ids = np.where(over_v > 0.0)[0]
        if len(fast_ids) > 0:
            b_mat = np.zeros((len(fast_ids), n_vars))
            b_rhs = np.zeros(len(fast_ids))
            for r_i, j_idx in enumerate(fast_ids):
                b_mat[r_i, 6 + j_idx] = 1.0
                v_cap = np.sign(v_act[j_idx]) * 0.68 * self.vel_limits[j_idx]
                b_rhs[r_i] = -(v_act[j_idx] - v_cap) / 0.014
            add_task(b_mat, b_rhs, 190.0)

        # 2. Upper-body UMR arm-swing & head coordination task
        if len(self.upper_dofs) > 0:
            add_task(
                np.c_[np.eye(nv)[self.upper_dofs], np.zeros((len(self.upper_dofs), 12))],
                a_ref[self.upper_dofs] + 110.0 * qerr[self.upper_dofs] + 20.0 * (v_ref[self.upper_dofs] - v_meas[self.upper_dofs]),
                8.0,
            )

        # 3. Floating-base pose / torso orientation task
        add_task(
            np.c_[np.eye(nv)[:6], np.zeros((6, 12))],
            a_ref[:6] + self.base_kp * qerr[:6] + self.base_kd * (v_ref[:6] - v_meas[:6]),
            9.0,
        )

        # 4. Direct Calibrated Whole-Body CoM / Capture-Point Momentum Task!
        jc_cur = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, jc_cur, self.schema.base_id)
        com_cur = d.subtree_com[self.schema.base_id].copy()
        com_vel_cur = jc_cur @ v_meas

        # Directional derivative (J_com_dot @ qvel) via tiny forward step
        eps_fd = 1e-5
        self.scratch_data.qpos[:] = q_meas
        mujoco.mj_integratePos(m, self.scratch_data.qpos, v_meas, eps_fd)
        mujoco.mj_forward(m, self.scratch_data)
        jc_next = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, self.scratch_data, jc_next, self.schema.base_id)
        jdot_v_com = ((jc_next - jc_cur) / eps_fd) @ v_meas

        if com_ref is None or com_vel_ref is None or com_acc_ref is None:
            jc_ref = np.zeros((3, nv))
            mujoco.mj_jacSubtreeCom(m, ref, jc_ref, self.schema.base_id)
            if com_ref is None:
                com_ref = ref.subtree_com[self.schema.base_id].copy()
            if com_vel_ref is None:
                com_vel_ref = jc_ref @ v_ref
            if com_acc_ref is None:
                com_acc_ref = jc_ref @ a_ref

        des_com_acc = (
            com_acc_ref
            + self.com_kp * (com_ref - com_cur)
            + self.com_kd * (com_vel_ref - com_vel_cur)
            - jdot_v_com
        )
        add_task(np.c_[jc_cur, np.zeros((3, 12))], des_com_acc, 45.0)

        # 5. Stance & Swing Foot Tasks + Contact Wrench Cone Constraints
        # Guard against "phantom air wrench": if both feet are planned in contact (DS),
        # check relative sole height from proprioceptive FK; if one foot is still >3.5mm
        # higher than the loaded stance foot, keep its wrench at 0 and drive it down to contact!
        active_contacts = list(contacts)
        if active_contacts[0] and active_contacts[1]:
            z0 = float(d.site_xpos[self.schema.sole_sites[0], 2])
            z1 = float(d.site_xpos[self.schema.sole_sites[1], 2])
            if z0 - z1 > 0.0035:
                active_contacts[0] = False
            elif z1 - z0 > 0.0035:
                active_contacts[1] = False

        total_weight_N = float(m.body_mass.sum() * 9.81)
        for ee, si in enumerate(self.schema.sole_sites):
            j_cur = jacs[ee]
            j_ref = jac_site_6d(m, ref, si)
            pos_err = ref.site_xpos[si] - d.site_xpos[si]
            # If planned contact is waiting for touchdown, pull sole gently downward (-3 mm)
            if contacts[ee] and not active_contacts[ee]:
                pos_err = pos_err.copy()
                pos_err[2] -= 0.003
            ori_err = rot_err_vec(ref.site_xmat[si].reshape(3, 3), d.site_xmat[si].reshape(3, 3))
            err_6d = np.r_[pos_err, ori_err]
            target_acc_6d = (
                j_ref @ a_ref
                + jacdot_site_v(m, ref, si)
                + 340.0 * err_6d
                + 38.0 * (j_ref @ v_ref - j_cur @ v_meas)
                - jacdot_site_v(m, d, si)
            )
            foot_mat = np.c_[j_cur, np.zeros((6, 12))]
            add_task(foot_mat, target_acc_6d, 520.0 if active_contacts[ee] else 120.0)

            wmat = np.zeros((6, n_vars))
            wmat[:, nv + 6 * ee : nv + 6 * (ee + 1)] = np.eye(6)
            if not active_contacts[ee]:
                constraints.append(wmat)
                low.append(np.zeros(6))
                high.append(np.zeros(6))
            else:
                rot_s = d.site_xmat[si].reshape(3, 3)
                tr = np.zeros((6, 6))
                tr[:3, :3] = rot_s.T
                tr[3:, 3:] = rot_s.T
                w_local = tr @ wmat
                # Normal force F_z >= 0
                constraints.append(w_local[2:3])
                low.append(np.array([0.0]))
                high.append(np.array([2.5 * total_weight_N]))
                # Friction pyramid & finite sole CoP bounds
                for dim, coeff in [(0, 0.70), (1, 0.70), (3, 0.024), (4, 0.064), (5, 0.012)]:
                    for sgn in (-1.0, 1.0):
                        constraints.append((sgn * w_local[dim] - coeff * w_local[2])[None, :])
                        low.append(np.array([-np.inf]))
                        high.append(np.array([0.0]))

        # 6. Actuated joint torque limits strictly from URDF ctrlrange
        ctrl_min = m.actuator_ctrlrange[:, 0]
        ctrl_max = m.actuator_ctrlrange[:, 1]
        constraints.append(dyn[6:])
        low.append(ctrl_min - h_total[6:])
        high.append(ctrl_max - h_total[6:])

        # Small torque & wrench regularization
        add_task(dyn[6:], -h_total[6:], 1e-4)
        add_task(np.c_[np.zeros((12, nv)), np.eye(12)], np.zeros(12), 1e-5)

        C_mat = np.vstack(costs)
        b_vec = np.concatenate(rhs)
        P_qp = sparse.csc_matrix(C_mat.T @ C_mat + 1e-7 * np.eye(n_vars))
        q_qp = -C_mat.T @ b_vec
        A_qp = sparse.csc_matrix(np.vstack(constraints))
        l_qp = np.concatenate(low)
        u_qp = np.concatenate(high)

        solver = osqp.OSQP()
        solver.setup(
            P=P_qp,
            q=q_qp,
            A=A_qp,
            l=l_qp,
            u=u_qp,
            verbose=False,
            eps_abs=2e-4,
            eps_rel=2e-4,
            max_iter=8000,
            polishing=False,
        )
        if self.prev_sol is not None:
            solver.warm_start(x=self.prev_sol)
        res = solver.solve(raise_error=False)

        if res.info.status_val in (1, 2) and res.x is not None:
            self.prev_sol = res.x
            tau = dyn[6:] @ res.x + h_total[6:]
            ids = self.leg_motors
            tau[ids] += self.leg_kp * qerr[6 + ids] + self.leg_kd * (v_ref[6 + ids] - v_meas[6 + ids])
        else:
            # Physical fallback when robot is already irreversibly tipping (e.g. raw UMR baseline):
            # use gravity-compensated PD impedance within motor limits so fall plays out physically.
            self.qp_fallbacks += 1
            tau = d.qfrc_bias[6:] + 85.0 * qerr[6:] + 3.5 * (v_ref[6:] - v_meas[6:])

        # Soft URDF velocity regenerative braking before actuator clip
        for j_idx in range(m.nu):
            vj = float(v_meas[6 + j_idx])
            vmax_j = float(self.vel_limits[j_idx])
            if abs(vj) > 0.74 * vmax_j:
                tau[j_idx] -= 6.5 * (vj - np.sign(vj) * 0.74 * vmax_j)

        return np.clip(tau, ctrl_min, ctrl_max)


@dataclass
class SurpriseSnapshot:
    dcm_surprise_m_s: float = 0.0
    swing_z_droop_mm: float = 0.0
    inferred_fz_L_N: float = 0.0
    inferred_fz_R_N: float = 0.0
    attribution: str = 'NOMINAL'
    reflex_foot_offset_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    reproject_delta_xy: np.ndarray | None = None


class SurpriseDrivenMonitor:
    """500 Hz Proprioceptive Prediction-Error Attribution & Reflex Step Adapter."""

    def __init__(self, schema: ExplicitBodySchema):
        self.schema = schema
        self.ext_force_filt = np.zeros(2)
        self.cumulative_reflex_xy = np.zeros(2)
        self.last_reproject_t = -10.0
        self.push_persist_ticks = 0
        self.step_droop_samples: list[float] = []
        self.surprise_events: list[dict] = []

    def update(
        self,
        t_sim: float,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        tau_cmd: np.ndarray,
        omega0: float,
        com_ref: np.ndarray,
        com_vel_ref: np.ndarray,
        zmp_plan: np.ndarray,
        planned_feet: np.ndarray,
        planned_contacts: list[bool],
        swing_leg: int | None,
        swing_progress: float,
        curriculum_phase: str,
    ) -> tuple[list[bool], np.ndarray, SurpriseSnapshot]:
        """Compare internal Body Schema forward predictions against actual proprioception."""
        # 1. Proprioceptively infer ground contact wrenches AND external body push residual!
        # CRITICAL: pass synchronized low-pass filtered tau_filt alongside qvel_filt & qacc_filt
        # so rapid swing-leg torque steps do not create false filter-lag force residuals!
        wrenches, f_ext_world = self.schema.infer_proprioceptive_external_base_force(
            q_meas, self.schema.qvel_filt, self.schema.qacc_filt, self.schema.tau_filt
        )
        fz_L = float(max(0.0, wrenches[0, 2]))
        fz_R = float(max(0.0, wrenches[1, 2]))

        # Low-pass filter inferred horizontal external body force (Newton residual not explained by legs!)
        self.ext_force_filt = 0.14 * f_ext_world[:2] + 0.86 * self.ext_force_filt
        # Normalize surprise score so 10 N external body force = 0.50 on telemetry plot
        dcm_surprise_mag = float(np.linalg.norm(self.ext_force_filt)) / 20.0

        # 2. Capture-Point (DCM) state from calibrated Body Schema
        com_obs, com_vel_obs = self.schema.whole_body_com_and_vel(q_meas, v_meas)
        dcm_obs = com_obs[:2] + com_vel_obs[:2] / omega0
        dcm_ref = com_ref[:2] + com_vel_ref[:2] / omega0
        dcm_err_xy = dcm_obs - dcm_ref
        dcm_err_norm = float(np.linalg.norm(dcm_err_xy))
        f_norm = float(np.linalg.norm(self.ext_force_filt))
        cos_align = float(np.dot(self.ext_force_filt, dcm_err_xy)) / (f_norm * dcm_err_norm + 1e-6)
        # Effective aligned external push surprise score (rejects unaligned foot-strike transients)
        aligned_push_surprise = (
            dcm_surprise_mag
            * max(0.0, cos_align)
            * float(np.clip(dcm_err_norm / 0.021, 0.0, 1.35))
        )

        # 3. Swing-foot vertical droop relative to stance sole
        self.schema.d.qpos[:] = q_meas
        mujoco.mj_forward(self.schema.m, self.schema.d)
        obs_feet = self.schema.d.site_xpos[self.schema.sole_sites].copy()

        droop_mm = 0.0
        effective_contacts = list(planned_contacts)
        adapted_feet = planned_feet.copy()
        attribution = 'NOMINAL'
        reproject_delta: np.ndarray | None = None

        if swing_leg is not None:
            stance_leg = 1 - swing_leg
            obs_rel_z = obs_feet[swing_leg, 2] - obs_feet[stance_leg, 2]
            plan_rel_z = planned_feet[swing_leg, 2] - planned_feet[stance_leg, 2]
            droop_m = float(plan_rel_z - obs_rel_z)
            droop_mm = 1000.0 * droop_m

            if 0.30 <= swing_progress <= 0.70:
                self.step_droop_samples.append(droop_m)

            # Early touchdown contact surprise in late swing
            fz_sw = fz_L if swing_leg == 0 else fz_R
            if swing_progress > 0.84 and (fz_sw > 32.0 or obs_rel_z < 0.0010):
                effective_contacts[swing_leg] = True
                attribution = 'EARLY_CONTACT_REFLEX'

            # External Torso Push Surprise: requires persistent aligned external wrench residual
            # (||F_ext|| > 10 N aligned with Capture-Point excursion ||dcm_err_xy|| > 21 mm)!
            dcm_err_xy = dcm_obs - dcm_ref
            dcm_err_norm = float(np.linalg.norm(dcm_err_xy))
            f_norm = float(np.linalg.norm(self.ext_force_filt))
            cos_align = float(np.dot(self.ext_force_filt, dcm_err_xy)) / (f_norm * dcm_err_norm + 1e-6)
            if (
                curriculum_phase == 'WALK_FORWARD'
                and dcm_surprise_mag > 0.58
                and dcm_err_norm > 0.021
                and cos_align > 0.45
            ):
                self.push_persist_ticks += 1
            else:
                self.push_persist_ticks = max(0, self.push_persist_ticks - 1)

            if (
                self.push_persist_ticks >= 8
                and (t_sim - self.last_reproject_t) > 1.80
                and swing_progress < 0.82
            ):
                attribution = 'EXTERNAL_PUSH_REFLEX'
                self.push_persist_ticks = 0
                # Shift swing foot strictly along actual Capture-Point excursion dcm_err_xy
                reproject_delta = np.clip(0.72 * dcm_err_xy, [-0.025, -0.017], [0.025, 0.017])
                self.cumulative_reflex_xy += reproject_delta
                self.last_reproject_t = t_sim
                self.surprise_events.append({
                    't_s': round(t_sim, 3),
                    'type': attribution,
                    'inferred_ext_force_N': [round(float(self.ext_force_filt[0]), 2), round(float(self.ext_force_filt[1]), 2)],
                    'dcm_err_mm': [round(1000 * dcm_err_xy[0], 1), round(1000 * dcm_err_xy[1], 1)],
                    'foot_reproject_mm': [round(1000 * reproject_delta[0], 1), round(1000 * reproject_delta[1], 1)],
                })
        else:
            self.push_persist_ticks = 0
            # At transition back to Double Support: update internal single-support compliance schema
            if len(self.step_droop_samples) >= 10:
                mean_droop = float(np.mean(self.step_droop_samples))
                if mean_droop > 0.0012:
                    self.schema.ss_z_droop_comp_m = float(
                        np.clip(self.schema.ss_z_droop_comp_m + 0.50 * mean_droop, 0.0, 0.010)
                    )
                    self.schema.swing_ff_gain = float(
                        np.clip(self.schema.swing_ff_gain + 2.5 * mean_droop, 1.0, 1.15)
                    )
                    if curriculum_phase == 'STEP_IN_PLACE':
                        self.schema.ss_lat_bias_comp_m = float(
                            np.clip(self.schema.ss_lat_bias_comp_m + 0.0008, 0.0, 0.0030)
                        )
                    attribution = 'INTERNAL_SCHEMA_CALIBRATED'
                self.step_droop_samples.clear()

        snap = SurpriseSnapshot(
            dcm_surprise_m_s=aligned_push_surprise,
            swing_z_droop_mm=droop_mm,
            inferred_fz_L_N=fz_L,
            inferred_fz_R_N=fz_R,
            attribution=attribution,
            reflex_foot_offset_xy=self.cumulative_reflex_xy.copy(),
            reproject_delta_xy=reproject_delta,
        )
        return effective_contacts, adapted_feet, snap
