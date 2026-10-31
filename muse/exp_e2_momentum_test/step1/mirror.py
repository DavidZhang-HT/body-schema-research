"""Sagittal-plane (x-z) mirror of a 20-DoF pi+ configuration.

Reflection M = diag(1,-1,1).  For a rigid pose (R, p): (M R M, M p).
For a revolute joint: M Rot(u, th) M = Rot(M u, -th) (improper conjugation flips sense).
All pi+ joint axes are coordinate axes and L/R pairs share the same axis (verified from
pi_plus_20dof.xml), so with u_l == u_r for each pair:

    axis (1,0,0) [roll]  : M u =  u -> th'_l = -th_r
    axis (0,1,0) [pitch] : M u = -u -> th'_l = +th_r
    axis (0,0,1) [yaw]   : M u =  u -> th'_l = -th_r

Base quaternion: q' from R' = M R M via mujoco mju_quat2Mat / mju_mat2Quat.
"""
import numpy as np
import mujoco

M = np.diag([1.0, -1.0, 1.0])

# joint-name substring -> mirror sign (from pi_plus_20dof.xml joint axes)
_SIGN = {
    "hip_pitch": +1.0, "hip_roll": -1.0, "thigh": -1.0, "calf": +1.0,
    "ankle_pitch": +1.0, "ankle_roll": -1.0,
    "shoulder_pitch": +1.0, "shoulder_roll": -1.0, "upper_arm": -1.0, "elbow": +1.0,
}


def _partner(name: str) -> str:
    if name.startswith("l_"):
        return "r_" + name[2:]
    if name.startswith("r_"):
        return "l_" + name[2:]
    raise ValueError(name)


def _sign(name: str) -> float:
    for k, s in _SIGN.items():
        if k in name:
            return s
    raise ValueError(name)


def mirror_q(q, joint_names, qadr):
    """q: full qpos (nq,). joint_names: model.JOINTS order; qadr: S.r.qadr (same order)."""
    q = np.asarray(q, float).copy()
    out = q.copy()
    # base position: y -> -y
    out[0:3] = M @ q[0:3]
    # base quaternion (w,x,y,z): R' = M R M
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, q[3:7])
    Rm = Rm.reshape(3, 3)
    Rp = M @ Rm @ M
    qq = np.zeros(4)
    mujoco.mju_mat2Quat(qq, Rp.reshape(9))
    # mju_mat2Quat can return a sign-flipped but equivalent quat; canonicalise w >= 0
    if qq[0] < 0:
        qq = -qq
    out[3:7] = qq
    # joints: swap L<->R with per-axis sign
    idx = {n: i for i, n in enumerate(joint_names)}
    for n in joint_names:
        p = _partner(n)
        out[qadr[idx[n]]] = _sign(n) * q[qadr[idx[p]]]
    return out
