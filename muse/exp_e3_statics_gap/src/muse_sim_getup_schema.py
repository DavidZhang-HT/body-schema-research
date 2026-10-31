"""Body-schema / contact-schema quantities for the get-up problem (F3).

Everything here is *explicit* and computed from the nominal internal model (URDF/MJCF prior +
the G0 primitive contact shell).  Nothing is learned.

  * Shell points: every floor-colliding primitive is reduced to support points
      box      -> its corners that lie on the convex hull of the link's shell (radius 0)
      capsule  -> two end spheres (centre, radius)
      sphere   -> centre, radius
    The lowest point of a sphere is centre - r*e_z; its vertical velocity equals the
    centre's, so the centre Jacobian is exact for the normal direction.
  * Regions: points are grouped into anatomically meaningful contact regions (head, upper
    trunk/back, lower back, pelvis/butt, thighs, knees, shanks, feet, upper arms, forearms,
    hands).  These are the "keys" the support-attention head attends over.
  * Kinematic snapshot `Schema.kin(qpos)`: point heights / Jacobians, CoM + Jacobian, base
    orientation Jacobian, gravity generalised force g(q) and its derivative G = dg/dq,
    self-collision distance pairs.
  * Exact quasi-static certificate `statics_lp`: for a configuration and a set of contact
    points, the minimum achievable peak torque ratio
        rho* = min_{f,tau} max_j |tau_j|/tau_max_j
        s.t.  sum_p J_p^T f_p + S^T tau = g(q),  f_p in friction pyramid
    (an LP; HiGHS).  rho* <= 0.8 is the F3 keyframe criterion.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import mujoco
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PiPlus, BodyParams, JOINTS, TAU_MAX  # noqa: E402

G = 9.81
MU_PLAN = 0.6          # planning friction (conservative vs the plant's floor mu = 1.0)
EPS_CONTACT = 0.004    # a point within 4 mm of the floor may carry load (force-distance cone)

# region display order (left/right merged in figures via REGION_GROUP)
REGIONS = ["head", "upper_trunk", "lower_back", "pelvis",
           "L_upper_arm", "R_upper_arm", "L_forearm", "R_forearm", "L_hand", "R_hand",
           "L_thigh", "R_thigh", "L_knee", "R_knee", "L_shank", "R_shank", "L_foot", "R_foot"]
REGION_GROUP = {"head": "head", "upper_trunk": "back", "lower_back": "back", "pelvis": "pelvis/butt",
                "L_upper_arm": "arms(elbow)", "R_upper_arm": "arms(elbow)", "L_forearm": "arms(elbow)",
                "R_forearm": "arms(elbow)", "L_hand": "hands", "R_hand": "hands",
                "L_thigh": "thighs", "R_thigh": "thighs", "L_knee": "knees/shanks", "R_knee": "knees/shanks",
                "L_shank": "knees/shanks", "R_shank": "knees/shanks", "L_foot": "feet", "R_foot": "feet"}
GROUPS = ["head", "back", "pelvis/butt", "arms(elbow)", "hands", "thighs", "knees/shanks", "feet"]
# distal effectors that may be *placed* on the floor by a landing sub-task
EFFECTORS = {"L_hand", "R_hand", "L_forearm", "R_forearm", "L_knee", "R_knee", "L_foot", "R_foot"}

# joint chains (for the motion-attention read-out)
CHAINS = {"L_leg": [j for j in JOINTS if j.startswith("l_") and j in JOINTS[:12]],
          "R_leg": [j for j in JOINTS if j.startswith("r_") and j in JOINTS[:12]],
          "L_arm": [j for j in JOINTS if j.startswith("l_") and j in JOINTS[12:]],
          "R_arm": [j for j in JOINTS if j.startswith("r_") and j in JOINTS[12:]]}
JGROUP = {}
for j in JOINTS:
    s = j[0].upper()
    if "hip" in j or "thigh" in j:
        JGROUP[j] = "hip"
    elif "calf" in j:
        JGROUP[j] = "knee"
    elif "ankle" in j:
        JGROUP[j] = "ankle"
    elif "shoulder" in j or "upper_arm" in j:
        JGROUP[j] = "shoulder"
    else:
        JGROUP[j] = "elbow"
JGROUPS = ["hip", "knee", "ankle", "shoulder", "elbow"]


def _region(body, gname, local):
    side = "L" if body.startswith("l_") else "R"
    if body == "base_link":
        if gname == "head":
            return "head"
        z = local[2]
        return "upper_trunk" if z > 0.10 else ("lower_back" if z > -0.005 else "pelvis")
    if "hip_roll" in body or "thigh" in body:
        return f"{side}_thigh"
    if "calf" in body:
        return f"{side}_knee" if gname.endswith("knee") else f"{side}_shank"
    if "ankle" in body:
        return f"{side}_foot"
    if "shoulder" in body or "upper_arm" in body:
        return f"{side}_upper_arm"
    if "elbow" in body:
        return f"{side}_hand" if gname.endswith("hand") else f"{side}_forearm"
    raise ValueError(body)


@dataclass
class ShellPoint:
    body: int
    local: np.ndarray      # centre in body frame
    radius: float
    region: str
    geom: int


@dataclass
class Kin:
    qpos: np.ndarray
    z: np.ndarray            # lowest-point height of each shell point
    xy: np.ndarray           # centre xy of each shell point
    Jp: np.ndarray           # (P, 3, nv) centre Jacobians
    com: np.ndarray
    Jcom: np.ndarray         # 3 x nv
    R: np.ndarray            # base rotation
    Jrot: np.ndarray         # 3 x nv (world angular Jacobian of the base)
    g: np.ndarray            # gravity generalised force (nv)
    Gq: np.ndarray | None    # dg/dq (nv x nv) or None
    pairs: list = field(default_factory=list)   # (dist, n(3), Jdiff(nv)) self-collision rows
    body_pos: dict = field(default_factory=dict)


class Schema:
    def __init__(self, params: BodyParams = BodyParams(), mu=MU_PLAN):
        self.r = PiPlus(params, shell="full")
        self.m = self.r.m
        self.d = mujoco.MjData(self.m)
        self.nv = self.m.nv
        self.mu = mu
        self.M = float(self.m.body_subtreemass[0])
        self.mg = self.M * G
        self.tau_max = TAU_MAX.copy()
        self.jv = self.r.vadr                  # joint dof indices (20)
        self.lo, self.hi = self.r.lo, self.r.hi
        self._build_points()
        self._build_pairs()

    # ------------------------------------------------------------------ shell
    def _build_points(self):
        m = self.m
        fl_ct, fl_ca = m.geom_contype[self.r.floor_gid], m.geom_conaffinity[self.r.floor_gid]
        pts = []
        trunk_corners = []
        for g in range(m.ngeom):
            if g == self.r.floor_gid:
                continue
            if not ((m.geom_contype[g] & fl_ca) or (m.geom_conaffinity[g] & fl_ct)):
                continue
            b = m.geom_bodyid[g]
            bname = m.body(b).name
            gname = m.geom(g).name or ""
            t = m.geom_type[g]
            pos, s = m.geom_pos[g], m.geom_size[g]
            Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g]); Rg = Rg.reshape(3, 3)
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                cs = [pos + Rg @ (s * np.array([a, bb, c])) for a in (-1, 1) for bb in (-1, 1) for c in (-1, 1)]
                if bname == "base_link":
                    trunk_corners += [(c, g, gname) for c in cs]
                else:
                    pts += [ShellPoint(b, c, 0.0, _region(bname, gname, c), g) for c in cs]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                pts.append(ShellPoint(b, pos.copy(), float(s[0]), _region(bname, gname, pos), g))
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ax = Rg[:, 2] * s[1]
                for c in (pos - ax, pos + ax):
                    pts.append(ShellPoint(b, c, float(s[0]), _region(bname, gname, c), g))
        # trunk: keep only convex-hull vertices of the rigid trunk shell (interior corners can
        # never be the lowest point on a flat floor)
        C = np.array([c for c, _, _ in trunk_corners])
        hv = ConvexHull(C).vertices
        for i in hv:
            c, g, gname = trunk_corners[i]
            pts.append(ShellPoint(self.r.base, c, 0.0, _region("base_link", gname, c), g))
        self.points = pts
        self.P = len(pts)
        self.p_region = np.array([REGIONS.index(p.region) for p in pts])
        self.p_body = np.array([p.body for p in pts])
        self.region_points = {r: np.where(self.p_region == i)[0] for i, r in enumerate(REGIONS)}

    def _build_pairs(self):
        """Self-collision pairs exactly as the plant's collision filter would test them."""
        m = self.m
        excl = {(int(s) >> 16, int(s) & 0xFFFF) for s in m.exclude_signature}
        geoms = [g for g in range(m.ngeom) if g != self.r.floor_gid and (m.geom_contype[g] or m.geom_conaffinity[g])]
        pairs = []
        for i, a in enumerate(geoms):
            for b in geoms[i + 1:]:
                ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
                if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                    continue
                if (ba, bb) in excl or (bb, ba) in excl:
                    continue
                if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
                    continue
                pairs.append((a, b))
        self.pairs = pairs

    # ------------------------------------------------------------------ kinematics
    def gravity(self, qpos, d=None):
        d = d or self.d
        d.qpos[:] = qpos; d.qvel[:] = 0
        mujoco.mj_forward(self.m, d)
        return d.qfrc_bias.copy()

    def integrate(self, qpos, dq, scale=1.0):
        q = qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, scale)
        return q

    def kin(self, qpos, with_G=True, with_pairs=True, pair_dist=0.03) -> Kin:
        m, d = self.m, self.d
        d.qpos[:] = qpos; d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        g = d.qfrc_bias.copy()
        P, nv = self.P, self.nv
        z = np.zeros(P); xy = np.zeros((P, 2)); Jp = np.zeros((P, 3, nv))
        jr = np.zeros((3, nv))
        for i, p in enumerate(self.points):
            R = d.xmat[p.body].reshape(3, 3)
            c = d.xpos[p.body] + R @ p.local
            z[i] = c[2] - p.radius
            xy[i] = c[:2]
            mujoco.mj_jac(m, d, Jp[i], jr, c, p.body)
        Jcom = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jcom, 0)
        Jrot = np.zeros((3, nv)); jp = np.zeros((3, nv))
        mujoco.mj_jacBody(m, d, jp, Jrot, self.r.base)
        R = d.xmat[self.r.base].reshape(3, 3).copy()
        com = d.subtree_com[0].copy()
        body_pos = {m.body(b).name: d.xpos[b].copy() for b in range(1, m.nbody)}
        # feet: sole-centre point + orientation Jacobians (foot-flat / placement sub-tasks)
        from model import FEET, SOLE_CENTER, SOLE_Z
        for f in FEET:
            b = m.body(f).id
            Rb = d.xmat[b].reshape(3, 3).copy()
            pc = d.xpos[b] + Rb @ np.array([SOLE_CENTER[0], 0.0, SOLE_Z])
            jp_ = np.zeros((3, nv)); jr_ = np.zeros((3, nv))
            mujoco.mj_jac(m, d, jp_, jr_, pc, b)
            body_pos[f + "/sole"] = (pc, Rb, jp_, jr_)
        pairs = []
        if with_pairs:
            ft = np.zeros(6)
            for a, b in self.pairs:
                dist = mujoco.mj_geomDistance(m, d, a, b, pair_dist, ft)
                if dist >= pair_dist:
                    continue
                p1, p2 = ft[:3].copy(), ft[3:].copy()
                n = p2 - p1
                if np.linalg.norm(n) < 1e-9 or dist < 0:
                    n = d.geom_xpos[b] - d.geom_xpos[a]
                    if dist < 0:
                        p1 = p2 = 0.5 * (d.geom_xpos[a] + d.geom_xpos[b]) if np.linalg.norm(p2 - p1) < 1e-9 else p1
                n = n / max(np.linalg.norm(n), 1e-9)
                J1 = np.zeros((3, nv)); J2 = np.zeros((3, nv))
                mujoco.mj_jac(m, d, J1, jr, p1, m.geom_bodyid[a])
                mujoco.mj_jac(m, d, J2, jr, p2, m.geom_bodyid[b])
                pairs.append((dist, n @ (J2 - J1), (a, b)))
        Gq = None
        if with_G:
            Gq = np.zeros((nv, nv))
            h = 1e-5
            d2 = mujoco.MjData(m) if not hasattr(self, "_d2") else self._d2
            self._d2 = d2
            for i in range(nv):
                e = np.zeros(nv); e[i] = h
                Gq[:, i] = (self.gravity(self.integrate(qpos, e), d2) - g) / h
        return Kin(qpos.copy(), z, xy, Jp, com, Jcom, R, Jrot, g, Gq, pairs, body_pos)

    def region_height(self, k: Kin):
        return np.array([k.z[self.region_points[r]].min() if len(self.region_points[r]) else np.inf
                         for r in REGIONS])

    def region_xy(self, k: Kin):
        out = np.zeros((len(REGIONS), 2))
        for i, r in enumerate(REGIONS):
            idx = self.region_points[r]
            j = idx[np.argmin(k.z[idx])]
            out[i] = k.xy[j]
        return out

    # ------------------------------------------------------------------ statics certificate
    def statics_lp(self, k: Kin, active=None, eps=EPS_CONTACT + 0.001, mu=None, objective="peak"):
        """Exact quasi-static feasibility of configuration k.

        active: boolean mask of contact points allowed to carry load (default: all points
        within eps of the floor).  Returns dict(rho, tau, f, loads per region, binding joint).
        objective="peak": min max_j |tau_j|/tau_max_j (J2);  the reported tau is the
        minimiser (ties broken by a small sum|tau| term)."""
        mu = self.mu if mu is None else mu
        if active is None:
            active = k.z <= eps
        idx = np.where(active)[0]
        nc = len(idx)
        nv = self.nv
        # variables: f (3nc), tau (20), rho (1)
        nx = 3 * nc + 20 + 1
        Aeq = np.zeros((nv, nx)); beq = k.g.copy()
        for c, i in enumerate(idx):
            Aeq[:, 3 * c:3 * c + 3] = k.Jp[i].T
        Aeq[self.jv, 3 * nc + np.arange(20)] = 1.0
        rows, b = [], []
        s = mu / np.sqrt(2)
        for c in range(nc):
            for ax in (0, 1):
                for sg in (1, -1):
                    r = np.zeros(nx); r[3 * c + ax] = sg; r[3 * c + 2] = -s; rows.append(r); b.append(0)
        for j in range(20):
            for sg in (1, -1):
                r = np.zeros(nx); r[3 * nc + j] = sg / self.tau_max[j]; r[-1] = -1; rows.append(r); b.append(0)
        bounds = [(None, None)] * nx
        for c in range(nc):
            bounds[3 * c + 2] = (0, None)
        bounds[-1] = (0, None)
        cost = np.zeros(nx); cost[-1] = 1.0
        cost[3 * nc:3 * nc + 20] = 0.0
        res = linprog(cost, A_ub=np.array(rows) if rows else None, b_ub=np.array(b) if b else None,
                      A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if res.status != 0:
            return dict(feasible=False, rho=np.inf, status=res.status, n_contacts=nc,
                        loads={}, binding=None, tau=None)
        x = res.x
        f = x[:3 * nc].reshape(nc, 3)
        tau = x[3 * nc:3 * nc + 20]
        rho = float(x[-1])
        loads = {}
        for c, i in enumerate(idx):
            loads[self.points[i].region] = loads.get(self.points[i].region, 0.0) + f[c, 2] / self.mg
        ratio = np.abs(tau) / self.tau_max
        return dict(feasible=True, rho=rho, tau=tau, f=f, idx=idx, loads=loads, n_contacts=nc,
                    binding=JOINTS[int(np.argmax(ratio))], ratio=ratio)
