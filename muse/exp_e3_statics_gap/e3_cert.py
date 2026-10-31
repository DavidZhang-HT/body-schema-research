"""E3: port of scheme A's statics certificate (muse/sim/getup_schema.Schema.statics_lp and
muse/sim/getup_mpc.certify) to scheme B's 24-DoF plant (codex/assets/scene.xml, mesh collision).

What is ported 1:1 (same maths, same thresholds):
  * statics LP   rho* = min_{f,tau} max_j |tau_j|/tau_max_j
                 s.t. sum_p J_p^T f_p + S^T tau = g(q)   (g = qfrc_bias at qvel=0)
                      f_p in the 4-sided friction pyramid (mu/sqrt2), f_pz >= 0
                 solved with scipy linprog(method='highs'), as in A.
  * certify()    contact set = shell points within eps_cert = 5 mm of the floor;
                 pen = min shell point height; selfd = min self-collision distance between
                 plant-collidable geom pairs (mj_geomDistance, distmax 1 cm);
                 cert_ok = feasible & rho <= 0.8 & pen >= -1.5 mm & selfd >= -1 mm
                 (A: getup_keyframes.cert_ok, RHO_CAP = 0.8, PEN_OK, SELF_OK).
  * mu_plan = 0.6 (A's MU_PLAN, conservative vs B's floor mu = 0.8); mu = 0.8 also reported.

What had to change because the plant differs (documented in EXP_E3_REPORT.md):
  * Shell points: A reduces primitive boxes/capsules to corners/end-spheres.  B collides with
    meshes (MuJoCo uses each mesh's convex hull) plus 7 trunk boxes and a few ankle/calf boxes.
    Here: box -> 8 corners; mesh -> vertices of its convex hull (exact candidate set for a
    flat floor, since the lowest point of a convex body on a plane is a hull vertex).
  * tau_max = B's actuator ctrlrange (URDF effort): 20 N m legs, 10 N m arms/wrists, 3 N m head
    (24 entries; A had 20 entries with 10 N m arms).
  * Regions: feet = *_ankle_roll_link only (identical to B's metrics(): foot = nodes 6 and 12).
    ankle_pitch guard boxes count as NON-foot (stricter, and consistent with B's nonfoot_load).
  * Self-contacts carry no force in the LP (A's convention).  B's plant does allow self-contact
    forces, so this is conservative (it can only over-estimate rho).
"""
from __future__ import annotations

import numpy as np
import mujoco as mj
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

G = 9.81
MU_PLAN = 0.6
EPS_CERT = 0.005
RHO_CAP = 0.8
PEN_OK, SELF_OK = -0.0015, -0.001


def region_of(bname):
    if bname == 'base_link':
        return 'trunk'
    if bname.startswith('head'):
        return 'head'
    s = 'L' if bname.startswith('l_') else 'R'
    if 'ankle_roll' in bname:
        return s + '_foot'
    if 'ankle_pitch' in bname:
        return s + '_ankle'
    if 'calf' in bname:
        return s + '_shank'
    if 'hip' in bname or 'thigh' in bname:
        return s + '_thigh'
    if 'wrist' in bname:
        return s + '_hand'
    if 'elbow' in bname:
        return s + '_forearm'
    if 'shoulder' in bname or 'upper_arm' in bname:
        return s + '_upper_arm'
    raise ValueError(bname)


FEET = ('L_foot', 'R_foot')


class BSchema:
    """A's Schema/certify, on B's MjModel (a separate MjData; the plant data is never touched)."""

    def __init__(self, m: mj.MjModel, mu=MU_PLAN):
        self.m = m
        self.d = mj.MjData(m)
        self.nv = m.nv
        self.mu = mu
        self.M = float(m.body_subtreemass[1])     # body 0 = world; 1 = base_link subtree
        self.mg = self.M * G
        self.tau_max = np.abs(m.actuator_ctrlrange[:, 1]).copy()   # actuator i <-> joint i+1
        assert m.nu == 24 and all(int(m.actuator_trnid[i, 0]) == i + 1 for i in range(24))
        self.jv = np.array([m.jnt_dofadr[j] for j in range(1, 25)])
        self.floor = m.geom('ground').id
        self._build_points()
        self._build_pairs()

    def _build_points(self):
        m = self.m
        geoms, locs, regs = [], [], []
        for g in range(m.ngeom):
            if g == self.floor or not (m.geom_contype[g] & m.geom_conaffinity[self.floor]
                                       or m.geom_conaffinity[g] & m.geom_contype[self.floor]):
                continue
            b = int(m.geom_bodyid[g]); t = m.geom_type[g]
            Rg = np.zeros(9); mj.mju_quat2Mat(Rg, m.geom_quat[g]); Rg = Rg.reshape(3, 3)
            if t == mj.mjtGeom.mjGEOM_BOX:
                s = m.geom_size[g]
                v = np.array([[a, bb, c] for a in (-1, 1) for bb in (-1, 1) for c in (-1, 1)]) * s
            elif t == mj.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]; a = m.mesh_vertadr[k]; n = m.mesh_vertnum[k]
                V = m.mesh_vert[a:a + n].astype(float)
                v = V[ConvexHull(V).vertices]
            else:
                raise NotImplementedError(t)
            p = m.geom_pos[g] + v @ Rg.T
            geoms += [g] * len(p); locs.append(p); regs += [region_of(m.body(b).name)] * len(p)
        self.p_geom = np.array(geoms)
        self.p_body = m.geom_bodyid[self.p_geom].astype(int)
        self.p_local = np.vstack(locs)
        self.p_region = np.array(regs)
        self.P = len(self.p_geom)
        self.is_foot = np.isin(self.p_region, FEET)

    def _build_pairs(self):
        """Self-collision pairs exactly as MuJoCo's default filter tests them (no parent/child,
        no same body, contype/conaffinity match; B's model has no <exclude>)."""
        m = self.m
        gs = [g for g in range(m.ngeom) if g != self.floor and (m.geom_contype[g] or m.geom_conaffinity[g])]
        pairs = []
        for i, a in enumerate(gs):
            for b in gs[i + 1:]:
                ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
                if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                    continue
                if not (m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]):
                    continue
                pairs.append((a, b))
        self.pairs = pairs

    # ------------------------------------------------------------------ kinematics
    def forward(self, qpos):
        d = self.d
        d.qpos[:] = qpos; d.qvel[:] = 0
        mj.mj_forward(self.m, d)
        return d

    def points_world(self, d=None):
        d = d or self.d
        R = d.xmat[self.p_body].reshape(-1, 3, 3)
        return d.xpos[self.p_body] + np.einsum('pij,pj->pi', R, self.p_local)

    def self_distance(self, d=None, distmax=0.01, return_pair=False):
        d = d or self.d
        ft = np.zeros(6); best = distmax; bp = None
        for a, b in self.pairs:
            if np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) > self.m.geom_rbound[a] + self.m.geom_rbound[b] + distmax:
                continue
            dist = mj.mj_geomDistance(self.m, d, a, b, distmax, ft)
            if dist < best:
                best = dist; bp = (a, b)
        return (best, bp) if return_pair else best

    # ------------------------------------------------------------------ statics LP (A, verbatim maths)
    def statics_lp(self, qpos, active, mu=None):
        mu = self.mu if mu is None else mu
        d = self.forward(qpos)
        g = d.qfrc_bias.copy()
        W = self.points_world(d)
        idx = np.where(active)[0]
        nc = len(idx); nv = self.nv; nj = 24
        nx = 3 * nc + nj + 1
        Aeq = np.zeros((nv, nx))
        jp = np.zeros((3, nv)); jr = np.zeros((3, nv))
        for c, i in enumerate(idx):
            mj.mj_jac(self.m, d, jp, jr, W[i], int(self.p_body[i]))
            Aeq[:, 3 * c:3 * c + 3] = jp.T
        Aeq[self.jv, 3 * nc + np.arange(nj)] = 1.0
        rows = []
        s = mu / np.sqrt(2)
        for c in range(nc):
            for ax in (0, 1):
                for sg in (1, -1):
                    r = np.zeros(nx); r[3 * c + ax] = sg; r[3 * c + 2] = -s; rows.append(r)
        for j in range(nj):
            for sg in (1, -1):
                r = np.zeros(nx); r[3 * nc + j] = sg / self.tau_max[j]; r[-1] = -1; rows.append(r)
        bounds = [(None, None)] * nx
        for c in range(nc):
            bounds[3 * c + 2] = (0, None)
        bounds[-1] = (0, None)
        cost = np.zeros(nx); cost[-1] = 1.0
        res = linprog(cost, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Aeq, b_eq=g,
                      bounds=bounds, method='highs')
        if res.status != 0:
            return dict(feasible=False, rho=float('inf'), status=int(res.status), n_contacts=nc,
                        loads={}, binding=None, tau=None, ratio=None)
        x = res.x
        f = x[:3 * nc].reshape(nc, 3); tau = x[3 * nc:3 * nc + nj]; rho = float(x[-1])
        loads = {}
        for c, i in enumerate(idx):
            loads[self.p_region[i]] = loads.get(self.p_region[i], 0.0) + f[c, 2] / self.mg
        ratio = np.abs(tau) / self.tau_max
        return dict(feasible=True, rho=rho, tau=tau, f=f, idx=idx, loads=loads, n_contacts=nc,
                    binding=self.m.joint(int(np.argmax(ratio)) + 1).name, ratio=ratio)

    # ------------------------------------------------------------------ certify (A, getup_mpc.certify)
    def certify(self, qpos, mu=None, eps_cert=EPS_CERT, feet_only=False):
        d = self.forward(qpos)
        W = self.points_world(d); z = W[:, 2]
        active = z <= eps_cert
        if feet_only:
            active = active & self.is_foot
        lp = self.statics_lp(qpos, active, mu)
        d = self.forward(qpos)
        loaded = sorted({str(r) for r, v in lp.get('loads', {}).items() if v > 0.01})
        pen = float(z.min())
        selfd, sp = self.self_distance(d, return_pair=True)
        qj = qpos[7:]
        lo = self.m.jnt_range[1:, 0]; hi = self.m.jnt_range[1:, 1]
        jl = float(np.min(np.minimum(qj - lo, hi - qj)))
        touching = sorted({str(r) for r in self.p_region[z <= eps_cert]})
        c = dict(feasible=bool(lp['feasible']), rho=float(lp['rho']), binding=lp.get('binding'),
                 loads={str(k): round(float(v), 4) for k, v in lp.get('loads', {}).items()},
                 loaded=loaded, touching=touching, n_contacts=int(lp['n_contacts']),
                 feet_only=bool(loaded) and set(loaded) <= set(FEET), pen=pen, selfd=float(selfd),
                 self_pair=None if sp is None else [self.m.body(self.m.geom_bodyid[g]).name for g in sp],
                 jl=jl, com=d.subtree_com[1].tolist(), mu=self.mu if mu is None else mu,
                 restricted_to_feet=feet_only,
                 ratio=None if lp.get('ratio') is None else [round(float(r), 4) for r in lp['ratio']])
        c['ok'] = cert_ok(c)
        c['why'] = cert_why(c)
        return c


def cert_ok(c):
    return bool(c['feasible']) and c['rho'] <= RHO_CAP and c['pen'] >= PEN_OK and c['selfd'] >= SELF_OK


def cert_why(c):
    w = []
    if not c['feasible']:
        w.append('statics LP infeasible (no friction-cone forces balance gravity)')
    elif c['rho'] > RHO_CAP:
        w.append(f"rho {c['rho']:.2f} > {RHO_CAP}")
    if c['pen'] < PEN_OK:
        w.append(f"penetration {1e3 * c['pen']:.1f} mm")
    if c['selfd'] < SELF_OK:
        w.append(f"self-collision {1e3 * c['selfd']:.1f} mm")
    return '; '.join(w) or 'ok'
