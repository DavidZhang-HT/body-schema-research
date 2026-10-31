"""Phase 1 - kinematic retargeting of a LAFAN1 human walk (BVH) onto pi+.

Pipeline
  1. parse BVH (Y-up, cm, ZYX euler), convert to Z-up metres
  2. pick a steady straight-walking window (inside the first 33 s so that it can be
     cross-checked against HighTorque's own GMR retarget CSV)
  3. per-frame damped-least-squares IK on the pi+ kinematic tree:
       ankle-pitch-axis position relative to hip (scaled by leg-length ratio),
       sole pitch from the human foot->toe vector, sole roll = 0,
       elbow / wrist directions (arm-length normalised)
     joint limits enforced by projection; lateral foot clearance enforced from the
     self-collision boundary computed in step 5
  4. phase-average N strides into one periodic stride q_ref(phi), phi in [0,1)
     (phi=0: left touchdown) + nominal CoM trajectory + contact schedule
  5. reachable-space + self-collision boundary of the legs by sampling the URDF limits
Outputs: data/ref_walk.npz, data/phase1_summary.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PiPlus, JOINTS, ROOT, FEET  # noqa: E402

DATA = os.path.join(ROOT, "data")


# ----------------------------------------------------------------------------- BVH
def parse_bvh(path):
    lines = open(path).read().split("\n")
    names, parents, offsets, chans = [], [], [], []
    stack = []
    i = 0
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); chans.append([])
            cur = len(names) - 1
        elif tok and tok[0] == "End":
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1])
            offsets.append(None); chans.append([]); cur = len(names) - 1
        elif tok and tok[0] == "{":
            stack.append(cur)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET":
            offsets[cur] = np.array([float(x) for x in tok[1:4]])
        elif tok and tok[0] == "CHANNELS":
            chans[cur] = tok[2:]
        i += 1
    nframes = int(lines[i + 1].split()[1]); ft = float(lines[i + 2].split()[2])
    mot = np.array([[float(x) for x in l.split()] for l in lines[i + 3:i + 3 + nframes] if l.strip()])
    return names, np.array(parents), np.array(offsets), chans, mot, ft


def _rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bvh_fk(names, parents, offsets, chans, mot, frames):
    """Global joint positions (F x J x 3) in Z-up metres."""
    J = len(names)
    out = np.zeros((len(frames), J, 3))
    Rup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])    # Y-up -> Z-up
    for fi, f in enumerate(frames):
        row = mot[f]; c = 0
        Rg = [None] * J; pg = [None] * J
        for j in range(J):
            pos = offsets[j].copy(); R = np.eye(3)
            for ch in chans[j]:
                v = row[c]; c += 1
                if ch.endswith("position"):
                    pos["XYZ".index(ch[0])] = v
                else:
                    R = R @ _rot(ch[0], np.deg2rad(v))
            if parents[j] < 0:
                Rg[j], pg[j] = R, pos
            else:
                p = parents[j]
                pg[j] = pg[p] + Rg[p] @ pos; Rg[j] = Rg[p] @ R
        out[fi] = (np.array(pg) @ Rup.T) / 100.0
    return out


# ----------------------------------------------------------------------------- IK
class IK:
    def __init__(self):
        self.r = PiPlus()
        self.m, self.d = self.r.m, self.r.d
        m = self.m
        self.b = {n: m.body(n).id for n in ["l_hip_roll_link", "r_hip_roll_link", "l_ankle_pitch_link",
                                            "r_ankle_pitch_link", "l_ankle_roll_link", "r_ankle_roll_link",
                                            "l_shoulder_roll_link", "r_shoulder_roll_link",
                                            "l_elbow_link", "r_elbow_link"]}
        self.r.reset()
        self.q = self.r.default_q()
        d = self.d
        mujoco.mj_kinematics(m, d)
        self.leg_len = np.linalg.norm(d.xpos[self.b["l_hip_roll_link"]] - d.xpos[self.b["l_ankle_pitch_link"]])
        self.leg_len = 0.0977 + 0.045 + 0.14   # straight-leg hip-roll -> ankle-pitch length
        self.ua_len = 0.0724 + 0.038
        self.fa_len = 0.0724
        self.jacp = np.zeros((3, m.nv)); self.jacr = np.zeros((3, m.nv))

    def _set(self, q):
        d = self.d
        d.qpos[:3] = 0; d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.r.qadr] = q
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)

    def point_jac(self, body, local=np.zeros(3)):
        d = self.d
        p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
        mujoco.mj_jac(self.m, d, self.jacp, self.jacr, p, body)
        return p, self.jacp[:, self.r.vadr].copy(), self.jacr[:, self.r.vadr].copy()

    def solve(self, targets, q0, iters=25):
        """targets: list of (kind, body, local, target, weight); kind 'p' position, 'R' z/x-axis align."""
        q = q0.copy()
        for _ in range(iters):
            self._set(q)
            rows, errs = [], []
            for kind, body, local, tgt, w in targets:
                if kind == "p":
                    p, Jp, _ = self.point_jac(body, local)
                    rows.append(w * Jp); errs.append(w * (tgt - p))
                else:  # orientation: align body frame with target rotation matrix
                    R = self.d.xmat[body].reshape(3, 3)
                    _, _, Jr = self.point_jac(body)
                    e = 0.5 * (np.cross(R[:, 0], tgt[:, 0]) + np.cross(R[:, 1], tgt[:, 1]) + np.cross(R[:, 2], tgt[:, 2]))
                    rows.append(w * Jr); errs.append(w * e)
            J = np.vstack(rows); e = np.concatenate(errs)
            # regulariser towards the default posture (resolves thigh-yaw redundancy)
            lam = 1e-3
            dq = np.linalg.solve(J.T @ J + lam * np.eye(20), J.T @ e + 1e-4 * (self.r.default_q() - q))
            q = np.clip(q + dq, self.r.lo + 1e-3, self.r.hi - 1e-3)
            if np.linalg.norm(e) < 1e-4:
                break
        return q, float(np.linalg.norm(e))


def heading_frame(left_hip, right_hip):
    lv = left_hip - right_hip; lv[2] = 0; lv /= np.linalg.norm(lv)
    fwd = np.cross(lv, [0, 0, 1])
    return np.column_stack([fwd, lv, [0, 0, 1]])


def rot_y(a):
    return _rot("Y", a)


def main():
    names, parents, offsets, chans, mot, ft = parse_bvh(os.path.join(DATA, "walk1_subject1.bvh"))
    idx = {n: i for i, n in enumerate(names)}
    fps = 1.0 / ft
    # ---- choose a steady straight window inside the first 1000 frames (GMR overlap)
    allf = np.arange(0, 1000)
    P = bvh_fk(names, parents, offsets, chans, mot, allf)
    hips = P[:, idx["Hips"]]
    vel = np.gradient(hips[:, :2], ft, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    head = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    W = int(5.0 * fps)
    best, bscore = None, 1e9
    for s in range(0, len(allf) - W, 5):
        sp = speed[s:s + W]; hd = head[s:s + W]
        if sp.mean() < 0.8:
            continue
        score = np.std(sp) + 2.0 * (hd.max() - hd.min())
        if score < bscore:
            best, bscore = s, score
    s0, s1 = best, best + W
    print(f"window frames {s0}-{s1} ({s0/fps:.1f}-{s1/fps:.1f}s) speed {speed[s0:s1].mean():.2f} m/s "
          f"heading range {np.rad2deg(np.ptp(head[s0:s1])):.1f} deg")
    Pw = P[s0:s1]
    F = len(Pw)

    ik = IK()
    human_leg = np.mean(np.linalg.norm(Pw[:, idx["LeftUpLeg"]] - Pw[:, idx["LeftLeg"]], axis=1) +
                        np.linalg.norm(Pw[:, idx["LeftLeg"]] - Pw[:, idx["LeftFoot"]], axis=1))
    s_leg = ik.leg_len / human_leg
    print(f"human leg {human_leg:.3f} m, pi+ leg {ik.leg_len:.3f} m, scale {s_leg:.3f}")

    # ---- foot contact labels from the human (height + speed of ankle & toe)
    contact = np.zeros((F, 2), bool)
    for k, side in enumerate(("Left", "Right")):
        toe = Pw[:, idx[f"{side}Toe"]]; ank = Pw[:, idx[f"{side}Foot"]]
        zt = np.minimum(toe[:, 2], ank[:, 2])
        vt = np.linalg.norm(np.gradient(toe[:, :2], ft, axis=0), axis=1)
        contact[:, k] = (zt < np.percentile(zt, 35) + 0.02) & (vt < 0.6)
    # median stance foot pitch offset (human foot->toe vector is not parallel to ground)
    def foot_pitch(side, f):
        v = Pw[f, idx[f"{side}Toe"]] - Pw[f, idx[f"{side}Foot"]]
        Rh = heading_frame(Pw[f, idx["LeftUpLeg"]], Pw[f, idx["RightUpLeg"]])
        vl = Rh.T @ v
        return np.arctan2(-vl[2], vl[0])
    pitch0 = {s: np.median([foot_pitch(s, f) for f in range(F) if contact[f, k]]) for k, s in enumerate(("Left", "Right"))}

    # ---- self-collision boundary: min lateral distance between ankle centres
    w_min = self_collision_boundary(ik)
    print(f"self-collision boundary: min ankle lateral separation {w_min*100:.1f} cm")

    # ---- per-frame IK
    Q = np.zeros((F, 20)); trunk_pitch = np.zeros(F); errs = np.zeros(F)
    q = ik.r.default_q()
    bl = ik.b
    ik._set(q)
    hipL0 = ik.d.xpos[bl["l_hip_roll_link"]].copy(); hipR0 = ik.d.xpos[bl["r_hip_roll_link"]].copy()
    shL0 = ik.d.xpos[bl["l_shoulder_roll_link"]].copy(); shR0 = ik.d.xpos[bl["r_shoulder_roll_link"]].copy()
    for f in range(F):
        Rh = heading_frame(Pw[f, idx["LeftUpLeg"]], Pw[f, idx["RightUpLeg"]])
        loc = lambda n: Rh.T @ (Pw[f, idx[n]] - Pw[f, idx["Hips"]])   # heading-frame, pelvis-relative
        # human trunk pitch (pelvis->neck) w.r.t. vertical: pi+ has no waist, so the base takes it
        tv = loc("Neck") - loc("Hips")
        trunk_pitch[f] = np.arctan2(tv[0], tv[2])
        Rb = rot_y(trunk_pitch[f])            # base orientation in heading frame
        targets = []
        for side, s, hip0, sh0 in (("Left", "l", hipL0, shL0), ("Right", "r", hipR0, shR0)):
            # leg: ankle relative to hip, scaled, expressed in heading frame -> base frame
            a = (loc(f"{side}Foot") - loc(f"{side}UpLeg")) * s_leg
            a_b = Rb.T @ a
            tgt = hip0 + a_b
            # enforce self-collision boundary (lateral ankle separation >= w_min)
            sign = 1 if s == "l" else -1
            tgt[1] = sign * max(sign * tgt[1], w_min / 2)
            targets.append(("p", bl[f"{s}_ankle_pitch_link"], np.zeros(3), tgt, 1.0))
            fp = foot_pitch(side, f) - pitch0[side]
            Rf = Rb.T @ rot_y(fp)
            targets.append(("R", bl[f"{s}_ankle_roll_link"], None, Rf, 0.15))
            # arm: direction matching
            ue = loc(f"{side}ForeArm") - loc(f"{side}Arm"); ue /= np.linalg.norm(ue)
            uh = loc(f"{side}Hand") - loc(f"{side}ForeArm"); uh /= np.linalg.norm(uh)
            el = sh0 + ik.ua_len * (Rb.T @ ue)
            wr = el + ik.fa_len * (Rb.T @ uh)
            targets.append(("p", bl[f"{s}_elbow_link"], np.zeros(3), el, 0.3))
            targets.append(("p", bl[f"{s}_elbow_link"], np.array([0, 0, -0.0724]), wr, 0.2))
        q, e = ik.solve(targets, q, iters=40 if f == 0 else 15)
        Q[f] = q; errs[f] = e

    # ---- root trajectory (pelvis xy scaled, height from lowest sole point), CoM, self-collision
    rob = PiPlus()
    m, d = rob.m, rob.d
    root = np.zeros((F, 3)); com = np.zeros((F, 3)); ncoll = 0; feet = np.zeros((F, 2, 3))
    hh = (Pw[:, idx["Hips"]] - Pw[0, idx["Hips"]]) * s_leg
    for f in range(F):
        d.qpos[:3] = [hh[f, 0], hh[f, 1], 1.0]
        yaw = 0.0
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, (rot_y(trunk_pitch[f])).flatten())
        d.qpos[3:7] = quat; d.qpos[rob.qadr] = Q[f]
        mujoco.mj_kinematics(m, d)
        sp = rob.sole_points(d)
        d.qpos[2] -= sp[:, 2].min()
        mujoco.mj_forward(m, d)
        root[f] = d.qpos[:3]; com[f] = d.subtree_com[0]
        feet[f, 0] = d.xpos[rob.foot_bid[0]]; feet[f, 1] = d.xpos[rob.foot_bid[1]]
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if rob.floor_gid not in (g1, g2):
                ncoll += 1; break

    # ---- gait cycle extraction (left touchdowns)
    td = [f for f in range(1, F) if contact[f, 0] and not contact[f - 1, 0]]
    tdR = [f for f in range(1, F) if contact[f, 1] and not contact[f - 1, 1]]
    strides = list(zip(td[:-1], td[1:]))
    periods = np.array([b - a for a, b in strides]) / fps
    print(f"left touchdowns {td}, stride periods {np.round(periods,3)}")
    NPH = 100
    cyc = []
    for a, b in strides:
        tt = np.linspace(a, b, NPH, endpoint=False)
        cyc.append(np.column_stack([np.interp(tt, np.arange(F), Q[:, j]) for j in range(20)]))
    cyc = np.array(cyc)
    q_cycle = cyc.mean(0)
    q_cycle_std = cyc.std(0)
    # duty factor & double support from contact labels
    duty = contact[td[0]:td[-1], 0].mean()
    ds = (contact[td[0]:td[-1], 0] & contact[td[0]:td[-1], 1]).mean()
    stride_T = periods.mean()
    hip_speed = speed[s0:s1].mean()
    stride_len_h = hip_speed * stride_T
    stride_len_r = stride_len_h * s_leg
    step_w_h = np.mean([abs((Rh := heading_frame(Pw[f, idx["LeftUpLeg"]], Pw[f, idx["RightUpLeg"]])).T @
                            (Pw[f, idx["LeftFoot"]] - Pw[f, idx["RightFoot"]]))[1] for f in range(F)])
    froude_T = stride_T * np.sqrt(s_leg)    # dynamic similarity (Froude) scaling of time
    summary = dict(
        source="LAFAN1 walk1_subject1.bvh via HighTorque Mini-Pi-Plus_BeyondMimic repo",
        window_frames=[int(s0), int(s1)], fps=fps, human_speed=float(hip_speed),
        human_leg=float(human_leg), robot_leg=float(ik.leg_len), scale=float(s_leg),
        human_stride_period=float(stride_T), froude_stride_period=float(froude_T),
        human_stride_len=float(stride_len_h), geom_scaled_stride_len=float(stride_len_r),
        human_step_width=float(step_w_h), scaled_step_width=float(step_w_h * s_leg),
        duty_factor=float(duty), double_support_frac=float(ds),
        trunk_pitch_mean_deg=float(np.rad2deg(trunk_pitch.mean())),
        ik_residual_mean=float(errs.mean()), ik_residual_max=float(errs.max()),
        frames_with_self_or_limb_contact=int(ncoll), n_frames=int(F),
        self_collision_min_ankle_sep=float(w_min),
        n_strides=len(strides), q_cycle_std_mean_deg=float(np.rad2deg(q_cycle_std.mean())),
        com_height_mean=float(com[:, 2].mean()),
    )
    # ---- joint-limit proximity
    margin = np.minimum(Q - rob.lo, rob.hi - Q)
    summary["min_joint_limit_margin_deg"] = {JOINTS[j]: float(np.rad2deg(margin[:, j].min())) for j in range(20)}
    # ---- reachable space
    summary["reach"] = reachable_space(ik)
    # ---- GMR cross-check
    summary["gmr_check"] = gmr_crosscheck(Q, s0, s1)
    np.savez(os.path.join(DATA, "ref_walk.npz"), Q=Q, root=root, com=com, contact=contact, feet=feet,
             trunk_pitch=trunk_pitch, q_cycle=q_cycle, q_cycle_std=q_cycle_std, fps=fps,
             joints=np.array(JOINTS), stride_T=stride_T, s_leg=s_leg)
    json.dump(summary, open(os.path.join(DATA, "phase1_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("min_joint_limit_margin_deg",)}, indent=1))


def self_collision_boundary(ik):
    """Bring the ankles together (hip roll inward) until any limb/sole geoms touch."""
    r = PiPlus(); m, d = r.m, r.d
    li, ri = JOINTS.index("l_hip_roll_joint"), JOINTS.index("r_hip_roll_joint")
    q = r.default_q()
    best = None
    for a in np.linspace(0, 0.3, 61):
        qq = q.copy(); qq[li] = -a; qq[ri] = a
        d.qpos[:3] = [0, 0, 1]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[r.qadr] = qq
        mujoco.mj_forward(m, d)
        hit = any(r.floor_gid not in (c.geom1, c.geom2) for c in d.contact[:d.ncon])
        sep = d.xpos[m.body("l_ankle_pitch_link").id][1] - d.xpos[m.body("r_ankle_pitch_link").id][1]
        if hit:
            break
        best = sep
    return float(best) + 0.01   # +1 cm safety margin


def reachable_space(ik, n=20000, seed=0):
    """Monte-Carlo foot reachable set (ankle-pitch axis rel. to base) within URDF limits, no self-contact."""
    rng = np.random.default_rng(seed)
    r = PiPlus(); m, d = r.m, r.d
    lo, hi = r.lo.copy(), r.hi.copy()
    # restrict to a walking-relevant envelope of the URDF range (hip pitch / knee forward half-space)
    pts, ok = [], 0
    li = [JOINTS.index(j) for j in JOINTS[:6]]
    for _ in range(n):
        q = r.default_q()
        q[li] = rng.uniform(lo[li], hi[li])
        d.qpos[:3] = [0, 0, 1]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[r.qadr] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
        if any(r.floor_gid not in (c.geom1, c.geom2) for c in d.contact[:d.ncon]):
            continue
        ok += 1
        pts.append(d.xpos[m.body("l_ankle_pitch_link").id] - np.array([0, 0, 1]))
    pts = np.array(pts)
    below = pts[pts[:, 2] < -0.15]
    return dict(samples=n, collision_free_frac=ok / n,
                x_range_below_hip=[float(below[:, 0].min()), float(below[:, 0].max())],
                y_range_below_hip=[float(below[:, 1].min()), float(below[:, 1].max())],
                z_min=float(pts[:, 2].min()))


def gmr_crosscheck(Q, s0, s1):
    path = os.path.join(DATA, "gmr_pi_plus_walk1_subject1.csv")
    G = np.genfromtxt(path, delimiter=",", skip_header=1)
    hdr = open(path).readline().strip().split(",")
    out = {"rows": int(len(G))}
    if len(G) < s1:
        out["note"] = "GMR csv shorter than window"; return out
    gj = {h + "_joint": i for i, h in enumerate(hdr)}
    rms = {}
    for j, n in enumerate(JOINTS):
        if n in gj:
            g = G[s0:s1, gj[n]]
            rms[n] = float(np.rad2deg(np.sqrt(np.mean((g - Q[:, j]) ** 2))))
            # correlation of the motion shape (offset-free)
    corr = {n: float(np.corrcoef(G[s0:s1, gj[n]], Q[:, j])[0, 1]) for j, n in enumerate(JOINTS) if n in gj and np.std(Q[:, j]) > 1e-3}
    out["rms_deg"] = rms; out["corr"] = corr
    out["leg_rms_mean_deg"] = float(np.mean([rms[n] for n in JOINTS[:12]]))
    out["sagittal_corr_mean"] = float(np.mean([corr[n] for n in JOINTS[:12] if ("pitch" in n or "calf" in n) and n in corr]))
    return out


if __name__ == "__main__":
    main()
