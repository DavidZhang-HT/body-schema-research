"""Render E2 step2/step3 trajectories as side-view (y-z) stick videos, matplotlib Agg."""
import os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/hatch/workspace/research/body-schema-gait/muse/sim")
from model import build_xml

model = mj.MjModel.from_xml_string(build_xml(timestep=0.001, shell="full"))
data = mj.MjData(model)
NB = model.nbody
PARENT = model.body_parentid.copy()

def frame_xyz(qpos):
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    return data.xpos.copy()  # NB x 3

def render(npz_path, out_mp4, title, fps=30, show_r21=True):
    d = np.load(npz_path)
    q = d["qpos"]
    n = q.shape[0]
    step = max(1, int(round(50 / fps)))
    idx = list(range(0, n, step))
    tmpdir = "/tmp/e2vid"
    os.makedirs(tmpdir, exist_ok=True)
    # precompute R[2,1] roll signal from base body xmat
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    r21 = []
    for i in idx:
        data.qpos[:] = q[i]
        mj.mj_forward(model, data)
        R = data.xmat[base_id].reshape(3, 3)
        r21.append(R[2, 1])
    for k, i in enumerate(idx):
        xyz = frame_xyz(q[i])
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        for b in range(1, NB):
            p = PARENT[b]
            ax.plot([xyz[p, 1], xyz[b, 1]], [xyz[p, 2], xyz[b, 2]], "k-", lw=1.5)
        ax.plot(xyz[1:, 1], xyz[1:, 2], "o", ms=3, color="tab:blue")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.05, 1.0)
        ax.set_aspect("equal")
        t = i / 50.0
        ax.set_title(f"{title}  t={t:.2f}s" + (f"  R21={r21[k]:+.2f}" if show_r21 else ""))
        ax.set_xlabel("left [m] (view from front)"); ax.set_ylabel("height [m]")
        fig.savefig(f"{tmpdir}/f{k:04d}.png", dpi=80)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmpdir}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4],
                   check=True, capture_output=True)
    subprocess.run(["rm", "-rf", tmpdir])
    print("wrote", out_mp4)

if __name__ == "__main__":
    render(HERE + "/step2/traj_step2_clean_R_a.npz",
           HERE + "/e2_step2_roll_success.mp4",
           "E2 step2: dynamic roll SUPINE->SIDE (clean_R_a)")
    render(HERE + "/step3/traj_step3_clean_R_a.npz",
           HERE + "/e2_step3_capped_fail.mp4",
           "E2 step3: KE-capped (<=0.005J), no roll (clean_R_a)")
