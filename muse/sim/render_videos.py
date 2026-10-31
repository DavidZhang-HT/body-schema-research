"""Render side-view stick-figure videos of actual experiment episodes (headless, CPU-only).

Fidelity approach: monkeypatch experiment.episode's internals (RealRobot.step for
30 fps xpos logging, Planner.start_walk / PiPlus.set_mu for stage events) so the
simulated trajectory is bit-identical to the published experiment runs.
Renders with matplotlib Agg (no GL needed), encodes with ffmpeg.
Usage: .venv/bin/python render_videos.py [fixed|active_nodc] [seed]
Writes ../videos/gait_<cond>_s<seed>.mp4
"""
import os, sys, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiment as E
from model import RealRobot, PiPlus
from wbc import Planner

HERE = os.path.dirname(os.path.abspath(__file__))
VIDDIR = os.path.join(HERE, "..", "videos")
os.makedirs(VIDDIR, exist_ok=True)

SEGS = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),
        (1,12),(12,13),(13,14),(14,15),(15,16),(16,17),
        (1,8),(8,9),(9,10),(10,11),
        (1,18),(18,19),(19,20),(20,21)]
LEG_L = set([(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
LEG_R = set([(1,12),(12,13),(13,14),(14,15),(15,16),(16,17)])

frames, events = [], []


class LoggingRobot(RealRobot):
    def step(self, tau):
        o = super().step(tau)
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 33 == 1:
            frames.append((self.t, self.d.xpos.copy(), self.fallen()))
        return o


_orig_set_mu = PiPlus.set_mu
def _set_mu(self, mu):
    events.append((float(self.d.time), "fwd"))
    return _orig_set_mu(self, mu)


_orig_start_walk = Planner.start_walk
def _start_walk(self, t0, *a, **k):
    events.append((float(t0), "walk"))
    return _orig_start_walk(self, t0, *a, **k)


def stage_at(t):
    # walk events: first = sip begin, mu-change = fwd begin
    walks = sorted(e[0] for e in events if e[1] == "walk")
    fwds = sorted(e[0] for e in events if e[1] == "fwd")
    if fwds and t >= fwds[0]:
        return "fwd"
    if walks and t >= walks[0]:
        return "sip"
    if t >= 7.0:
        return "calib"
    if t >= 1.0:
        return "calib"
    return "stand"


def draw_video(out_mp4, title, fell):
    fr = list(frames)
    if fell and fr:
        t_last, xp_last, _ = fr[-1]
        fr += [(t_last, xp_last, True)] * 30  # 1 s freeze on the fall
    tmpd = out_mp4 + ".frames"
    os.makedirs(tmpd, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    for i, (t, xp, f) in enumerate(fr):
        ax.clear()
        bx = xp[1, 0]
        for (a, b) in SEGS:
            xs = [xp[a, 0] - bx, xp[b, 0] - bx]
            zs = [xp[a, 2], xp[b, 2]]
            col = "#1f77b4" if (a, b) in LEG_L else ("#ff7f0e" if (a, b) in LEG_R else "#2ca02c")
            ax.plot(xs, zs, color=col, lw=3, solid_capstyle="round")
        ax.plot(xp[1, 0] - bx, xp[1, 2], "ko", ms=8)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.05, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("forward [m] (camera follows base)")
        ax.set_ylabel("height [m]")
        stat = "FELL" if f else stage_at(t)
        ax.set_title(f"{title}   t={t:.2f}s   {stat}", fontsize=11,
                     color="red" if f else "black")
        ax.text(0.02, 0.95, "blue=left leg  orange=right leg  green=arms",
                transform=ax.transAxes, fontsize=8, va="top")
        fig.savefig(os.path.join(tmpd, f"f{i:04d}.png"))
        if i % 150 == 0:
            print(f"  frame {i}/{len(fr)}", flush=True)
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
                    "-i", os.path.join(tmpd, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4],
                   check=True)
    shutil.rmtree(tmpd)
    print("wrote", out_mp4, os.path.getsize(out_mp4) // 1024, "KB")


if __name__ == "__main__":
    cond = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    E.RealRobot = LoggingRobot
    PiPlus.set_mu = _set_mu
    Planner.start_walk = _start_walk
    print(f"running episode {cond} seed {seed} ...", flush=True)
    res = E.episode(cond, seed, stride=0.06)
    print(f"done: {len(frames)} frames, fell={res['fell']}, "
          f"stage={res['stage_reached']}, t_end={res['t_end']:.2f}", flush=True)
    out = os.path.join(VIDDIR, f"gait_{cond}_s{seed}.mp4")
    draw_video(out, f"pi+ body-schema gait [{cond} seed {seed}]", res["fell"])
