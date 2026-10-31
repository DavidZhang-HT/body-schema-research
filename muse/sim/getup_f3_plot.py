"""F3 attention visualisation (matplotlib Agg, no GL).

python getup_f3_plot.py TAG  ->  results/f3/attention_TAG.png  +  attention_TAG_table.md

Panels (x = MPC tick, phase boundaries marked):
  1. SUPPORT attention a_cur grouped by body part (head/back/pelvis/arms/hands/thighs/knees/feet)
  2. certified load share per body part (exact statics LP) -- the physical quantity a_cur should track
  3. MOTION attention a_joint grouped by joint type (hip/knee/ankle/shoulder/elbow)
  4. certified peak torque ratio rho with the 0.8 cap and the binding joint group
The table gives the phase-averaged weights (the per-keyframe attention matrices).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from getup_schema import REGIONS, REGION_GROUP, GROUPS, JGROUP, JGROUPS  # noqa: E402
from model import JOINTS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "f3")


def group(M, names, gmap, gnames):
    out = np.zeros((M.shape[0], len(gnames)))
    for i, n in enumerate(names):
        out[:, gnames.index(gmap[n])] += np.nan_to_num(M[:, i])
    return out


def main(tag):
    d = np.load(os.path.join(ROOT, f"run_{tag}.npz"))
    summ = json.load(open(os.path.join(ROOT, f"run_{tag}.json")))
    ph = d["phase"]
    T = np.arange(len(ph))
    aG = group(d["a_cur"], REGIONS, REGION_GROUP, GROUPS)
    adG = group(d["a_des"], REGIONS, REGION_GROUP, GROUPS)
    lG = group(d["loads"], REGIONS, REGION_GROUP, GROUPS)
    lG = lG / np.maximum(lG.sum(1, keepdims=True), 1e-9)
    jG = group(d["a_joint"], JOINTS, JGROUP, JGROUPS)
    eG = group(d["alpha"], JOINTS, JGROUP, JGROUPS)
    eG = eG / np.maximum(eG.sum(1, keepdims=True), 1e-9)
    phases = list(dict.fromkeys(ph.tolist()))
    bounds = [int(np.where(ph == p)[0][0]) for p in phases]

    cols = plt.cm.tab10(np.arange(10))
    fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    def deco(a, title):
        for b, p in zip(bounds, phases):
            a.axvline(b, color="k", lw=0.8, ls="--")
            a.text(b + 1, a.get_ylim()[1] * 0.97, p, fontsize=8, va="top")
        a.set_title(title, fontsize=10)
    ax[0].stackplot(T, aG.T, labels=GROUPS, colors=cols[:len(GROUPS)])
    ax[0].set_ylim(0, 1); ax[0].legend(loc="upper right", fontsize=7, ncol=4)
    deco(ax[0], "Head 1 SUPPORT attention a_cur (contact-link attention), grouped by body part")
    ax[1].stackplot(T, lG.T, labels=GROUPS, colors=cols[:len(GROUPS)])
    ax[1].set_ylim(0, 1)
    deco(ax[1], "Certified load share per body part (exact statics LP) -- physical ground truth")
    ax[2].stackplot(T, jG.T, labels=JGROUPS, colors=cols[[0, 1, 2, 3, 4]])
    ax[2].set_ylim(0, 1); ax[2].legend(loc="upper right", fontsize=7, ncol=5)
    deco(ax[2], "Head 2 MOTION attention a_joint (joint attention for the phase's region-to-target task)")
    ax[3].stackplot(T, eG.T, labels=JGROUPS, colors=cols[[0, 1, 2, 3, 4]])
    ax[3].set_ylim(0, 1)
    deco(ax[3], "Head 3 EFFORT attention alpha (joint marginal, normalised) -- bottleneck joint")
    ax[4].plot(T, d["rho"], "k-", lw=1, label="certified peak |tau|/tau_max")
    ax[4].axhline(0.8, color="r", ls=":", label="F3 cap 0.8")
    ax[4].set_ylim(0, 1.05); ax[4].legend(loc="upper right", fontsize=8)
    ax2 = ax[4].twinx(); ax2.plot(T, d["z"], "b-", lw=0.8); ax2.set_ylabel("pelvis z [m]", color="b")
    deco(ax[4], f"rho and pelvis height  --  verdict {summ['verdict']}")
    ax[4].set_xlabel("MPC tick (quasi-static step)")
    fig.tight_layout()
    png = os.path.join(ROOT, f"attention_{tag}.png")
    fig.savefig(png, dpi=110)

    # ---------------- per-phase (keyframe) attention matrices
    lines = [f"# F3 attention per phase (run {tag}, verdict {summ['verdict']})", "",
             "Phase-mean of each head; rows = phase, columns = body-part group / joint group.", ""]
    def table(title, M, names):
        lines.append(f"## {title}")
        lines.append("| phase | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for p in phases:
            m = M[ph == p].mean(0)
            lines.append(f"| {p} | " + " | ".join(f"{v:.2f}" for v in m) + " |")
        # first tick (entry) and last tick
        lines.append("")
    table("Support attention a_cur (contact-link)", aG, GROUPS)
    table("Support attention a_des (desired, contact-agnostic)", adG, GROUPS)
    table("Certified load share (statics LP)", lG, GROUPS)
    table("Motion attention a_joint", jG, JGROUPS)
    table("Effort attention alpha (normalised joint marginal)", eG, JGROUPS)
    # entry/exit snapshots
    lines.append("## Snapshots at phase entry / exit (a_cur argmax group, load-share argmax group)")
    lines.append("| phase | entry a_cur top | entry load top | exit a_cur top | exit load top |")
    lines.append("|---|---|---|---|---|")
    for p in phases:
        ii = np.where(ph == p)[0]
        f = lambda M, i: f"{GROUPS[int(np.argmax(M[i]))]} ({M[i].max():.2f})"
        lines.append(f"| {p} | {f(aG, ii[0])} | {f(lG, ii[0])} | {f(aG, ii[-1])} | {f(lG, ii[-1])} |")
    with open(os.path.join(ROOT, f"attention_{tag}_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", png)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a")
