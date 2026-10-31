"""Minimal external reference that closes the common-mode zero-offset nullspace.

The scan (zero_motion_scan.py) + sim validation (zero_motion_validate.py) show that no safe
double-support motion brings the common-mode calf-vs-ankle-pitch allocation below the 2 deg
prior.  Physically the missing quantity is the BASE POSE RELATIVE TO THE (common) FEET: both
weak directions move both soles by the same vector in the base frame (mostly fore-aft x, some z),
which neither the IMU nor the L/R loop closure sees.  A known L/R foot fixture (jig) does not
help (scan: jig weak std == free weak std).

This script adds a few static external measurements of the sole positions in the base frame
(e.g. one camera + AprilTags on the pelvis and on a foot plate, or a pelvis-datum/foot-plate
calibration fixture), at sigma_ext per axis, at n_post static postures, to the DEPLOYED Phase-2
information of each seed (calib_active_<seed>.json, tau_int-inflated, misfit-scaled), and
reports the posterior std along the 4 baseline weak directions and the full posterior RMS std
(prior 2 deg included).

Usage: .venv/bin/python zero_external_ref.py   -> results/q0/external_ref.json
"""
import os, json
import numpy as np
from zero_motion_scan import Kin, baseline_cached, make_motion, trajectory, OUT, EPS
from zero_id import NL

PRIOR = np.deg2rad(2.0)
POSTURES = {  # static double-support postures (base relative to planted feet)
    "stand (-2 cm)": dict(sway_amp=0.0, squat=0.02),
    "squat (-6 cm)": dict(sway_amp=0.0, squat=0.06),
    "stand, base 2 cm fwd": dict(sway_amp=0.0, squat=0.02, base_dx=0.02),
}
AXES = {"x+z": [0, 2], "z only (height)": [2], "x only (fore-aft)": [0], "x+y+z": [0, 1, 2]}


def sole_jac(K, q12, h=1e-6):
    """d(sole positions in base frame, both feet: 6) / d(delta) (q_true = q_meas - delta)."""
    f0 = K.est._feet(q12)
    p0 = np.r_[f0[0][0], f0[1][0]]
    J = np.zeros((6, NL))
    for j in range(NL):
        dq = np.zeros(NL); dq[j] = h
        f1 = K.est._feet(q12 - dq)
        J[:, j] = (np.r_[f1[0][0], f1[1][0]] - p0) / h
    return J


def main():
    K = Kin()
    Vw = baseline_cached(K)[0]
    Jp = {}
    for name, kw in POSTURES.items():
        Q, _, w = trajectory(K, make_motion(**kw))
        Jp[name] = sole_jac(K, Q[0])
    # sensitivity: mm of sole displacement (base frame) per degree along each weak direction
    sens = {n: [(np.abs(J @ v).reshape(2, 3).mean(0) * np.deg2rad(1) * 1e3).round(2).tolist() for v in Vw.T]
            for n, J in Jp.items()}
    print("sole displacement per 1 deg along weak dir (mm, |x|,|y|,|z| mean of feet):")
    for n, s in sens.items():
        print("  ", n, s)
    out = dict(sensitivity_mm_per_deg=sens, rows=[])
    for seed in range(4):
        f = json.load(open(os.path.join(OUT, f"calib_active_{seed}.json")))["free"]
        V = np.array(f["eigvecs"]); sd = np.deg2rad(np.array(f["eig_std_deg"]))
        Hd = V @ np.diag(1 / sd ** 2) @ V.T
        for ax_name, ax in AXES.items():
            for n_post in (1, 2, 3):
                for sig_mm in (1.0, 2.0, 5.0):
                    H = Hd + np.eye(NL) / PRIOR ** 2
                    for name in list(POSTURES)[:n_post]:
                        rows = [i * 3 + a for i in range(2) for a in ax]
                        Jm = Jp[name][rows] / (sig_mm * 1e-3)
                        H = H + Jm.T @ Jm
                    C = np.linalg.inv(H)
                    ws = [float(np.rad2deg(np.sqrt(v @ C @ v))) for v in Vw.T]
                    out["rows"].append(dict(seed=seed, axes=ax_name, n_post=n_post, sigma_mm=sig_mm, weak_post_std_deg=ws,
                                            post_rms_std_deg=float(np.rad2deg(np.sqrt(np.trace(C) / NL)))))
        H0 = Hd + np.eye(NL) / PRIOR ** 2; C0 = np.linalg.inv(H0)
        out["rows"].append(dict(seed=seed, axes="none (Phase-2 sway only)", n_post=0, sigma_mm=None,
                                weak_post_std_deg=[float(np.rad2deg(np.sqrt(v @ C0 @ v))) for v in Vw.T],
                                post_rms_std_deg=float(np.rad2deg(np.sqrt(np.trace(C0) / NL)))))
    # summary: mean over seeds
    import itertools
    keys = sorted({(r["axes"], r["n_post"], r["sigma_mm"] or 0) for r in out["rows"]}, key=lambda k: (k[0], k[1], k[2]))
    print(f"{'axes':26s} n_post sigma_mm | posterior std along weak dirs (mean seeds 0-3, prior incl.) | post RMS std")
    summ = []
    for k in keys:
        rr = [r for r in out["rows"] if (r["axes"], r["n_post"], r["sigma_mm"] or 0) == k]
        ws = np.mean([r["weak_post_std_deg"] for r in rr], 0); pr = np.mean([r["post_rms_std_deg"] for r in rr])
        wmax = np.max([r["weak_post_std_deg"] for r in rr], 0)
        summ.append(dict(axes=k[0], n_post=k[1], sigma_mm=k[2], weak_mean=ws.tolist(), weak_max=wmax.tolist(), post_rms=pr))
        print(f"{k[0]:26s} {k[1]:6d} {k[2]:8.1f} | " + " ".join("%5.2f" % x for x in ws) + "  (max " +
              " ".join("%5.2f" % x for x in wmax) + f") | {pr:5.2f}")
    out["summary"] = summ
    json.dump(out, open(os.path.join(OUT, "external_ref.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
