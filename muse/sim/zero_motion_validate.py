"""Validation of the scan prediction in the full simulation (deployed estimator, free stance).

For each calib_<mode>_<seed>.json (zero_experiment.py) report, along the 4 baseline weak directions
(zero_motion_scan.baseline_cached: weakest eigvecs of the seed-0 Phase-2 calibration):
  deployed formal (data-only, tau_int-inflated, misfit-scaled) marginal std, posterior error of the
  committed estimate along the direction, plus totals.  Also the trunk-yaw / hip-yaw excursion
  actually realised (cached samples) and the number of gated samples.

Usage: .venv/bin/python zero_motion_validate.py [modes...]  -> results/q0/motion_validate.json
"""
import os, sys, json
import numpy as np
from zero_motion_scan import Kin, baseline_cached, EPS, OUT
from zero_id import NL

MODES = sys.argv[1:] or ["active", "yaw15", "yaw15sq"]


def main():
    K = Kin()
    Vw = baseline_cached(K)[0]
    rows = []
    for mode in MODES:
        for seed in range(5):
            fn = os.path.join(OUT, f"calib_{mode}_{seed}.json")
            if not os.path.exists(fn):
                continue
            rep = json.load(open(fn)); f = rep["free"]
            z = np.load(os.path.join(OUT, f"samples_{mode}_{seed}.npz"))
            thigh = np.rad2deg(z["Q"][:, [2, 8]]) if len(z["Q"]) else np.zeros((0, 2))
            row = dict(mode=mode, seed=seed, fell=rep["fell"], n=f.get("n_used", f["n"]),
                       committed=f["committed"], n_obs=f.get("n_obs_dirs"),
                       thigh_range_deg=(thigh.max(0) - thigh.min(0)).round(1).tolist() if len(thigh) else None,
                       err_rms=f.get("err_rms_deg"), prior_rms=f.get("prior_err_rms_deg"),
                       err_obs=f.get("err_rms_obs_subspace_deg"), err_null=f.get("err_rms_null_subspace_deg"))
            if "eigvecs" in f:
                V = np.array(f["eigvecs"]); sd = np.deg2rad(np.array(f["eig_std_deg"]))
                Hi = np.linalg.inv(V @ np.diag(1 / sd ** 2) @ V.T + EPS * np.eye(NL))
                true = np.array(rep["true_deg"]); dc = np.array(f["delta_commit_deg"])
                row["weak_std_deg"] = [float(np.rad2deg(np.sqrt(v @ Hi @ v))) for v in Vw.T]
                row["weak_err_deg"] = [float(v @ (dc - true)) for v in Vw.T]
                row["weak_true_deg"] = [float(v @ true) for v in Vw.T]
            rows.append(row)
            print(f"{mode:8s} s{seed} fell={row['fell']!s:5s} n={row['n']} commit={row['committed']} obs={row['n_obs']} "
                  f"thigh-range={row['thigh_range_deg']} | weak std " +
                  " ".join("%6.2f" % x for x in row.get("weak_std_deg", [])) + " | weak err " +
                  " ".join("%+5.2f" % x for x in row.get("weak_err_deg", [])) + " (true " +
                  " ".join("%+5.2f" % x for x in row.get("weak_true_deg", [])) + ")"
                  f" | RMS {row['err_rms'] or float('nan'):.2f} (prior {row['prior_rms'] or float('nan'):.2f}) "
                  f"obs {row['err_obs'] or float('nan'):.2f} null {row['err_null'] or float('nan'):.2f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "motion_validate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
