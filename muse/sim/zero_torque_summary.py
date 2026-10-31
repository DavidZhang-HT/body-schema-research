"""Summary tables for the Phase-2b torque channel (reads calib_*.json; no simulation).

Per seed / mode: per-joint commit error (deg) for loop-closure-only (published calib_<mode>_<seed>),
torque channel (calib_tq_<mode>_<seed>) and variants; error on the LC-only committed subspace
(regression check), error along W1 (weakest LC direction), friction / trunk co-identification.
Usage: .venv/bin/python zero_torque_summary.py   -> results/q0/torque_summary.json + markdown on stdout
"""
import os, json
import numpy as np
from zero_experiment import OUT, NAMES


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def row(rep, lc):
    """metrics of rep (free stance) w.r.t. truth; subspaces from the LC-only run lc."""
    f = rep["free"]; t = np.array(rep["true_deg"])
    e = np.array(f["err_commit_deg"])
    out = dict(committed=f.get("committed"), rms=float(np.sqrt((e ** 2).mean())), err=e.round(2).tolist(),
               fallback=f.get("fallback"), reason=f.get("reason"))
    if lc is not None and "eigvecs" in lc["free"]:
        V = np.array(lc["free"]["eigvecs"]); ol = np.array(lc["free"]["obs_mask"], bool)
        out["err_lc_obs_subspace"] = float(np.linalg.norm(V[:, ol].T @ e) / np.sqrt(12))
        out["err_W1"] = float(V[:, 0] @ e); out["true_W1"] = float(V[:, 0] @ t)
        out["err_W2"] = float(V[:, 1] @ e); out["true_W2"] = float(V[:, 1] @ t)
        out["err_lc_null_subspace"] = float(np.linalg.norm(V[:, ~ol].T @ e) / np.sqrt(12))
    for k in ("mu_hat", "v0_hat", "mu_std", "trunk_hat", "agree_zmax", "channel_agree", "n_obs_dirs", "misfit_scale_torque"):
        if k in f:
            out[k] = f[k]
    if "eig_std_deg" in f and f["eig_std_deg"]:
        out["weakest_std"] = float(max(f.get("torque_eig_std_deg", f["eig_std_deg"])))
    out["true_trunk"] = rep.get("true_trunk"); out["true_mu"] = rep.get("true_frictionloss")
    return out


if __name__ == "__main__":
    S = {}
    for mode in ("active", "quiet"):
        for seed in range(5):
            lc = load(f"calib_{mode}_{seed}.json")
            S[f"{mode}_{seed}"] = {"lc": row(lc, lc)}
            for v in ("", "v0fix", "gain5", "gainrnd"):
                r = load(f"calib_tq{v}_{mode}_{seed}.json")
                if r is not None:
                    S[f"{mode}_{seed}"]["tq" + v] = row(r, lc)
    json.dump(S, open(os.path.join(OUT, "torque_summary.json"), "w"), indent=1)

    print("| Seed | Mode | Estimator | Committed | Total RMS err | err on LC-committed subspace | W1 err (true) | W2 err (true) | mu_hat (true) | v0_hat | trunk dm / dx / dz (true) | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, d in S.items():
        mode, seed = key.split("_")
        for est in ("lc", "tq", "tqv0fix", "tqgain5", "tqgainrnd"):
            if est not in d:
                continue
            r = d[est]
            tr = r.get("trunk_hat"); tt = r.get("true_trunk")
            trs = f"{tr[0]:.2f} / {100*tr[1]:.1f} / {100*tr[3]:.1f} ({tt[0]:.2f} / {100*tt[1]:.1f} / {100*tt[3]:.1f})" if tr and tt else "—"
            mus = f"{r['mu_hat']:.3f} ({r['true_mu']:.3f})" if "mu_hat" in r else "—"
            note = r.get("fallback") or r.get("reason") or ""
            print(f"| {seed} | {mode} | {est} | {r['committed']} | {r['rms']:.2f} | {r.get('err_lc_obs_subspace', np.nan):.2f} | "
                  f"{r.get('err_W1', np.nan):+.2f} ({r.get('true_W1', np.nan):+.2f}) | {r.get('err_W2', np.nan):+.2f} ({r.get('true_W2', np.nan):+.2f}) | "
                  f"{mus} | {r.get('v0_hat', float('nan')):.3f} | {trs} | {note} |")
    print()
    print("Per-joint commit error (deg), free stance:")
    print("| Seed | Mode | Estimator | " + " | ".join(NAMES) + " | RMS |")
    print("|---|---|---|" + "---|" * (len(NAMES) + 1))
    for key, d in S.items():
        mode, seed = key.split("_")
        for est in ("lc", "tq"):
            if est in d:
                r = d[est]
                print(f"| {seed} | {mode} | {est} | " + " | ".join(f"{x:+.2f}" for x in r["err"]) + f" | {r['rms']:.2f} |")
