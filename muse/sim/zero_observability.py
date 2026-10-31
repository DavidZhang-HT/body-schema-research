"""Noise-free observability analysis of joint zero offsets from double-support loop closure.

Takes the TRUE joint trajectories of a calibration run (cached measured q minus the true
offsets; backlash/noise excluded) and the true gravity direction, and computes the singular
values / directions of the marginal (nuisance x, y, psi Schur-complemented) loop-closure
Jacobian w.r.t. the 12 leg offsets.  No prior, no noise model tuning: this is the pure
geometry of the problem.  sigma: gravity 3e-3 rad, rel. rotation 1e-3 rad, rel. position 1 mm
(only used to make channels commensurable).

Usage: .venv/bin/python zero_observability.py   -> results/q0/observability.json
"""
import os, json
import numpy as np
from scenarios import true_params_q0
from zero_id import ZeroEstimator, NL, NP, label_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "q0")


def analyse(seed, mode, z):
    tp = true_params_q0(seed)
    Q = z["Q"] - np.array(tp.q0[:12])[None, :]           # ~ true q (+2e-4 noise, +backlash)
    Gb = z["Gb"]
    est = ZeroEstimator(use_tau=False)
    th = np.zeros(NP); th[NL:] = est.nominal_stance()
    _, J = est.residuals(th, Q, Gb)
    Jw = (J / est.sig[None, :, None]).reshape(-1, NP)
    H = Jw.T @ Jw
    Hm = H[:NL, :NL] - H[:NL, NL:] @ np.linalg.solve(H[NL:, NL:], H[NL:, :NL])
    lam, V = np.linalg.eigh(Hm)
    lam = np.maximum(lam, 0)
    # equivalent 1-sample-per-second std (deg) -> geometry strength independent of sample count
    n_sec = len(Q) / 100.0
    std = np.rad2deg(1 / np.sqrt(np.maximum(lam / n_sec, 1e-30)))
    return dict(seed=seed, mode=mode, n=len(Q),
                rel_sv=(np.sqrt(lam / lam.max())).tolist(),
                std_deg_per_1s=std.tolist(),
                dirs=[label_dir(V[:, i]) for i in range(NL)])


if __name__ == "__main__":
    out = []
    for seed in range(4):
        for mode in ("active", "quiet"):
            z = np.load(os.path.join(OUT, f"samples_{mode}_{seed}.npz"))
            a = analyse(seed, mode, z)
            out.append(a)
            print(f"seed {seed} {mode:6s} rel.singular values: " + " ".join("%.1e" % s for s in a["rel_sv"]))
            if seed == 0:
                for s, sd, d in zip(a["rel_sv"], a["std_deg_per_1s"], a["dirs"]):
                    print(f"    sv {s:.1e}  std(1 s of data) {sd:9.3f} deg   {d}")
    json.dump(out, open(os.path.join(OUT, "observability.json"), "w"), indent=1)
