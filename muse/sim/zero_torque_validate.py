"""Validation of the joint-torque channel (Phase 2b) on the recorded Phase-2 calibrations.

Re-solves each recorded calibration (zero_torque_record.py -> samples_tau_<mode>_<seed>.npz; the
gated rows are exactly the samples the deployed estimator accepted in that run) with
ZeroEstimator(torque=True) and writes results/q0/calib_tq<variant>_<mode>_<seed>.json in the same
format as zero_experiment.analyse (free stance + jig).  The loop-closure-only reference is the
published calib_<mode>_<seed>.json (bit-identical runs).

Variants (robustness of the torque model, NOT used for the headline numbers):
  ''        : as recorded
  'gain5'   : all measured leg torques x 1.05 (systematic torque-constant error)
  'gainrnd' : per-joint torque-constant error ~ U(-5 %, +5 %) (fixed RNG)
  'v0fix'   : friction transition velocity fixed at 0.02 rad/s (only mu co-identified) - ablation

Usage: .venv/bin/python zero_torque_validate.py <variant|base> <mode> <seed> [<seed> ...]
"""
import os, sys, json, time
import numpy as np
from scenarios import true_params_q0, Q0_PRIOR_STD_DEG
from zero_id import ZeroEstimator, NL
from zero_experiment import analyse, summary_line, OUT


def run(seed, mode, variant=""):
    rec = dict(np.load(os.path.join(OUT, f"samples_tau_{mode}_{seed}.npz")))
    z = np.load(os.path.join(OUT, f"samples_{mode}_{seed}.npz"))
    if variant == "gain5":
        rec["tau"] = rec["tau"].copy(); rec["tau"][:, :NL] *= 1.05
    elif variant == "gainrnd":
        g = 1 + np.random.default_rng(77 + seed).uniform(-0.05, 0.05, NL)
        rec["tau"] = rec["tau"].copy(); rec["tau"][:, :NL] *= g[None, :]
    zest = ZeroEstimator(prior_std=np.deg2rad(Q0_PRIOR_STD_DEG), torque=True, co_v0=(variant != "v0fix"))
    zest.load_recording(rec)
    assert len(zest.Q) == len(z["Q"]) and np.allclose(np.array(zest.Q), z["Q"]), "recording != published samples"
    zest.n_seen, zest.n_gate_rej = int(z["n_seen"]), int(z["n_gate_rej"])
    t0 = time.time()
    rep = analyse(zest, true_params_q0(seed), seed, mode, list(z["tilt"]), list(z["bl"]),
                  float(z["slip"]), bool(z["fell"]))
    rep["wall_s"] = time.time() - t0
    tp = true_params_q0(seed)
    rep["true_trunk"] = [tp.trunk_dm, tp.trunk_dx, tp.trunk_dy, tp.trunk_dz]
    rep["true_frictionloss"] = tp.frictionloss
    rep["variant"] = variant
    return rep


if __name__ == "__main__":
    variant = "" if sys.argv[1] == "base" else sys.argv[1]
    mode = sys.argv[2]
    for seed in [int(s) for s in sys.argv[3:]]:
        rep = run(seed, mode, variant)
        json.dump(rep, open(os.path.join(OUT, f"calib_tq{variant}_{mode}_{seed}.json"), "w"), indent=1)
        f = rep["free"]
        print(summary_line(rep, "free"), flush=True)
        print(summary_line(rep, "jig"), flush=True)
        print("   n_torque_rej", f.get("n_torque_rej"), "agree", f.get("channel_agree"), "zmax", round(f.get("agree_zmax", 0), 2),
              "evid_ok", f.get("torque_evidence_ok"), "fallback", f.get("fallback"), "reason", f.get("reason"),
              "| trunk_hat", f.get("trunk_hat"), "true", np.round(rep["true_trunk"], 4).tolist(),
              "| mu_hat", round(f.get("mu_hat", np.nan), 3), "v0_hat", round(f.get("v0_hat", np.nan), 4), "+-", round(f.get("mu_std", np.nan), 3), "true", round(rep["true_frictionloss"], 3),
              "| s2", round(f.get("misfit_scale", np.nan), 2), round(f.get("misfit_scale_torque", np.nan), 2),
              "wall", round(rep["wall_s"], 1), flush=True)
