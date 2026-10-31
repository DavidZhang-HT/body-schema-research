"""Re-run the Phase-2 calibrations with full-rate recording (torque-channel scan input).
Bit-identical to the published calib runs (recording consumes no RNG); checks that.
Usage: .venv/bin/python zero_torque_record.py [seed ...]   -> results/q0/samples_tau_<mode>_<seed>.npz"""
import sys, os, json
import numpy as np
from zero_experiment import calibrate, OUT

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(5))
    for seed in seeds:
        for mode in ("active", "quiet"):
            rep = calibrate(seed, mode, record=True)
            old = json.load(open(os.path.join(OUT, f"calib_{mode}_{seed}.json")))
            same = np.allclose(rep["free"]["delta_map_deg"], old["free"]["delta_map_deg"]) if "delta_map_deg" in old["free"] else (rep["free"]["n"] == old["free"]["n"])
            print(seed, mode, "n", rep["free"]["n"], "identical_to_published", same, flush=True)
