"""Step 1 pre-check: is roll_probe_q_0.npy a CLEAN supine start?
Clean = mode_check(SUPINE) True, certify feasible, selfd >= -1mm.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
import numpy as np
from getup_schema import Schema, REGIONS
from getup_mpc import certify
from getup_modes import mode_check, classify

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..")

def main():
    S = Schema()
    q = np.load(os.path.join(SIM, "results", "getup_search", "roll_probe_q_0.npy"))
    c = certify(S, q, S.mu)
    k = S.kin(q, with_G=False, with_pairs=False)
    ok, why = mode_check(S, k, c, "SUPINE")
    print("mode_check SUPINE:", ok)
    for w in why:
        print("   -", w)
    print("classify:", classify(S, k, c))
    print("feasible:", c["feasible"], "rho:", round(float(c["rho"]), 4), "binding:", c["binding"])
    print("pen_mm:", round(1e3 * c["pen"], 2), "selfd_mm:", round(1e3 * c["selfd"], 2))
    d = S.region_height(k)
    print("touching:", [r for r, v in zip(REGIONS, d) if v <= 0.005])
    print("trunk R[2]:", np.round(k.R[2], 4))
    print("CLEAN:", ok and c["feasible"] and c["selfd"] >= -0.001)

if __name__ == "__main__":
    main()
