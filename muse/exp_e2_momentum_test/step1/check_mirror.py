"""Verify mirror_q: mirrored start must be the exact sagittal mirror of the clean start.
Checks: certify identical (rho, feasible, selfd, pen), loads L/R-swapped, region heights
L/R-swapped, trunk R[2,1] sign flip, still classifies SUPINE, selfd >= -1mm.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from getup_schema import Schema, REGIONS
from getup_mpc import certify
from getup_modes import mode_check, classify
from model import JOINTS
from mirror import mirror_q

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..")

def swap_lr(d):
    out = {}
    for k, v in d.items():
        if k.startswith("L_"):
            out["R_" + k[2:]] = v
        elif k.startswith("R_"):
            out["L_" + k[2:]] = v
        else:
            out[k] = v
    return out

def main():
    S = Schema()
    q = np.load(os.path.join(SIM, "results", "getup_search", "roll_probe_q_0.npy"))
    qm = mirror_q(q, JOINTS, S.r.qadr)
    c0 = certify(S, q, S.mu); cm = certify(S, qm, S.mu)
    k0 = S.kin(q, with_G=False, with_pairs=False)
    km = S.kin(qm, with_G=False, with_pairs=False)
    d0 = dict(zip(REGIONS, S.region_height(k0)))
    dm = dict(zip(REGIONS, S.region_height(km)))
    print("rho:", round(float(c0["rho"]), 6), "->", round(float(cm["rho"]), 6))
    print("feasible:", c0["feasible"], "->", cm["feasible"])
    print("selfd_mm:", round(1e3*c0["selfd"], 3), "->", round(1e3*cm["selfd"], 3))
    print("pen_mm:", round(1e3*c0["pen"], 3), "->", round(1e3*cm["pen"], 3))
    print("loads orig:", {k: round(v,3) for k,v in c0["loads"].items() if v > 0.005})
    print("loads mirr:", {k: round(v,3) for k,v in cm["loads"].items() if v > 0.005})
    print("loads == swapped:", all(abs(cm["loads"].get(k,0) - v) < 1e-9
          for k, v in swap_lr({k: v for k,v in c0["loads"].items()}).items()))
    hswap = all(abs(dm[k] - v) < 1e-9 for k, v in swap_lr(d0).items())
    print("region heights L/R-swapped exactly:", hswap)
    if not hswap:
        for k, v in swap_lr(d0).items():
            if abs(dm[k]-v) >= 1e-9: print("   mismatch", k, dm[k], v)
    print("trunk R[2,1]:", round(float(k0.R[2,1]),6), "->", round(float(km.R[2,1]),6))
    print("base y:", round(float(q[1]),6), "->", round(float(qm[1]),6))
    ok, why = mode_check(S, km, cm, "SUPINE")
    print("mirror classifies SUPINE:", ok, why if not ok else "")
    print("classify mirror:", classify(S, km, cm))
    exact = (abs(float(c0["rho"])-float(cm["rho"])) < 1e-9 and c0["feasible"] == cm["feasible"]
             and abs(c0["selfd"]-cm["selfd"]) < 1e-9 and hswap and ok and cm["selfd"] >= -0.001)
    print("MIRROR EXACT:", exact)
    if exact:
        np.save(os.path.join(HERE, "q_supine_clean_mirrored.npy"), qm)

if __name__ == "__main__":
    main()
