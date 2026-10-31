"""Quick dev runner for the F3 MPC (one phase at a time)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from getup_schema import Schema, REGIONS
from getup_attention import SchemaAttention
from getup_mpc import QSMPC, MPCParams, make_phases, run_phase, certify

S = Schema()
A = SchemaAttention(S)
p = MPCParams()
mpc = QSMPC(S, A, p)
q = np.load(os.path.join(os.path.dirname(__file__), "..", "results", "f3", "q_supine_settled.npy"))
phases = make_phases(S)
nph = int(sys.argv[1]) if len(sys.argv) > 1 else 1
logs = []
t0 = time.time()
for ph in phases[:nph]:
    q, out = run_phase(S, mpc, q, ph, p.mu, logs)
    L = logs[-1]
    print(ph.name, out, "ticks", L["tick"], "rho", L["cert"]["rho"], "time", time.time() - t0)
    sup = L.get("sup")
    if sup is not None:
        print("  a_cur", {REGIONS[i]: round(v, 3) for i, v in enumerate(sup["a_cur"]) if v > 0.01})
        print("  a_des", {REGIONS[i]: round(v, 3) for i, v in enumerate(sup["a_des"]) if v > 0.01})
    print("  loads", {k: round(v, 3) for k, v in L["cert"]["loads"].items() if v > 0.005})
    if "infeasible_diag" in L: print("  DIAG", L["infeasible_diag"])
    np.save(f"/tmp/q_{ph.name}.npy", q)
    if out != "DONE":
        break
if "elastic" in L:
    for e in L["elastic"]: print("  ELASTIC", e)
