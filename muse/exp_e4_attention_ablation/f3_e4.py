#!/usr/bin/env python
"""E4 F3 scenario: supine->standing keyframe search (getup_f3_run pipeline) with
ablated attention prices.

HARNESS-LOCAL only (sim/ untouched): imports getup_f3_run as a module and
monkeypatches its module-level `SchemaAttention` reference to an arm factory,
plus `ROOT` to this workdir.  getup_f3_run.main(tag) then builds
QSMPC(S, <arm attention>, p) itself.  The q start (results/f3/q_supine_settled.npy),
phases, and stall policy are the committed F3 ones; baseline is run_a
(verdict NOT_FOUND, stalls at P1_head_up after 122 ticks, wall ~46 s).

Usage: python f3_e4.py <arm>        arm in {attn, uniform, permuted}
Writes: run_f3_e4_<arm>.npz / .json in this workdir (ROOT redirected; results/f3/ untouched).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = "/home/hatch/workspace/research/body-schema-gait/muse/sim"
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)

from run_e4 import UniformAttention, PermutedAttention, PERM_SEED  # noqa: E402
from getup_attention import SchemaAttention, AttnParams  # noqa: E402
import getup_f3_run as f3mod  # noqa: E402

ARMS = {
    "attn": lambda S: SchemaAttention(S, AttnParams()),
    "uniform": lambda S: UniformAttention(S),
    "permuted": lambda S: PermutedAttention(S, seed=PERM_SEED),
}


def main(arm):
    assert arm in ARMS, f"arm must be one of {sorted(ARMS)}"
    f3mod.SchemaAttention = ARMS[arm]     # factory called as SchemaAttention(S) in main()
    f3mod.ROOT = HERE                    # outputs -> this workdir
    print(f"[E4-F3] arm={arm} seed={PERM_SEED if arm == 'permuted' else '-'}", flush=True)
    f3mod.main(f"f3_e4_{arm}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "attn")
