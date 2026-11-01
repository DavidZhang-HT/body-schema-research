"""Official B plant; reuse E3's certifier without its experiment-module stub."""
import os
import sys
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codex'))
sys.path.insert(0, str(ROOT / 'muse/exp_e3_statics_gap'))
import numpy as np
import ground_schema as gs
from ground_balance import balance
from e3_cert import BSchema

# E3 augmented basis, same joint-index convention, installed once per process.
B = np.zeros((24, 16))
B[:, :12] = gs.B[:, :12]
B[[1, 7], 12] = 1
B[[5, 11], 13] = 1
B[5, 14], B[11, 14] = 1, -1
B[3, 15], B[9, 15] = 1, -1
BOUNDS = np.r_[gs.BOUNDS[:12], .30, .30, .30, .50]
gs.B, gs.BOUNDS = B, BOUNDS
GroundWorld = gs.GroundWorld
