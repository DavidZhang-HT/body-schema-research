"""E3 plant shim: loads scheme B's (codex/) GroundWorld + score() UNMODIFIED.

codex/ground_schema.py imports `experiment` (needs osqp, imageio etc.) and `attention_data`
(imports zero_identification -> experiment).  Only four symbols are actually used by
ground_schema / ground_search: experiment.{model, standing, ROOT} and
attention_data.motor_envelope.  We register light stub modules that contain VERBATIM copies
of those functions (copied from codex/experiment.py L12-L33 and codex/attention_data.py
L14-L18 @ main) so that ground_schema.py and ground_search.py run byte-for-byte unchanged.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
CODEX = HERE / 'codex'

# ---- stub `experiment` (verbatim copies) -------------------------------------------------
_exp = types.ModuleType('experiment')
_exp.ROOT = CODEX
_exp.DT = .004
_exp.FOOT = np.array([-.025, 0, -.05])


def _model():
    return mj.MjModel.from_xml_path(str(CODEX / 'assets/scene.xml'))


def _kinematics(m, d):
    out = []
    for side in ['l', 'r']:
        b = m.body(side + '_ankle_roll_link').id
        rot = d.xmat[b].reshape(3, 3); pos = d.xpos[b] + rot @ _exp.FOOT
        jp = np.zeros((3, m.nv)); jr = jp.copy(); mj.mj_jac(m, d, jp, jr, pos, b)
        out.append((pos, rot, np.vstack([jp, jr])))
    return out


def _standing(m):
    d = mj.MjData(m); d.qpos[3] = 1
    for side in ['l', 'r']:
        for j, v in [('hip_pitch', -.25), ('calf', .5), ('ankle_pitch', -.25), ('shoulder_roll', .25 if side == 'l' else -.25)]:
            d.qpos[m.joint(side + '_' + j + '_joint').qposadr[0]] = v
    mj.mj_forward(m, d); feet = _kinematics(m, d)
    d.qpos[2] -= min(f[0][2] for f in feet) - .0005; mj.mj_forward(m, d)
    return d.qpos.copy()


_exp.model = _model
_exp.kinematics = _kinematics
_exp.standing = _standing
sys.modules['experiment'] = _exp

# ---- stub `attention_data` (verbatim motor envelope) -------------------------------------
_ad = types.ModuleType('attention_data')
_ad.VELOCITY_LIMIT = np.r_[np.full(12, 5.45), np.full(10, 15.91), np.full(2, 30.37)]


def motor_envelope(torque, velocity):
    drive = torque * velocity > 0
    return torque * np.where(drive, np.clip(1 - np.abs(velocity) / _ad.VELOCITY_LIMIT, 0, 1), 1)


_ad.motor_envelope = motor_envelope
sys.modules['attention_data'] = _ad

sys.path.insert(0, str(CODEX))
import ground_schema as gs  # noqa: E402
from ground_search import score  # noqa: E402

GroundWorld = gs.GroundWorld
B = gs.B
BOUNDS = gs.BOUNDS
SCHEMAS = gs.SCHEMAS


def load_plateau():
    z = np.load(CODEX / 'ground_results/hybrid_supine_101_best.npz')
    return z['initial'], z['actions'], z['final']
