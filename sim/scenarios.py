"""Ground-truth body-schema scenarios (what the URDF does not know).

The 'base' scenario is a plausible pi+ build deviation:
  +0.8 kg on the trunk (battery / compute / cabling not in the URDF, ~7 % of 11 kg),
  trunk CoM shifted +2.5 cm forward, -1.2 cm lateral, +3 cm up,
  12 ms actuation latency (CAN bus + driver loop),
  3x the MJCF joint friction on the legs (0.6 Nm stiction/Coulomb),
  softer foot pads (contact time constant 35 ms vs 20 ms),
  mu = 0.55 floor (MJCF/URDF default assumes 1.0),
  0.8 deg ankle backlash.
Per-seed scenarios are random draws around it.
"""
import numpy as np
from model import BodyParams

IDENT_KEYS = ["trunk_dm", "trunk_dx", "trunk_dy", "trunk_dz", "frictionloss", "delay"]   # identified
PRIOR_KEYS = ["mu", "contact_tc", "backlash"]                                            # kept as prior

PRIOR_MEAN = BodyParams()          # = URDF / MJCF nominal
PRIOR_STD = dict(trunk_dm=0.6, trunk_dx=0.02, trunk_dy=0.015, trunk_dz=0.03,
                 frictionloss=0.3, delay=0.008, mu=0.25, contact_tc=0.01, backlash=0.01)


def true_params(seed=0):
    base = BodyParams(trunk_dm=0.8, trunk_dx=0.025, trunk_dy=-0.012, trunk_dz=0.03, delay=0.012,
                      frictionloss=0.6, contact_tc=0.035, mu=0.55, backlash=0.014)
    if seed == 0:
        return base
    rng = np.random.default_rng(1000 + seed)
    return BodyParams(
        trunk_dm=float(np.clip(base.trunk_dm + rng.normal(0, 0.3), 0.2, 1.5)),
        trunk_dx=float(base.trunk_dx * rng.choice([-1, 1]) + rng.normal(0, 0.006)),
        trunk_dy=float(rng.normal(0, 0.012)),
        trunk_dz=float(base.trunk_dz + rng.normal(0, 0.01)),
        delay=float(np.clip(base.delay + rng.normal(0, 0.003), 0.004, 0.02)),
        frictionloss=float(np.clip(base.frictionloss + rng.normal(0, 0.15), 0.3, 1.0)),
        contact_tc=float(np.clip(base.contact_tc + rng.normal(0, 0.005), 0.02, 0.05)),
        mu=float(np.clip(base.mu + rng.normal(0, 0.08), 0.4, 0.8)),
        backlash=base.backlash,
    )


# ----------------------------------------------------------------------------- joint zero offsets
# Encoder / assembly zero-position bias of the 12 leg joints (arms kept at 0: they are not
# part of the double-support kinematic loop, so loop closure cannot see them anyway).
# Plausible magnitudes 1-3 deg (absolute-encoder mounting / gear-tooth indexing error).
# Order: l_{hip_pitch, hip_roll, thigh(yaw), calf(knee), ankle_pitch, ankle_roll}, r_{...}
Q0_BASE_DEG = [+2.0, -1.5, +1.0, -2.5, +1.5, +1.0,
               -1.0, +2.0, -1.5, +3.0, -2.0, -1.2]
Q0_PRIOR_STD_DEG = 2.0      # prior on each leg offset: N(0, 2 deg^2)


def true_zero_offsets(seed=0):
    """20-vector [rad] in JOINTS order. Separate RNG stream -> true_params() is unchanged."""
    if seed == 0:
        leg = np.array(Q0_BASE_DEG)
    else:
        rng = np.random.default_rng(3000 + seed)
        leg = rng.uniform(1.0, 3.0, 12) * rng.choice([-1, 1], 12)
    return [float(x) for x in np.deg2rad(np.concatenate([leg, np.zeros(8)]))]


def true_params_q0(seed=0):
    """'q0' scenario = base scenario of this seed + leg joint zero offsets."""
    p = true_params(seed)
    p.q0 = true_zero_offsets(seed)
    return p
