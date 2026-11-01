"""E5 mode predicates on the 24-DoF mesh plant (pi_plus_24dof_250826).

A's mode predicates were written for the 20-DoF primitive plant. For a FAIR
comparison, E5 runs ALL arms on the same 24-DoF mesh plant, so the modes are
re-implemented here as lightweight approximations from base orientation +
node contact weights. They are documented approximations, not the original
A predicates.

Grounding (measured 2026-11-01 on settled passive_wide states, this plant):
  - supine settle: up_z ~= +0.10, local-x world-z ~= +0.995 (chest faces UP),
    base_height ~= 0.081, base node carries ~56/108 N
  - prone settle:  up_z ~= +0.07, local-x world-z ~= -0.994 (chest faces DOWN),
    base_height ~= 0.081, head_pitch node carries ~10 N (face-down head contact)
  - local-z (up_z) is the SPINE axis here, so supine/prone are NOT separable
    by up_z: both have up_z ~= 0 (spine horizontal). The separating axis is
    the base local x-axis (chest direction): chest-up => SUPINE, chest-down
    => PRONE. Side-lying => local-y world-z near +/-1.

Predicates (world = GroundWorld, restored to the state being classified):
  SUPINE: lx_z > +0.6 and base_height < 0.25        (chest faces up, low)
  PRONE : lx_z < -0.6 and base_height < 0.25        (chest faces down, low)
  SIDE  : |ly_z| > 0.7 and base_height < 0.25       (flank down, low)
  STAND : base_height > 0.30 and up_z > 0.95 and foot_frac > 0.80
          (foot_frac = foot_load / total_load; matches the task spec)
  else  : TRANSITION (unclassified)

Mode DAG (descent allowed):
  SUPINE -> SIDE -> STAND
  SUPINE -> PRONE -> STAND
STAND -> STAND is the refinement edge (feasibility = E1 handover criterion).
Descent allowed = the controller re-classifies the achieved state after every
edge and continues from whatever DAG node it actually landed in (e.g. a
SIDE->STAND attempt that regresses into PRONE continues as PRONE->STAND).
"""
import numpy as np

SUPINE = 'SUPINE'
PRONE = 'PRONE'
SIDE = 'SIDE'
STAND = 'STAND'
TRANSITION = 'TRANSITION'

# DAG successors (STAND->STAND is handled as the refinement edge)
SUCCESSORS = {
    SUPINE: [SIDE, PRONE],
    SIDE: [STAND],
    PRONE: [STAND],
}
# Directed-edge indices for the matched candidate RNG namespace
EDGE_INDEX = {
    (SUPINE, SIDE): 0,
    (SUPINE, PRONE): 1,
    (SIDE, STAND): 2,
    (PRONE, STAND): 3,
    (STAND, STAND): 4,
}


def orientation(world):
    """Base-frame orientation signals + contact load fractions."""
    rot = world.d.xmat[world.base].reshape(3, 3)
    r = world.metrics()
    total = max(r['total_load'], 1e-9)
    return {
        'up_z': float(rot[2, 2]),
        'lx_z': float(rot[2, 0]),   # local x (chest dir) world-z
        'ly_z': float(rot[2, 1]),   # local y (left dir) world-z
        'base_height': float(r['base_height']),
        'foot_frac': float(r['foot_load'] / total),
        'total_load': float(r['total_load']),
        'foot_load': float(r['foot_load']),
        'nonfoot_load': float(r['nonfoot_load']),
    }


def classify(world):
    """Return (mode, signals). Lightweight approximation; see module doc."""
    s = orientation(world)
    bh = s['base_height']
    if s['up_z'] > 0.95 and bh > 0.30 and s['foot_frac'] > 0.80 and s['total_load'] > 1.0:
        return STAND, s
    if bh < 0.25 and s['total_load'] > 1.0:
        if abs(s['ly_z']) > 0.7:
            return SIDE, s
        if s['lx_z'] > 0.6:
            return SUPINE, s
        if s['lx_z'] < -0.6:
            return PRONE, s
    return TRANSITION, s


def mode_term(dst, s):
    """Edge-guidance bonus in [-1, 1.5]-ish range for destination mode dst."""
    if dst == PRONE:
        return -s['lx_z']
    if dst == SIDE:
        return abs(s['ly_z'])
    if dst == STAND:
        return (0.5 * s['up_z']
                + 0.5 * min(s['base_height'] / 0.45, 1.0)
                + 0.5 * s['foot_frac'])
    return 0.0
