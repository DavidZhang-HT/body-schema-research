"""A-style contact predicates adapted to B's mesh regions, without static gates.

Dynamic membership and static certification are deliberately separate. SIDE_R
means right side down (local +y points up). B merges pelvis/back/chest in trunk;
SIT is therefore an upright low trunk/thigh-supported approximation.
"""
import numpy as np
from .plant import gs

MODES = ('SUPINE', 'SIDE_L', 'SIDE_R', 'PRONE', 'SIT', 'ALLFOURS',
         'HAND_SUPPORT', 'CROUCH', 'STAND', 'TRANSITION')
GRAPH = {a: tuple(b for b in MODES if a != b) for a in MODES}


def prior(src, dst):
    # P-F3 is a weak preference, never a removed edge. Descent is free.
    return .05 if src == 'SUPINE' and dst in ('SIT', 'CROUCH', 'STAND') else 0.


def classify(w, r=None):
    r = r or w.metrics()
    rot = w.d.xmat[w.base].reshape(3, 3)
    touching = set()
    floor = w.m.geom('ground').id
    for c in w.d.contact:
        if floor in (c.geom1, c.geom2) and c.dist <= .005:
            g = c.geom2 if c.geom1 == floor else c.geom1
            touching.add(w.m.body(int(w.m.geom_bodyid[g])).name)
    feet = {s + '_ankle_roll_link' for s in ('l', 'r')}
    arms = any(any(k in b for k in ('wrist', 'elbow', 'shoulder')) for b in touching)
    legs = any(any(k in b for k in ('calf', 'ankle')) for b in touching)
    trunk = 'base_link' in touching
    if feet <= touching and r['nonfoot_load'] < 5 and r['base_height'] > .30 and rot[2, 2] > .95:
        return 'STAND'
    if trunk and rot[2, 0] > .7:
        return 'SUPINE'
    if (trunk or arms) and abs(rot[2, 1]) > .7 and r['base_height'] < .25:
        return 'SIDE_R' if rot[2, 1] > 0 else 'SIDE_L'
    if trunk and rot[2, 0] < -.7:
        return 'PRONE'
    if arms and legs and not trunk:
        return 'ALLFOURS' if rot[2, 0] < -.5 else 'HAND_SUPPORT'
    if rot[2, 2] > .5 and (trunk or any('hip' in b or 'thigh' in b for b in touching)):
        return 'SIT'
    if feet <= touching and r['nonfoot_load'] < 8 and rot[2, 2] > .5:
        return 'CROUCH'
    return 'TRANSITION'


def handover(r):
    return r['base_height'] > .16 and r['up_z'] > .9 and r['foot_load'] > 75 and r['nonfoot_load'] < 8
