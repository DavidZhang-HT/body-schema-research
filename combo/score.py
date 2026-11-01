"""Score v3: robust terminal progress, no low-height support-fraction moat."""
import numpy as np


def progress(r, foot_height):
    # No foot fraction and no penalty for a supporting hand, at any height.
    return (4*r['com_height'] + 6*r['base_height'] + 1.2*r['up_z']
            - 2*max(foot_height-r['com_height'], 0))


def score_v3(samples):
    """Samples (metrics, foot height) from final 0.2 s of a dynamic block.

    Taking the minimum suppresses brief height peaks. Penalize terminal-window
    KE, not action-wide peak KE: momentum remains available to cross gaps.
    Parent age limits in search prevent immortal transient snapshots.
    """
    values = [progress(r, h) for r, h in samples]
    ke = max(r['kinetic_j'] for r, _ in samples)
    speed = max(r['com_speed'] for r, _ in samples)
    return float(min(values) - .35*np.log1p(ke/.03) - .3*speed)


def sample(w):
    r = w.metrics()
    h = float(np.mean(w.d.xipos[[w.bodies[6], w.bodies[12]], 2]))
    return r, h


def evaluate(w, frames):
    end = w.snapshot()
    samples = []
    for snap in frames[-10::4]:
        w.restore(snap)
        samples.append(sample(w))
    w.restore(end)
    samples.append(sample(w))
    return score_v3(samples)


def exploration_priority(r):
    """B's instantaneous priority, only for explicitly exploratory beam seats.

    Absolute foot load is bounded at 0.6; no division by total contact load.
    Adding hand support cannot reduce this score at fixed foot load/geometry.
    This priority makes no claim of settled-state quality.
    """
    return float(4*r['com_height'] + 6*r['base_height'] + 1.2*r['up_z']
                 + .6*min(r['foot_load']/108, 1)
                 - .12*r['kinetic_j'] - .3*r['com_speed']
                 - 2*max(r.get('foot_height', 0)-r['com_height'], 0)
                 - max(0, .5-r['total_load']/108))
