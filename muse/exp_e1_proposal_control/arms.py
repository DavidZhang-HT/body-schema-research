"""E1/E5 arm proposal logic: matched-budget proposal generators for pi+ get-up search.

Each arm maps (world, state, candidates) -> list of (proposal_index, tag),
where tag records which sub-selector chose the action:
  'net'  - network-top-8 (attention screening)
  'heur' - FK-heuristic-top-8
  'rand' - uniform-random exploratory pick

The proposal *distribution* (64 candidates with the half-coordinated structure)
is identical across arms; only the *selection* differs. Budget (4000 physics
evals), beam width (14), depth (<=24), diversity cells, handover criterion and
balance() are defined by the harness (run_e1.py), not here.
"""
import copy
import numpy as np
import mujoco as mj
import torch

_B = None
_BOUNDS = None
_value = None


def configure(B, BOUNDS, value_fn):
    """Bind vendored constants: synergy matrix, action bounds, network value fn."""
    global _B, _BOUNDS, _value
    _B, _BOUNDS, _value = B, BOUNDS, value_fn


N_PROP = 64   # proposals generated per beam state per level
N_VAL = 12    # proposals validated with physics per beam state
N_TOP = 8     # taken by the arm's scoring rule
N_RAND = 4    # uniform-random exploratory picks


def generate_proposals(rng):
    """64 proposals ~ N(0,0.7)*BOUNDS clipped, incl. half-coordinated structure.

    Identical to vendor/ground_hybrid.py run(): first half has differential
    synergies zeroed and hip-spread/elbow damped; proposal 0 is the null action.
    """
    a = np.clip(rng.normal(0, .7, (N_PROP, 12)) * _BOUNDS, -_BOUNDS, _BOUNDS)
    a[:N_PROP // 2, 9:] = 0
    a[:N_PROP // 2, 3:5] *= .3
    a[:N_PROP // 2, 8] *= .3
    a[0] = 0
    return a


def _topk_plus_random(scores, rng, top_tag):
    scores = np.asarray(scores, dtype=float)
    ranking = np.argsort(scores)[::-1]
    chosen = list(ranking[:N_TOP])
    tags = [top_tag] * N_TOP
    unused = [i for i in range(N_PROP) if i not in chosen]
    extra = list(rng.choice(unused, N_RAND, replace=False))
    chosen += extra
    tags += ['rand'] * N_RAND
    return list(zip([int(i) for i in chosen], tags))


def select_network(state, candidates, net, rng, top_tag='net'):
    """Arm (a)/(d): score 64 proposals with the ensemble, top 8 + 4 random."""
    states = np.repeat(state[None], N_PROP, 0)
    pred, unc = net.predict(states, candidates)
    scores = np.asarray(_value(pred, unc), dtype=float)
    if not np.all(np.isfinite(scores)):
        # Degenerate network output (only plausible for the shuffled arm):
        # fall back to uniform random and flag it.
        return select_random(candidates, rng, fallback=True)
    return _topk_plus_random(scores, rng, top_tag)


def select_random(candidates, rng, fallback=False):
    """Arm (b): 12 uniform-random proposals from the same 64-proposal pool."""
    idx = rng.choice(N_PROP, N_VAL, replace=False)
    return [(int(i), 'rand') for i in idx]


def select_heuristic(world, candidates, rng):
    """Arm (c): FK-only proxy per proposal, no learning, no network.

    For each proposal a: set qpos[7:] = clip(q + B@a), mj_forward, then
        h = 4*com_z + 1.2*up_z - 2.0*foot_z
    where com_z is the base subtree_com z, up_z the base z-axis world-z
    component, foot_z the mean z of the two ankle_roll_link xipos.
    True state is restored via snapshot/restore after every probe.
    Select top 8 by h + 4 uniform random.
    """
    snap = world.snapshot()
    q = world.d.qpos[7:].copy()
    base = world.base
    ank_l, ank_r = world.bodies[6], world.bodies[12]
    hs = np.empty(N_PROP)
    for j in range(N_PROP):
        world.d.qpos[7:] = np.clip(q + _B @ candidates[j], world.qlo, world.qhi)
        mj.mj_forward(world.m, world.d)
        com_z = float(world.d.subtree_com[base, 2])
        up_z = float(world.d.xmat[base].reshape(3, 3)[2, 2])
        foot_z = float((world.d.xipos[ank_l, 2] + world.d.xipos[ank_r, 2]) / 2)
        hs[j] = 4 * com_z + 1.2 * up_z - 2.0 * foot_z
    world.restore(snap)
    return _topk_plus_random(hs, rng, top_tag='heur')


def build_shuffled_ensemble(ens, seed=777):
    """Arm (d): deep copy of the ensemble with every parameter tensor of every
    member independently permuted (fixed RNG seed). Registered buffers
    (attention mask, body properties, synergy basis) and the input/output
    normalization are kept as-is, so the predict() API and shapes are unchanged
    but all learned structure is destroyed."""
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(ens)
    for net in out.models:
        for name, p in net.named_parameters():
            flat = p.detach().cpu().numpy().reshape(-1)
            perm = rng.permutation(flat.shape[0])
            p.data = torch.from_numpy(flat[perm].reshape(p.shape)).to(p.device)
    return out


def handover_ok(r):
    """Standing-handover gate (from vendor/ground_hybrid.py)."""
    return (r['base_height'] > .16 and r['up_z'] > .9
            and r['foot_load'] > 75 and r['nonfoot_load'] < 8)
