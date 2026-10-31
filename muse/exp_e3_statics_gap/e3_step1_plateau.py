"""E3 step 1: certify scheme B's supine plateau (hybrid_supine_101) with A's statics certificate.

States certified:
  * sanity: B's nominal standing pose (feet-only expected), B's passive supine seed-101 state
  * every block-end state of the replayed best path (10 blocks; replay is deterministic)
  * the plateau state itself (= saved 'final' snapshot, beam best from level 14 to 24)
  * the plateau held for 5 more zero-action blocks (B's own action a=0 = hold PD target), to get
    a near-rest version of the plateau (the saved snapshot still has KE 0.126 J).
For each: all-contacts certificate and feet-only certificate, at mu 0.6 (A) and 0.8 (B floor).
"""
import json
import numpy as np
from e3_plant import GroundWorld, load_plateau, score, CODEX
from e3_cert import BSchema, MU_PLAN
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'results'; OUT.mkdir(exist_ok=True)


def both(S, q):
    r = {}
    for mu in (MU_PLAN, 0.8):
        r[f'all_mu{mu}'] = S.certify(q, mu=mu)
        r[f'feet_mu{mu}'] = S.certify(q, mu=mu, feet_only=True)
    return r


def brief(c):
    return f"ok={c['ok']} feas={c['feasible']} rho={c['rho']:.3f} bind={c['binding']} loaded={c['loaded']} pen={1e3*c['pen']:.1f}mm selfd={1e3*c['selfd']:.1f}mm"


def main():
    w = GroundWorld(); S = BSchema(w.m)
    print('shell points', S.P, 'self pairs', len(S.pairs), 'mass', S.M)
    out = {'shell_points': S.P, 'self_pairs': len(S.pairs), 'mass': S.M, 'states': []}

    def rec(name, extra=None):
        q = w.d.qpos.copy(); r = {'name': name, 'metrics': w.metrics(), 'score': float(score(w)),
                                  'qpos': q.tolist(), 'cert': both(S, q)}
        if extra: r.update(extra)
        out['states'].append(r)
        print(f"[{name}] base {r['metrics']['base_height']:.3f} up {r['metrics']['up_z']:.3f} "
              f"feet {r['metrics']['foot_load']:.1f} nonfoot {r['metrics']['nonfoot_load']:.1f} KE {r['metrics']['kinetic_j']:.3f}")
        for k in ('all_mu0.6', 'feet_mu0.6', 'all_mu0.8', 'feet_mu0.8'):
            print('   ', k, brief(r['cert'][k]))
        return r

    # sanity: standing (B's experiment.standing)
    import experiment
    w.d.qpos[:] = experiment.standing(w.m); w.d.qvel[:] = 0
    import mujoco as mj; mj.mj_forward(w.m, w.d); rec('sanity_standing')
    ini, acts, fin = load_plateau()
    w.restore(ini); rec('passive_supine_seed101')
    for i, a in enumerate(acts):
        w.advance(a); rec(f'block_{i + 1}', {'action': np.asarray(a).tolist()})
    err = float(np.max(np.abs(w.snapshot()[:61] - fin[:61])))
    print('replay error vs saved final', err); out['replay_error'] = err
    w.restore(fin); rec('plateau_snapshot')
    np.save(OUT / 'plateau_snapshot.npy', w.snapshot())
    for k in range(5):
        w.advance(np.zeros(12)); rec(f'plateau_hold_{k + 1}')
    np.save(OUT / 'plateau_settled.npy', w.snapshot())
    (OUT / 'step1_plateau_cert.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
