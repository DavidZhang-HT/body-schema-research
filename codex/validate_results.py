"""Artifact integrity checks; these do not assert successful walking."""
import json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for mode in ['direct','projected','adapted']:
 a=np.load(ROOT/f'results/{mode}.npz');r=json.loads((ROOT/f'results/{mode}.json').read_text())
 assert np.isfinite(a['log']).all() and np.isfinite(a['qpos']).all()
 assert len(a['log'])==len(a['qpos'])
 assert np.max(abs(np.linalg.norm(a['qpos'][:,3:7],axis=1)-1))<1e-8
 assert np.max(a['log'][:,6])<=20+1e-9
 assert np.allclose(np.diff(a['log'][:,0]),.004)
 assert r['external_base_forces'] is False
 assert r['fall_time'] is not None
 assert abs(a['log'][-1,0]-r['fall_time'])<1e-9
 assert a['log'][-1,3]<.18 or a['log'][-1,4]>1
 assert r['qp_failures']==0
r=json.loads((ROOT/'results/adapted.json').read_text())
assert abs(r['calibration']['total_mass_estimate']-11.82873031)<.01
print('PASS: finite physics logs, unit quaternions, torque cap, time grid, fall labels, calibration sanity.')
