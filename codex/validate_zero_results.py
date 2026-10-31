"""Validate actual rollouts and sensor-only reconstruction of the reported estimate."""
from zero_identification import *
for selection in ['active','fixed']:
 for seed in [7,11,23]:
  name=f'{selection}_seed{seed}';r=json.loads((OUT/f'{name}.json').read_text());a=np.load(OUT/f'{name}.npz');log=a['log']
  assert not r['aborted'] and r['qp_failures']==0
  assert r['observability']['relative_rank_1e3']==12
  assert np.isfinite(a['qpos']).all() and np.isfinite(log).all()
  assert np.allclose(np.linalg.norm(a['qpos'][:,3:7],axis=1),1,atol=1e-8)
  assert np.allclose(np.diff(log[:,0]),DT)
  assert np.max(log[:,5])<=20
  assert np.max(abs(np.diff(log[:,6:],axis=0)))<=np.deg2rad(.015)+1e-12
  assert np.all(log[log[:,1]==0,6:]==0), 'Unobservable initial estimate was applied'
  assert r['training_packets']==120 and r['heldout_packets']==45
  assert len({h['candidate'] for h in r['history'][1:]})==7
  assert r['after_offset_rmse_deg']<r['before_offset_rmse_deg']
  assert r['heldout_after']['sole_normal_component_rmse_deg']<r['heldout_before']['sole_normal_component_rmse_deg']
rows=np.loadtxt(OUT/'active_seed7_sensors.csv',delimiter=',',skiprows=1)
packets=[Packet(row[1:25],row[25:29]) for row in rows if row[0]<=7]
e=ZeroEstimator(model());e.fit(packets)
r=json.loads((OUT/'active_seed7.json').read_text())
assert np.max(abs(np.rad2deg(e.x)-r['estimated_offsets_deg']))<.001
print('PASS: all six physics runs, support completion, finite states, torque and update-rate bounds, observability gating, withheld split, sensor-only replay within 0.001 deg.')
