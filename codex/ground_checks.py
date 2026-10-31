"""Physical checks and honest terminal standing criteria."""
from ground_schema import *
w=GroundWorld();passive=[]
for pose in ['supine','prone']:
 r=w.passive(pose,0);metrics=w.metrics()
 assert r['settled'] and np.max(abs(w.d.ctrl))==0 and r['kinetic_j']<.002
 assert abs(metrics['total_load']-sum(w.m.body_mass)*9.81)<3
 snap=w.snapshot();action=BOUNDS*.1;obs1,_=w.advance(action);end1=w.snapshot();w.restore(snap);obs2,_=w.advance(action)
 assert np.allclose(end1[:61],w.snapshot()[:61],atol=1e-9)
 assert np.allclose(obs1,obs2,atol=1e-6)
 passive.append({'pose':pose,'settled':True,'ground_load_N':metrics['total_load'],'weight_N':sum(w.m.body_mass)*9.81})
replays=[]
for pose in ['prone','supine']:
 r=json.loads((OUT/f'getup_replay_{pose}.json').read_text());hist=r['standing_history'];tail=hist[-50:]
 okay=all(h['base_height']>.30 and h['up_z']>.95 and h['foot_load']>80 and h['nonfoot_load']<5 and h['com_speed']<.12 for h in tail)
 assert len(tail)==50 and okay
 assert r['replay_endpoint_max_qpos_error']<1e-8
 a=np.load(OUT/f'getup_replay_{pose}.npz')['trace'];assert np.isfinite(a).all()
 assert np.max(abs(a[:,85:109])-w.m.actuator_ctrlrange[:,1])<=1e-9
 replays.append({'pose':pose,'final_one_second_standing_verified':okay,'continuous_replay_matches_search':True})
a=np.load(OUT/'exploration.npz');assert np.isfinite(a['x']).all() and np.isfinite(a['y']).all()
assert np.max(abs(a['u'])-BOUNDS)<1e-9
split=json.loads((OUT/'split.json').read_text());assert not(set(split['train_cases'])&set(split['test_cases']))
hybrid=[]
for path in sorted(OUT.glob('hybrid_*.json')):
 r=json.loads(path.read_text())
 if 'success' not in r:continue
 assert r['continuous_replay_max_state_error']<1e-8
 if r['success']:
  tail=r['standing_history'][-50:]
  assert len(tail)==50 and all(h['base_height']>.30 and h['up_z']>.95 and h['foot_load']>80 and h['nonfoot_load']<5 and h['com_speed']<.12 for h in tail)
  frames=np.load(OUT/f"hybrid_{r['pose']}_{r['seed']}_replay.npz")['trace']
  assert np.isfinite(frames).all() and np.max(abs(frames[:,85:109])-w.m.actuator_ctrlrange[:,1])<=1e-9
 hybrid.append({'pose':r['pose'],'seed':r['seed'],'success':r['success'],'continuous_replay_matches_search':True,'final_standing_verified':r['success']})
(OUT/'checks.json').write_text(json.dumps({'passive':passive,'replays':replays,'hybrid':hybrid,'data_finite':True,'no_initial_case_overlap':True},indent=2))
print('PASS: unpowered equilibria, weight balance, exact branch replay, final-second standing, applied torque limits and case isolation.')
