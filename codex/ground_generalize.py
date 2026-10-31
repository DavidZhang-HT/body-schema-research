"""New passive initializations: fixed discovered schema path + balance feedback."""
from ground_schema import *
from ground_balance import balance
results=[]
for pose in ['prone','supine']:
 actions=np.load(OUT/f'oracle_{pose}_best.npz')['actions']
 for seed in [101,102,103]:
  w=GroundWorld();settle=w.passive(pose,seed);initial=w.snapshot();trace=[];metrics=[]
  for action in actions:
   _,frames=w.advance(action,trace=True);trace+=frames;metrics.append(w.metrics())
  success,frames,records=balance(w);trace+=frames
  r={'pose':pose,'seed':seed,'passive_stable':settle['settled'],'success':success,'final':w.metrics(),'method':'fixed schema path discovered by physics search, not neural selection'};results.append(r)
  np.savez_compressed(OUT/f'transfer_{pose}_{seed}.npz',initial=initial,trace=trace)
  print(r,flush=True)
(OUT/'transfer_results.json').write_text(json.dumps(results,indent=2))
