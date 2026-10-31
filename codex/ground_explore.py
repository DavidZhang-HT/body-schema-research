"""Exploration starts from unpowered settled poses; no motion-capture targets."""
from ground_schema import *

def generate():
 w=GroundWorld();rng=np.random.default_rng(703);x=[];u=[];y=[];groups=[];epids=[];initial=[];passive=[];e=0
 for case in range(24):
  pose=['supine','prone'][case%2];report=w.passive(pose,50+case//2)
  if not report['settled']:raise RuntimeError(f'Passive equilibrium not reached: {case}')
  snap=w.snapshot();initial.append(snap);passive.append({k:v for k,v in report.items() if k!='trace'})
  for trial in range(12):
   w.restore(snap)
   for step in range(6):
    action=rng.normal(0,.5,12)*BOUNDS
    if trial<4:
     action[:]=0;axis=(trial*6+step)%12;action[axis]=BOUNDS[axis]*([-1,1][trial%2])*.6
    if step<2:action*=.5
    action=np.clip(action,-BOUNDS,BOUNDS);before=w.observe();after,_=w.advance(action)
    x.append(before);u.append(action);y.append(after);groups.append(case);epids.append(e)
   e+=1
  print('passive case',case,'pairs',len(x),'end',w.metrics(),flush=True)
 np.savez_compressed(OUT/'exploration.npz',x=x,u=u,y=y,case=groups,episode=epids,initial=initial)
 (OUT/'exploration_manifest.json').write_text(json.dumps({'passive_cases':passive,'schemas':SCHEMAS,'duration':DURATION,'pairs':len(x),'scope':'Contact force/J^T ground load are ideal simulated sensor proxies. Body CoM is privileged model-derived state. No external assistance.'},indent=2))
if __name__=='__main__':generate()
