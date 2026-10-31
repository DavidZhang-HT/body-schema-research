"""Physics-guided exploration oracle. Its results are NOT learned-model-only control."""
from ground_schema import *
import time

def score(w):
 r=w.metrics();q=w.d.qpos[7:];foot=r['foot_load']/max(r['total_load'],1)
 foot_height=np.mean(w.d.xipos[[w.bodies[6],w.bodies[12]],2])
 return 4*r['com_height']+6*r['base_height']+1.2*r['up_z']+.6*min(r['foot_load']/108,1)-.12*r['kinetic_j']-.3*r['com_speed']-2*max(foot_height-r['com_height'],0)-max(0,.5-r['total_load']/108)

def search(pose='supine',depth=14,width=14,branches=16,resume=False):
 w=GroundWorld();w.passive(pose,0);initial=w.snapshot();rng=np.random.default_rng(2027)
 beam=[(score(w),initial,[])];tag=pose
 if resume:
  saved=np.load(OUT/f'oracle_{pose}_best.npz');initial=saved['initial'];w.restore(saved['final']);beam=[(score(w),w.snapshot(),list(saved['actions']))];tag=pose+'_refine';rng=np.random.default_rng(2028)
 x=[];u=[];y=[];snapshots=[];start=time.time();history=[]
 for level in range(depth):
  pool=list(beam)
  for value,snap,path in beam:
   for j in range(branches):
    action=rng.normal(0,.7,12)*BOUNDS
    # Half of the probes are coordinated symmetric motions; asymmetry remains available.
    if j<branches//2:action[9:]=0;action[3:5]*=.3;action[8]*=.3
    if j==0:action=np.zeros(12)
    action=np.clip(action,-BOUNDS,BOUNDS);w.restore(snap);before=w.observe();after,_=w.advance(action);v=score(w)
    x.append(before);u.append(action);y.append(after)
    pool.append((v,w.snapshot(),path+[action.copy()]))
  pool.sort(key=lambda e:e[0],reverse=True);beam=[];cells=set()
  for entry in pool:
   w.restore(entry[1]);r=w.metrics();q=w.d.qpos[7:];cell=(int(r['com_height']/.025),int(r['up_z']/.12),int((q[0]+q[6])/.6),int((q[3]+q[9])/.7))
   if cell not in cells:beam.append(entry);cells.add(cell)
   if len(beam)>=width:break
  if len(beam)<width:
   for entry in pool:
    if all(entry is not b for b in beam):beam.append(entry)
    if len(beam)>=width:break
  w.restore(beam[0][1]);r=w.metrics();history.append(r);print(pose,'depth',level+1,'score',round(beam[0][0],3),'metrics',r,'wall',round(time.time()-start),flush=True)
  np.savez_compressed(OUT/f'oracle_{tag}_best.npz',initial=initial,actions=beam[0][2],final=beam[0][1])
  if r['base_height']>.3 and r['up_z']>.92 and r['foot_load']>75 and r['nonfoot_load']<8 and r['com_speed']<.2:break
 np.savez_compressed(OUT/f'oracle_{tag}_data.npz',x=x,u=u,y=y)
 (OUT/f'oracle_{tag}_search.json').write_text(json.dumps({'type':'physics-guided exploration oracle, not learned-only planner','pose':pose,'history':history,'evaluations':len(x),'seconds':time.time()-start},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pose',default='supine');p.add_argument('--resume',action='store_true');p.add_argument('--depth',type=int,default=14);a=p.parse_args();search(a.pose,a.depth,resume=a.resume)
