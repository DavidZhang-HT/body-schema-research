"""Attention-proposed, physics-validated schema planning. This uses a physics oracle."""
from ground_schema import *
from ground_model import Ensemble
from ground_search import score
from ground_imagine import value
from ground_balance import balance
import time

def run(pose='prone',seed=101,depth=24,width=14,proposals=64,validated=12):
 world=GroundWorld();settled=world.passive(pose,seed);initial=world.snapshot()
 net=Ensemble();rng=np.random.default_rng(3400+seed)
 beam=[(score(world),initial,[])];history=[];xs=[];us=[];ys=[];started=time.time();success=False
 tag=f'hybrid_{pose}_{seed}'
 for level in range(depth):
  states=[];candidates=[]
  for _,snap,_ in beam:
   world.restore(snap);state=world.observe();a=np.clip(rng.normal(0,.7,(proposals,12))*BOUNDS,-BOUNDS,BOUNDS)
   a[:proposals//2,9:]=0;a[:proposals//2,3:5]*=.3;a[:proposals//2,8]*=.3;a[0]=0
   states.extend(np.repeat(state[None],proposals,0));candidates.extend(a)
  states=np.array(states);candidates=np.array(candidates);pred,unc=net.predict(states,candidates);scores=value(pred,unc)
  pool=list(beam)
  for b,(_,snap,path) in enumerate(beam):
   offset=b*proposals;ranking=np.argsort(scores[offset:offset+proposals])[::-1];chosen=list(ranking[:validated-4])
   unused=[i for i in range(proposals) if i not in chosen];chosen+=list(rng.choice(unused,4,replace=False))
   for j in chosen:
    action=candidates[offset+j];world.restore(snap);before=world.observe();after,_=world.advance(action)
    xs.append(before);us.append(action);ys.append(after);pool.append((score(world),world.snapshot(),path+[action.copy()]))
  pool.sort(key=lambda e:e[0],reverse=True);beam=[];cells=set()
  for entry in pool:
   world.restore(entry[1]);r=world.metrics();q=world.d.qpos[7:]
   cell=(int(r['com_height']/.025),int(r['up_z']/.12),int((q[0]+q[6])/.6),int((q[3]+q[9])/.7))
   if cell not in cells:beam.append(entry);cells.add(cell)
   if len(beam)>=width:break
  if len(beam)<width:
   for entry in pool:
    if all(entry is not b for b in beam):beam.append(entry)
    if len(beam)>=width:break
  world.restore(beam[0][1]);r=world.metrics();history.append({'level':level+1,'physics_evaluations':len(xs),**r})
  print(pose,seed,'hybrid level',level+1,'evals',len(xs),'height',round(r['base_height'],3),'up',round(r['up_z'],3),'feet',round(r['foot_load'],1),'elapsed',round(time.time()-started),flush=True)
  np.savez_compressed(OUT/f'{tag}_best.npz',initial=initial,actions=beam[0][2],final=beam[0][1])
  # A standing handover is itself physically tested, not assumed from a scalar score.
  if r['base_height']>.16 and r['up_z']>.9 and r['foot_load']>75 and r['nonfoot_load']<8:
   success,_,_=balance(world)
   if success:break
 np.savez_compressed(OUT/f'{tag}_data.npz',x=xs,u=us,y=ys)
 # Independently execute from the original passive state without branching or resets.
 world.restore(initial);trace=[];actions=beam[0][2]
 for action in actions:
  _,frames=world.advance(action,trace=True);trace+=frames
 replay_error=float(np.max(abs(world.snapshot()[:61]-beam[0][1][:61])))
 success,frames,standing_history=balance(world);trace+=frames
 result={'pose':pose,'seed':seed,'passively_settled':settled['settled'],'success':success,'method':'Contact-attention proposes 64 actions per state; 12 are validated by cloned MuJoCo future rollouts (8 learned-score and 4 exploratory); beam planning then continuous torque-only replay and QP standing. Explicit physics oracle; not neural-only control.','physics_evaluations':len(xs),'schema_blocks':len(actions),'search_wall_seconds':time.time()-started,'continuous_replay_max_state_error':replay_error,'history':history,'standing_history':standing_history,'final':world.metrics()}
 (OUT/f'{tag}.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/f'{tag}_replay.npz',initial=initial,actions=actions,trace=trace)
 print(json.dumps({k:v for k,v in result.items() if k not in ['history','standing_history']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pose',default='prone');p.add_argument('--seed',type=int,default=101);p.add_argument('--depth',type=int,default=24);a=p.parse_args();run(a.pose,a.seed,a.depth)
