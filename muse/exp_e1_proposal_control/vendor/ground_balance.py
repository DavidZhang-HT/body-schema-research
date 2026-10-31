"""Optional handover to the existing torque-only standing controller."""
import inspect
import experiment
from experiment import *
from ground_schema import GroundWorld
from attention_data import motor_envelope
# Same QP, with foot orientation target rotated to the current heading.
source=inspect.getsource(Controller).replace('class Controller:', 'class BalanceController:').replace('R.from_matrix(rot.T).as_rotvec()', 'R.from_matrix(rot.T@self.target_rotation[k]).as_rotvec()')
exec(source)

def balance(w,seconds=5):
 experiment.DT=.004;ctrl=BalanceController(model());d=w.d;m=w.m
 yaw=R.from_matrix(d.xmat[w.base].reshape(3,3)).as_euler('xyz')[2];rot=R.from_euler('z',yaw);ctrl.target_rotation=[R.from_euler('z',R.from_matrix(f[1]).as_euler('xyz')[2]).as_matrix() for f in kinematics(m,d)]
 feet0=np.array([x[0] for x in kinematics(m,d)]);feet0[:,2]=0
 goal=standing(m);goal[19:]=d.qpos[19:]
 for side,baseidx in [('l',7),('r',13)]:
  sign=1 if d.qpos[baseidx+3]>=0 else -1
  goal[baseidx]=-.25*sign;goal[baseidx+3]=.5*sign;goal[baseidx+4]=-.25*sign
 test=mj.MjData(m);test.qpos[:]=goal;mj.mj_forward(m,test)
 offset=test.subtree_com[w.base,:2]-goal[:2];goal[:2]=feet0[:,:2].mean(0)-rot.apply(np.r_[offset,0])[:2];goal[3:7]=rot.as_quat()[[3,0,1,2]]
 start=d.qpos.copy();trace=[];records=[];stable=0;success=False
 for step in range(int(seconds/.004)):
  u=min(1,step*.004/3.5);u=u*u*(3-2*u);target=(1-u)*start+u*goal;target[3:7]/=np.linalg.norm(target[3:7]);tau=ctrl.solve(d.qpos,d.qvel,target,feet0,[True,True])
  for _ in range(2):
   w.actual+=(tau-w.actual)*(1-np.exp(-.002/.015));d.ctrl[:]=motor_envelope(w.actual,d.qvel[6:]);mj.mj_step(m,d)
  if step%5==0:
   r=w.metrics();records.append(r);trace.append(w.snapshot())
   okay=r['base_height']>.30 and r['up_z']>.95 and r['foot_load']>80 and r['nonfoot_load']<5 and r['com_speed']<.12
   stable=stable+1 if okay else 0
   if stable>=50:success=True
   if r['base_height']<.08 or r['up_z']<.35:break
 return bool(success and stable>=50),trace,records

if __name__=='__main__':
 from ground_schema import OUT
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--pose',default='prone');a=p.parse_args();w=GroundWorld();v=np.load(OUT/f'oracle_{a.pose}_best.npz');w.restore(v['final']);success,trace,records=balance(w)
 np.savez_compressed(OUT/f'balance_{a.pose}.npz',trace=trace);(OUT/f'balance_{a.pose}.json').write_text(json.dumps({'success':success,'final':records[-1]},indent=2));print(success,records[-1])
