"""Flat-ground passive equilibrium, contact sensing and bounded body-schema probes."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np,json,argparse
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from experiment import model,standing,ROOT
from attention_data import motor_envelope
OUT=ROOT/'ground_results';OUT.mkdir(exist_ok=True)
NODES=25;NODE_DIM=15;GLOBAL_DIM=23;DIM=NODES*NODE_DIM+GLOBAL_DIM
DURATION=.6;NSCHEMA=12
B=np.zeros((24,12))
for j,c,sgn in [(0,0,1),(6,0,1),(3,1,1),(9,1,1),(4,2,1),(10,2,1),(1,3,1),(7,3,-1),(2,4,1),(8,4,-1),(12,5,1),(17,5,1),(13,6,1),(18,6,-1),(15,7,1),(20,7,1),(14,8,1),(19,8,-1),(0,9,1),(6,9,-1),(12,10,1),(17,10,-1),(15,11,1),(20,11,-1)]:B[j,c]=sgn
BOUNDS=np.array([.55,.65,.40,.30,.30,.8,.6,.8,.5,.30,.5,.5])
SCHEMAS=['bilateral_hip_pitch','bilateral_knee','bilateral_ankle_pitch','hip_spread','hip_yaw','bilateral_shoulder_pitch','shoulder_spread','bilateral_elbow','upper_arm_yaw','differential_hip_pitch','differential_shoulder_pitch','differential_elbow']

class GroundWorld:
 def __init__(self):
  self.m=model();self.d=mj.MjData(self.m);self.base=self.m.body('base_link').id
  self.bodies=[self.base]+[int(self.m.jnt_bodyid[j]) for j in range(1,25)]
  self.body2node={b:i for i,b in enumerate(self.bodies)};self.actual=np.zeros(24)
  self.qlo=self.m.jnt_range[1:,0]+.02;self.qhi=self.m.jnt_range[1:,1]-.02
  self.kp=np.r_[np.full(12,50),np.full(10,25),np.full(2,5)];self.kd=np.r_[np.full(12,2.0),np.full(10,1.0),np.full(2,.3)]
 def snapshot(self):return np.r_[self.d.qpos,self.d.qvel,self.actual,self.d.ctrl,self.d.qacc_warmstart,self.d.time]
 def restore(self,v):
  mj.mj_resetData(self.m,self.d);self.d.qpos[:]=v[:31];self.d.qvel[:]=v[31:61];self.actual[:]=v[61:85];self.d.time=v[-1]
  self.d.ctrl[:]=v[85:109] if len(v)>86 else motor_envelope(self.actual,self.d.qvel[6:])
  mj.mj_forward(self.m,self.d)
  if len(v)>86:self.d.qacc_warmstart[:]=v[109:139]
 def passive(self,pose='supine',seed=0):
  rng=np.random.default_rng(seed);mj.mj_resetData(self.m,self.d);self.actual[:]=0
  self.d.qpos[:]=standing(self.m);self.d.qpos[2]=.26
  self.d.qpos[3:7]=R.from_euler('xyz',[rng.uniform(-.05,.05),(-1 if pose=='supine' else 1)*np.pi/2,rng.uniform(-.1,.1)]).as_quat()[[3,0,1,2]]
  self.d.qpos[7:]+=rng.normal(0,.015,24);mj.mj_forward(self.m,self.d)
  trace=[];stable=0
  for k in range(6000):
   self.d.ctrl[:]=0;mj.mj_step(self.m,self.d)
   if k%20==0:
    kinetic=self.kinetic();trace.append([self.d.time,kinetic,np.linalg.norm(self.d.qvel[:6]),np.max(abs(self.d.qvel[6:]))])
    stable=stable+1 if kinetic<.002 and np.linalg.norm(self.d.qvel[:6])<.03 and np.max(abs(self.d.qvel[6:]))<.15 else 0
    if stable>=15:break
  return {'pose':pose,'seed':seed,'settled':stable>=15,'seconds':self.d.time,'kinetic_j':self.kinetic(),'trace':trace}
 def kinetic(self):
  M=np.zeros((self.m.nv,self.m.nv));mj.mj_fullM(self.m,self.d,M)
  return float(.5*self.d.qvel@M@self.d.qvel)
 def contacts(self):
  m,d=self.m,self.d;forces=np.zeros((25,3));moments=np.zeros((25,3));points=np.zeros((25,3));weight=np.zeros(25);load=np.zeros(m.nv)
  ground=m.geom('ground').id
  for i in range(d.ncon):
   c=d.contact[i]
   if ground not in [c.geom1,c.geom2]:continue
   other=c.geom2 if c.geom1==ground else c.geom1;b=int(m.geom_bodyid[other])
   if b not in self.body2node:continue
   node=self.body2node[b];w=np.zeros(6);mj.mj_contactForce(m,d,i,w);f=c.frame.reshape(3,3).T@w[:3];torque=c.frame.reshape(3,3).T@w[3:]
   if c.geom2==ground:f=-f;torque=-torque
   forces[node]+=f;moments[node]+=np.cross(c.pos-d.xpos[b],f)+torque
   weight[node]+=max(f[2],0);points[node]+=max(f[2],0)*(c.pos-d.qpos[:3])
   jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jac(m,d,jp,jr,c.pos,b);load+=jp.T@f+jr.T@torque
  points/=np.maximum(weight[:,None],1e-8)
  return forces,points,weight,load
 def observe(self):
  m,d=self.m,self.d;mj.mj_forward(m,d);mj.mj_subtreeVel(m,d);rot=d.xmat[self.base].reshape(3,3)
  force,point,weight,load=self.contacts();node=np.zeros((25,15))
  node[1:,0]=d.qpos[7:];node[1:,1]=d.qvel[6:];node[1:,2]=d.qfrc_actuator[6:];node[1:,3]=load[6:]
  node[:,4:7]=d.xipos[self.bodies]-d.qpos[:3];node[:,7:10]=force;node[:,10:13]=point;node[:,13]=(weight>1).astype(float);node[:,14]=d.xipos[self.bodies,2]
  global_=np.r_[rot[:,:2].T.ravel(),d.qvel[:3],d.qvel[3:6],d.qpos[2],d.subtree_com[self.base]-d.qpos[:3],d.subtree_linvel[self.base],d.subtree_com[self.base,2],rot[:,2]]
  return np.r_[node.ravel(),global_].astype(np.float32)
 def metrics(self):
  m,d=self.m,self.d;mj.mj_forward(m,d);mj.mj_subtreeVel(m,d);f,p,w,l=self.contacts();up=d.xmat[self.base].reshape(3,3)[2,2]
  foot=w[6]+w[12];other=w.sum()-foot
  return {'com_height':float(d.subtree_com[self.base,2]),'base_height':float(d.qpos[2]),'up_z':float(up),'kinetic_j':self.kinetic(),'foot_load':float(foot),'nonfoot_load':float(other),'total_load':float(w.sum()),'com_speed':float(np.linalg.norm(d.subtree_linvel[self.base]))}
 def advance(self,delta,duration=DURATION,trace=False):
  start=self.d.qpos[7:].copy();target=np.clip(start+B@np.clip(delta,-BOUNDS,BOUNDS),self.qlo,self.qhi)
  rows=[];count=round(duration/self.m.opt.timestep);command=np.zeros(24)
  for k in range(count):
   if k%5==0:
    u=min(1,(k*self.m.opt.timestep)/min(.25,.5*duration));u=u*u*(3-2*u)
    qt=(1-u)*start+u*target
    command=np.clip(self.kp*(qt-self.d.qpos[7:])-self.kd*self.d.qvel[6:],self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
   self.actual+=(command-self.actual)*(1-np.exp(-self.m.opt.timestep/.015));self.d.ctrl[:]=motor_envelope(self.actual,self.d.qvel[6:]);mj.mj_step(self.m,self.d)
   if trace and k%10==0:rows.append(self.snapshot())
  return self.observe(),rows

if __name__=='__main__':
 w=GroundWorld();reports=[];states=[]
 for pose in ['supine','prone']:
  r=w.passive(pose);states.append(w.snapshot());reports.append(r);print({k:v for k,v in r.items() if k!='trace'},w.metrics(),flush=True)
 np.savez(OUT/'passive.npz',states=states);(OUT/'passive.json').write_text(json.dumps(reports,indent=2))
