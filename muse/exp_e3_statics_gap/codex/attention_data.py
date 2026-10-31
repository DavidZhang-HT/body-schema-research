"""Generate independent floating-base MuJoCo episodes for action-conditioned learning."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,argparse
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
import experiment
from experiment import ROOT,model,standing,Controller,kinematics
from zero_identification import candidate_poses
OUT=ROOT/'attention_results';OUT.mkdir(exist_ok=True)
STEP=.02;HISTORY=8;STATE=67;ACTION=24
VELOCITY_LIMIT=np.r_[np.full(12,5.45),np.full(10,15.91),np.full(2,30.37)]

def motor_envelope(torque,velocity):
    drive=torque*velocity>0
    return torque*np.where(drive,np.clip(1-np.abs(velocity)/VELOCITY_LIMIT,0,1),1)

def observe(m,d):
    mj.mj_forward(m,d);mj.mj_subtreeVel(m,d)
    base=m.body('base_link').id;rot=d.xmat[base].reshape(3,3)
    contact=np.zeros(3);ground=m.geom('ground').id
    for i in range(d.ncon):
        c=d.contact[i]
        if ground not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==ground else c.geom1
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[other])
        contact[0 if name=='l_ankle_roll_link' else 1 if name=='r_ankle_roll_link' else 2]=1
    return np.r_[d.qpos[7:],d.qvel[6:],rot.T@np.array([0,0,1]),d.qvel[3:6],rot.T@d.qvel[:3],d.qpos[2],rot.T@(d.subtree_com[base]-d.qpos[:3]),rot.T@d.subtree_linvel[base],contact].astype(np.float32)

def load_reference(m):
    p=ROOT/'upstream/GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_walk1_subject1.csv'
    header=p.open().readline().strip().split(',');a=np.loadtxt(p,delimiter=',',skiprows=1);q=np.zeros((len(a),24))
    for i in range(1,m.njnt):
        n=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i).removesuffix('_joint')
        if n in header:q[:,i-1]=a[:,header.index(n)]
    return q

def generate(episodes=96,seconds=6):
    experiment.DT=STEP
    m=model();nom=model();q0=standing(m);poses=candidate_poses(m,q0);ref=load_reference(m)
    orig_mass=m.body_mass.copy();orig_inertia=m.body_inertia.copy();orig_friction=m.geom_friction.copy()
    ss=[];aa=[];meta=[];dtsteps=round(STEP/m.opt.timestep)
    families=['stand','squat','walk_reference','excited','supine_attempt','prone_attempt']
    for ep in range(episodes):
        rng=np.random.default_rng(1000+ep);family=families[ep%6]
        controller=Controller(nom)
        nd=mj.MjData(nom);nd.qpos[:]=q0;mj.mj_forward(nom,nd);foot0=np.array([f[0] for f in kinematics(nom,nd)]);foot0[:,2]=0
        scale=rng.uniform(.9,1.1);m.body_mass[:]=orig_mass*scale;m.body_inertia[:]=orig_inertia*scale;m.geom_friction[:]=orig_friction;m.geom_friction[:,0]=rng.uniform(.65,1.)
        d=mj.MjData(m);mj.mj_setConst(m,d);d.qpos[:]=q0
        if family in ['prone_attempt','supine_attempt']:
            d.qpos[3:7]=R.from_euler('y',np.pi/2 if family=='prone_attempt' else -np.pi/2).as_quat()[[3,0,1,2]];d.qpos[2]=.28
        else:
            d.qpos[7:19]+=rng.normal(0,.012,12);d.qvel[:2]=rng.normal(0,.04,2)
        mj.mj_forward(m,d);actual=np.zeros(24);lag=rng.uniform(.008,.035)
        kp=rng.uniform(35,65);kd=rng.uniform(1.5,3.2);amp=rng.uniform(.02,.09,24);freq=rng.uniform(.25,1.2,24);phase=rng.uniform(0,2*np.pi,24)
        ids=rng.choice(len(poses),4,replace=False);start=int(rng.integers(90,250));speed=rng.uniform(.65,1.2);states=[observe(m,d)];acts=[]
        for k in range(round(seconds/STEP)):
            t=k*STEP;target=q0[7:].copy()
            if family=='stand':target+=amp*.3*np.sin(2*np.pi*freq*t+phase)
            elif family=='squat':
                phase_i=t/1.5;i=int(phase_i)%3;u=phase_i-int(phase_i);u=u*u*(3-2*u);target=(1-u)*poses[ids[i]][7:]+u*poses[ids[i+1]][7:]
            elif family=='walk_reference':
                ix=min(len(ref)-1,start+int(t*30*speed));target=.8*ref[ix]+.2*target
            elif family=='excited':target+=amp*3*np.sin(2*np.pi*freq*t+phase)
            else:
                # Attempted motions only: there is no successful get-up controller here.
                target+=amp*2*np.sin(2*np.pi*freq*t+phase)
                target[[3,9]]+=.7*(.5+.5*np.sin(2*np.pi*.35*t));target[[15,20]]=-1.1
            command=np.clip(kp*(target-d.qpos[7:])-kd*d.qvel[6:]+rng.normal(0,.15,24),m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
            if family in ['stand','squat']:
                qt=q0.copy()
                if family=='squat':
                    qt=poses[ids[i]]*(1-u)+poses[ids[i+1]]*u;qt[3:7]/=np.linalg.norm(qt[3:7])
                else:
                    qt[0]+=.006*np.sin(t*2);qt[1]+=.006*np.sin(t*1.7)
                command=controller.solve(d.qpos,d.qvel,qt,foot0,[True,True])
            for _ in range(dtsteps):
                actual+=(command-actual)*(1-np.exp(-m.opt.timestep/lag));d.ctrl[:]=motor_envelope(actual,d.qvel[6:]);mj.mj_step(m,d)
            acts.append(command);states.append(observe(m,d))
        ss.append(states);aa.append(acts);meta.append({'episode':ep,'seed':1000+ep,'family':family,'mass_scale':scale,'actuator_time_constant':lag})
        if ep%12==0:print('generated',ep+1,'/',episodes,flush=True)
    s=np.asarray(ss,dtype=np.float32);a=np.asarray(aa,dtype=np.float32)
    assert np.isfinite(s).all() and np.isfinite(a).all()
    np.savez_compressed(OUT/'transitions.npz',states=s,actions=a)
    (OUT/'data_manifest.json').write_text(json.dumps({'step_seconds':STEP,'history_steps':HISTORY,'episodes':meta,'state_layout':{'joint_angle':[0,24],'joint_velocity':[24,48],'up_vector':[48,51],'angular_velocity':[51,54],'local_base_velocity':[54,57],'base_height':[57,58],'local_com_position':[58,61],'local_com_velocity':[61,64],'contacts':[64,67]},'input_note':'Privileged simulation state incl. CoM; encoders assumed calibrated. No external base assistance.','motor_model':'First-order lag plus linear torque-speed envelope using URDF velocity limits; engineering assumption, not measured motor data.','ground_motion_note':'Prone/supine attempts are data coverage, not successful get-up demonstrations.'},indent=2))
    print('saved',s.shape,a.shape,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=96);p.add_argument('--seconds',type=float,default=6);x=p.parse_args();generate(x.episodes,x.seconds)
