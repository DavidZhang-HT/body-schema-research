"""Pi Plus: torque-only floating-base physics, contact-constrained ID, sensor calibration."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json, argparse
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from scipy import sparse
import osqp
ROOT=Path(__file__).resolve().parent
DT=.004
FOOT=np.array([-.025,0,-.05])

def model(): return mj.MjModel.from_xml_path(str(ROOT/'assets/scene.xml'))
def kinematics(m,d):
    out=[]
    for side in ['l','r']:
        b=m.body(side+'_ankle_roll_link').id
        rot=d.xmat[b].reshape(3,3);pos=d.xpos[b]+rot@FOOT
        jp=np.zeros((3,m.nv));jr=jp.copy();mj.mj_jac(m,d,jp,jr,pos,b)
        out.append((pos,rot,np.vstack([jp,jr])))
    return out

def standing(m):
    d=mj.MjData(m);d.qpos[3]=1
    for side in ['l','r']:
        for j,v in [('hip_pitch',-.25),('calf',.5),('ankle_pitch',-.25),('shoulder_roll',.25 if side=='l' else -.25)]:
            d.qpos[m.joint(side+'_'+j+'_joint').qposadr[0]]=v
    mj.mj_forward(m,d);feet=kinematics(m,d)
    d.qpos[2]-=min(f[0][2] for f in feet)-.0005;mj.mj_forward(m,d)
    # CoM over midfeet. Move both ankle pitches by tiny IK is unnecessary: root tasks settle it.
    return d.qpos.copy()

class Reference:
    def __init__(self,m):
        path=ROOT/'upstream/GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_walk1_subject1.csv'
        header=path.open().readline().strip().split(',');a=np.loadtxt(path,delimiter=',',skiprows=1)
        q=np.zeros((len(a),m.nq));q[:,:3]=a[:,:3];q[:,3:7]=a[:,[6,3,4,5]]
        for j in range(1,m.njnt):
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).removesuffix('_joint')
            if name in header:q[:,m.jnt_qposadr[j]]=a[:,header.index(name)]
        # Use first 12 seconds after initial retargeting transient. Preserve source motion.
        q=q[30:390].copy()
        yaw=R.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];rot=R.from_euler('z',-yaw)
        q[:,:3]=rot.apply(q[:,:3]-np.r_[q[0,:2],0]);q[:,3:7]=(rot*R.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
        d=mj.MjData(m);fp=[]
        for qi in q:
            d.qpos[:]=qi;mj.mj_forward(m,d);fp.append([f[0] for f in kinematics(m,d)])
        fp=np.array(fp);self.z_offset=-np.percentile(fp[:,:,2],5)+.001
        q[:,2]+=self.z_offset;fp[:,:,2]+=self.z_offset
        self.q=q;self.feet=fp
        np.savez(ROOT/'results/reference.npz',qpos=q,feet=fp,fps=30,z_offset=self.z_offset)
    def get(self,t):
        x=np.clip(t*30,0,len(self.q)-1.001);i=int(x);u=x-i
        q=(1-u)*self.q[i]+u*self.q[i+1];q[3:7]/=np.linalg.norm(q[3:7])
        return q

class Controller:
    def __init__(self,m):
        self.m=m;self.d=mj.MjData(m);self.lastJ=None;self.failures=0;self.residual=0
    def solve(self,q,v,target,foot_targets,stance):
        m,d=self.m,self.d;d.qpos[:]=q;d.qvel[:]=v;mj.mj_forward(m,d)
        n=m.nv;na=m.nu;N=n+12+na
        M=np.zeros((n,n));mj.mj_fullM(m,d,M)
        feet=kinematics(m,d);J=np.vstack([f[2] for f in feet]);jd=np.zeros(12) if self.lastJ is None else (J-self.lastJ)@v/DT;self.lastJ=J.copy();jd=np.clip(jd,-50,50)
        err=np.zeros(n);mj.mj_differentiatePos(m,err,1,q,target)
        des=100*err-20*v;des[:3]=140*err[:3]-24*v[:3];des[3:6]=160*err[3:6]-25*v[3:6]
        # Joint posture is soft; base and contact tasks can deform the reference.
        weights=np.r_[np.full(3,25),np.full(3,12),np.full(na,.18)]
        H=np.eye(N)*1e-7;g=np.zeros(N)
        H[:n,:n]+=np.diag(weights);g[:n]-=weights*des
        for k,(pos,rot,jac) in enumerate(feet):
            ft=foot_targets[k]
            er=np.r_[ft-pos,R.from_matrix(rot.T).as_rotvec()]
            # orientation error expressed in world frame
            er[3:]=rot@er[3:]
            goal=np.r_[500*er[:3],150*er[3:]]-30*(jac@v)-jd[6*k:6*k+6]
            wt=np.diag([80,80,100,15,15,8] if stance[k] else [35,35,60,4,4,2])
            H[:n,:n]+=jac.T@wt@jac;g[:n]-=jac.T@wt@goal
        H[n:n+12,n:n+12]+=np.eye(12)*.0003;H[n+12:,n+12:]+=np.eye(na)*.0001
        Aeq=np.zeros((n,N));Aeq[:,:n]=M;Aeq[:,n:n+12]=-J.T;Aeq[6:,n+12:]=-np.eye(na)
        rhs=-d.qfrc_bias+d.qfrc_passive
        rows=[Aeq];lo=[rhs];hi=[rhs]
        for k in range(2):
            start=n+6*k
            if not stance[k]:
                c=np.zeros((6,N));c[:,start:start+6]=np.eye(6);rows.append(c);lo.append(np.zeros(6));hi.append(np.zeros(6));continue
            c=np.zeros((11,N));c[0,start+2]=1
            for row,(axis,coef,sgn) in enumerate([(a,c,s) for a,c in [(0,.7),(1,.7),(3,.030),(4,.075),(5,.015)] for s in [-1,1]],1):
                c[row,start+axis]=sgn;c[row,start+2]=-coef
            rows.append(c);lo.append(np.r_[0,np.full(10,-np.inf)]);hi.append(np.r_[250,np.zeros(10)])
        c=np.zeros((na,N));c[:,n+12:]=np.eye(na);rows.append(c);lo.append(m.actuator_ctrlrange[:,0]);hi.append(m.actuator_ctrlrange[:,1])
        # Velocity limits over a short prediction horizon (URDF legs/arms/head).
        c=np.zeros((na,N));c[:,:n]=np.eye(n)[6:];rows.append(c)
        vmax=np.r_[np.full(12,5.45),np.full(10,15.91),np.full(2,30.37)]
        lo.append((-vmax-v[6:])/.04);hi.append((vmax-v[6:])/.04)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(H)),q=g,A=sparse.csc_matrix(np.vstack(rows)),l=np.concatenate(lo),u=np.concatenate(hi),verbose=False,eps_abs=2e-3,eps_rel=2e-3,max_iter=3000,polishing=False)
        res=solver.solve()
        if res.info.status_val not in (1,2):
            self.failures+=1
            return np.clip(35*err[6:]-2*v[6:],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        self.residual=max(self.residual,float(np.max(np.abs(Aeq@res.x-rhs))))
        return np.clip(res.x[n+12:],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])

def wrench_sensor(m,d):
    force=np.zeros(3);moment=np.zeros(3)
    ground=m.geom('ground').id
    for i in range(d.ncon):
        c=d.contact[i]
        if ground not in (c.geom1,c.geom2):continue
        w=np.zeros(6);mj.mj_contactForce(m,d,i,w);f=c.frame.reshape(3,3).T@w[:3];tor=c.frame.reshape(3,3).T@w[3:]
        if c.geom2==ground:f=-f;tor=-tor
        force+=f;moment+=np.cross(c.pos,f)+tor
    cop=np.array([-moment[1],moment[0]])/max(force[2],1e-5)
    return force,cop

def run(mode='projected',duration=10,render=False,calibrate=True,perturb=True):
    plant=model();nom=model();base=plant.body('base_link').id
    if perturb:plant.body_mass[base]*=1.3;plant.body_inertia[base]*=1.3;plant.body_ipos[base,0]+=.025
    pd=mj.MjData(plant);mj.mj_setConst(plant,pd)
    ref=Reference(nom);q0=standing(nom)
    pd.qpos[:]=q0;mj.mj_forward(plant,pd)
    controller=Controller(nom);nd=mj.MjData(nom);nd.qpos[:]=q0;mj.mj_forward(nom,nd)
    foot0=np.array([f[0] for f in kinematics(nom,nd)]);foot0[:,2]=0
    records=[];states=[];samples=[];est={};frames=[];renderer=None
    if render:
        renderer=mj.Renderer(plant,480,640);camera=mj.MjvCamera();camera.distance=1.65;camera.azimuth=130;camera.elevation=-15
    caltime=6.;blend=2.;total=caltime+blend+duration;laststance=np.ones(2,dtype=bool);anchors=foot0.copy();events=0;fall=None
    for step in range(int(total/DT)):
        t=step*DT;target=q0.copy();phase='calibration' if t<caltime else 'walking'
        stance=np.ones(2,dtype=bool);ft=foot0.copy()
        if t<caltime:
            if calibrate:
                target[0]+=.006*np.sin(2*np.pi*.4*t);target[1]+=.006*np.sin(2*np.pi*.3*t)
        else:
            walk_t=max(0,t-caltime-blend)*.6
            qr=ref.get(walk_t)
            # Fade arms and posture into reference, keep initial feet during transition.
            u=min(1,(t-caltime)/blend);u=u*u*(3-2*u)
            target=(1-u)*q0+u*qr;target[3:7]/=np.linalg.norm(target[3:7])
            if mode!='direct':
                # Flatten torso orientation and remove measured reference root-height drift.
                yaw=R.from_quat(target[[4,5,6,3]]).as_euler('xyz')[2]
                target[3:7]=R.from_euler('z',yaw).as_quat()[[3,0,1,2]]
                target[2]=q0[2]-.035*u
                target[0]*=.5
            if t>=caltime+blend:
                nd.qpos[:]=qr;mj.mj_forward(nom,nd);rf=np.array([f[0] for f in kinematics(nom,nd)])
                # Floor-relative mocap foot heights define support; always retain lowest foot.
                rf[:,0]=foot0[:,0]+.5*(rf[:,0]-ref.feet[0,:,0])
                h=rf[:,2];stance=h < min(h)+.012;stance[np.argmin(h)]=True
                ft=rf.copy();ft[:,2]=np.maximum(0,h-np.min(h))
                for k in range(2):
                    if stance[k] and not laststance[k]:anchors[k]=ft[k];anchors[k,2]=0;events+=1
                    if stance[k]:ft[k]=anchors[k]
                laststance=stance.copy()
                if mode!='direct' and sum(stance)==1:
                    target[1]=.85*ft[np.argmax(stance),1]+.15*target[1]
        if mode=='direct' and t>=caltime:
            e=target[7:]-pd.qpos[7:];torque=np.clip(45*e-2.2*pd.qvel[6:],plant.actuator_ctrlrange[:,0],plant.actuator_ctrlrange[:,1])
        else:torque=controller.solve(pd.qpos,pd.qvel,target,ft,stance)
        pd.ctrl[:]=torque
        for _ in range(2):mj.mj_step(plant,pd)
        force,cop=wrench_sensor(plant,pd)
        if .8<t<caltime-.2 and np.linalg.norm(pd.qvel[:3])<.08:
            nd.qpos[:]=pd.qpos;nd.qvel[:]=0;mj.mj_forward(nom,nd)
            samples.append((force[2]/9.81,cop.copy(),nd.subtree_com[base].copy(),pd.qpos[:3].copy(),pd.qpos[3:7].copy()))
        if step==int(caltime/DT)-1 and calibrate and mode=='adapted' and samples:
            vals=np.array([s[0] for s in samples]);mass=float(np.median(vals));oldmass=sum(nom.body_mass);delta=np.clip(mass-oldmass,-1.5,1.5)
            # First-moment conservation: infer root-link COM correction from quasi-static COP.
            offsets=[]
            for mass_i,cop_i,com_i,pos_i,quat_i in samples:
                predicted=(oldmass*com_i[:2]+delta*pos_i[:2])/(oldmass+delta)
                offsets.append(R.from_quat(quat_i[[1,2,3,0]]).inv().apply(np.r_[cop_i-predicted,0])[:2])
            newmass=nom.body_mass[base]+delta
            # Root added mass carries old local COM; account for its moment explicitly.
            corr=np.median(offsets,axis=0)*(oldmass+delta)/newmass-delta/newmass*nom.body_ipos[base,:2]
            corr=np.clip(corr,-.06,.06)
            nom.body_mass[base]=newmass;nom.body_ipos[base,:2]+=corr;mj.mj_setConst(nom,nd)
            est={'samples':len(samples),'total_mass_estimate':mass,'nominal_total_mass':oldmass,'base_mass_delta':float(delta),'base_com_xy_correction':corr.tolist(),'mass_sample_std':float(np.std(vals))}
        rot=R.from_quat(pd.qpos[[4,5,6,3]]).as_euler('xyz');tilt=np.linalg.norm(rot[:2])
        states.append(pd.qpos.copy())
        records.append([t,*pd.qpos[:3],tilt,float(np.sqrt(np.mean((target[7:]-pd.qpos[7:])**2))),float(np.max(np.abs(torque))),force[2],*cop,*stance.astype(int),np.linalg.norm(pd.qvel[6:12],ord=np.inf)])
        if render and step%25==0:
            camera.lookat[:]=[pd.qpos[0],pd.qpos[1],.28];renderer.update_scene(pd,camera);frames.append(renderer.render().copy())
        if pd.qpos[2]<.18 or tilt>1.0:
            fall=t;break
    if renderer:renderer.close()
    name=mode+('_nominal' if not perturb else '')
    a=np.array(records);np.savez(ROOT/f'results/{name}.npz',log=a,qpos=np.array(states),final_qpos=pd.qpos)
    result={'mode':mode,'plant_perturbed':perturb,'simulated_seconds':float(a[-1,0]),'requested_seconds':total,'fall_time':fall,'walking_survival_seconds':max(0,float(a[-1,0])-caltime-blend),'xy_displacement_m':float(np.linalg.norm(a[-1,1:3]-a[0,1:3])),'max_tilt_rad':float(a[:,4].max()),'joint_rmse_rad':float(a[:,5].mean()),'planned_contact_transitions':events,'qp_failures':controller.failures,'max_dynamics_residual':controller.residual,'calibration':est,'reference_z_offset':ref.z_offset,'external_base_forces':False}
    (ROOT/f'results/{name}.json').write_text(json.dumps(result,indent=2))
    if render:
        import imageio.v2 as imageio
        imageio.mimsave(ROOT/f'results/{name}.mp4',frames,fps=1/(DT*25),quality=7)
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='adapted',choices=['direct','projected','adapted']);p.add_argument('--duration',type=float,default=10);p.add_argument('--render',action='store_true');p.add_argument('--nominal',action='store_true');a=p.parse_args();run(a.mode,a.duration,a.render,perturb=not a.nominal)
