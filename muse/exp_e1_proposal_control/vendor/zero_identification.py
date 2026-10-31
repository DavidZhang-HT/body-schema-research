"""Active encoder-zero identification. Estimator API accepts ONLY sensor packets.
q_encoder = q_physical + bias; control uses q_encoder - bias_estimate.
No physical joint position, plant body pose, or injection truth enters the estimator.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from dataclasses import dataclass
import json,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from experiment import model,standing,Controller,kinematics,DT,FOOT,ROOT
OUT=ROOT/'zero_results';OUT.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Packet:
    encoder: np.ndarray
    imu_quat: np.ndarray  # wxyz; ideal attitude with added measurement noise

def feet(m,d):
    return [(d.xpos[m.body(s+'_ankle_roll_link').id]+d.xmat[m.body(s+'_ankle_roll_link').id].reshape(3,3)@FOOT,
             d.xmat[m.body(s+'_ankle_roll_link').id].reshape(3,3).copy()) for s in ['l','r']]

class ZeroEstimator:
    def __init__(self,m):
        self.m=m;self.d=mj.MjData(m);self.x=np.zeros(12)
    def residual(self,x,packets):
        out=[]
        for p in packets:
            self.d.qpos[:]=0;self.d.qpos[3:7]=p.imu_quat
            self.d.qpos[7:]=p.encoder;self.d.qpos[7:19]-=x[:12]
            mj.mj_kinematics(self.m,self.d);f=feet(self.m,self.d)
            # Same level, flat soles; do not assume invariant xy/yaw under soft contact.
            out.extend(np.r_[(f[1][0][2]-f[0][0][2])/.001,
                              f[0][1][:2,2]/.003,f[1][1][:2,2]/.003])
        return np.array(out)
    def jacobian(self,x,packets):
        jac=[]
        for j in range(12):
            xp=x.copy();xm=x.copy();xp[j]+=1e-5;xm[j]-=1e-5
            jac.append((self.residual(xp,packets)-self.residual(xm,packets))/2e-5)
        return np.array(jac).T
    def fit(self,packets):
        lo=np.full(12,-np.deg2rad(6))
        hi=np.full(12,np.deg2rad(6))
        # No bias prior: report data-only observability rather than prior-induced certainty.
        res=least_squares(lambda x:self.residual(x,packets),self.x,bounds=(lo,hi),loss='soft_l1',f_scale=2,max_nfev=65,xtol=1e-8,ftol=1e-8,gtol=1e-8)
        self.x=res.x
        B=self.jacobian(self.x,packets)
        sv=np.linalg.svd(B,compute_uv=False)
        return {'success':bool(res.success),'cost':float(res.cost),'data_singular_values':sv.tolist(),'relative_rank_1e3':int(sum(sv>sv[0]*1e-3)),'condition':float(sv[0]/max(sv[-1],1e-12))}
    def choose(self,candidates,used,packets,bhat):
        J=self.jacobian(self.x,packets) if packets else np.zeros((0,12))
        # D-optimal information gain, normalized by declared parameter design scales.
        scale=np.full(12,.05)
        H=np.eye(12)+(J*scale).T@(J*scale)
        base=np.linalg.slogdet(H)[1];scores=[]
        for i,q in enumerate(candidates):
            if i in used:continue
            enc=q[7:].copy();enc[:12]+=bhat
            jac=self.jacobian(self.x,[Packet(enc,q[3:7])])*scale
            scores.append((float(np.linalg.slogdet(H+12*jac.T@jac)[1]-base),i))
        gain,idx=max(scores)
        return idx,gain

def candidate_poses(m,q0,seed=17):
    d=mj.MjData(m);d.qpos[:]=q0;mj.mj_kinematics(m,d);f0=feet(m,d)
    rng=np.random.default_rng(seed);qs=[]
    for _ in range(30):
        q=q0.copy();q[:3]+=rng.uniform([-.015,-.018,-.055],[.015,.018,-.008])
        q[3:7]=R.from_euler('xyz',rng.uniform([-.055,-.08,-.08],[.055,.08,.08])).as_quat()[[3,0,1,2]]
        def residual(leg):
            d.qpos[:]=q;d.qpos[7:19]=leg;mj.mj_kinematics(m,d);f=feet(m,d)
            return np.concatenate([np.r_[(f[k][0]-f0[k][0])*10,(R.from_matrix(f0[k][1].T@f[k][1])).as_rotvec()] for k in range(2)])
        res=least_squares(residual,q0[7:19],max_nfev=70)
        q[7:19]=res.x
        if np.linalg.norm(res.fun)<1e-4 and np.max(abs(q[7:19]))<1.6:
            # Conservative geometry prefilter; execution additionally checks loads/tilt.
            qs.append(q)
    assert len(qs)>=12
    return qs

def contact_sensor(m,d):
    loads=np.zeros(2);ground=m.geom('ground').id
    for i in range(d.ncon):
        c=d.contact[i]
        if ground not in [c.geom1,c.geom2]:continue
        other=c.geom2 if c.geom1==ground else c.geom1;b=m.geom_bodyid[other]
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)
        if name not in ['l_ankle_roll_link','r_ankle_roll_link']:continue
        w=np.zeros(6);mj.mj_contactForce(m,d,i,w)
        world=c.frame.reshape(3,3).T@w[:3]
        loads[0 if name.startswith('l_') else 1]+=abs(world[2])
    return loads

def interpolate(a,b,u):
    u=u*u*(3-2*u);q=(1-u)*a+u*b;q[3:7]/=np.linalg.norm(q[3:7]);return q

def run(seed=7,probes=7,selection='active'):
    plant=model();nom=model();d=mj.MjData(plant);q0=standing(nom);d.qpos[:]=q0;mj.mj_forward(plant,d)
    rng=np.random.default_rng(seed)
    # Plant-only fixture; estimator/selector never receive this value.
    truth=np.deg2rad(rng.choice([-1,1],12)*rng.uniform(1,3,12))
    controller=Controller(nom);est=ZeroEstimator(nom);candidates=candidate_poses(nom,q0)
    kd=mj.MjData(nom);kd.qpos[:]=q0;mj.mj_forward(nom,kd);foot0=np.array([f[0] for f in feet(nom,kd)]);foot0[:,2]=0
    bhat=np.zeros(12);applied=np.zeros(12);used=[];training=[];heldout=[];history=[];states=[];logs=[];packets_log=[];aborted=False;prev=q0.copy();static_x=None
    # Calibration targets train on indices <len-3; last three poses are never candidates.
    pool=candidates[:-3];validation=candidates[-3:]
    for block in range(probes+1+len(validation)):
        is_validation=block>probes
        if block==0:target=q0.copy();gain=0.;idx=-1
        elif is_validation:target=validation[block-probes-1];gain=0.;idx=-2
        else:
            if selection=='active':idx,gain=est.choose(pool,used,training,bhat)
            else:idx=next(i for i in range(len(pool)) if i not in used);gain=0.
            used.append(idx);target=pool[idx]
        duration=2.4 if block==0 else 2.0;block_packets=[]
        for s in range(int(duration/DT)):
            enc=d.qpos[7:].copy();enc[:12]+=truth;enc+=rng.normal(0,np.deg2rad(.01),24)
            imu=(R.from_quat(d.qpos[[4,5,6,3]])*R.from_rotvec(rng.normal(0,np.deg2rad(.02),3))).as_quat()[[3,0,1,2]]
            packet=Packet(enc,imu)
            applied+=np.clip(bhat-applied,-np.deg2rad(.015),np.deg2rad(.015))
            observed=np.r_[d.qpos[:3],imu,enc];observed[7:19]-=applied
            qt=interpolate(prev,target,min(1,s*DT/1.25))
            tau=controller.solve(observed,d.qvel.copy(),qt,foot0,[True,True]);d.ctrl[:]=tau
            mj.mj_step(plant,d);mj.mj_step(plant,d)
            loads=contact_sensor(plant,d);tilt=np.linalg.norm(R.from_quat(imu[[1,2,3,0]]).as_euler('xyz')[:2])
            states.append(d.qpos.copy());logs.append([d.time,block,*loads,tilt,np.max(abs(tau)),*applied])
            # Stationary tail only, and both feet loaded. No ground-truth foot pose gate.
            if s*DT>duration-.6 and s%10==0 and min(loads)>15 and np.linalg.norm(d.qvel)<.45:
                block_packets.append(packet);packets_log.append(np.r_[block,enc,imu])
            if d.qpos[2]<.24 or tilt>.35 or min(loads)<3 and s*DT>1.5:
                aborted=True;break
        prev=target.copy()
        if is_validation:heldout.extend(block_packets)
        else:
            training.extend(block_packets)
            if training:
                fit=est.fit(training)
                # Do not deploy unsupported offsets from a single posture.
                if block==0:static_x=est.x.copy()
                if block>=2 and fit['relative_rank_1e3']==12:bhat=est.x[:12].copy()
                history.append({'block':block,'candidate':idx,'expected_logdet_gain':gain,'accepted_packets':len(block_packets),'estimate_rad':est.x[:12].tolist(),**fit})
        print(f'seed={seed} block={block} packets={len(block_packets)} rank={history[-1]["relative_rank_1e3"] if history else 0} abort={aborted}',flush=True)
        if aborted:break
    fit=est.fit(training) if training else {}
    # Evaluation below is the only place the injected truth is compared with estimates.
    rmse=lambda b:float(np.rad2deg(np.sqrt(np.mean((b-truth)**2))))
    def held_metric(x):
        if not heldout:return None
        residual=est.residual(x,heldout).reshape(-1,5)
        return {'foot_height_difference_rmse_mm':float(np.sqrt(np.mean(residual[:,0]**2))), 'sole_normal_component_rmse_deg':float(np.rad2deg(.003*np.sqrt(np.mean(residual[:,1:]**2))))}
    zero_x=est.x.copy();zero_x[:12]=0
    report={'seed':seed,'selection':selection,'aborted':aborted,'simulated_seconds':d.time,'probes_requested':probes,'probes_executed':len(used),'training_packets':len(training),'heldout_packets':len(heldout),'noise_encoder_std_deg':.01,'noise_imu_std_deg':.02,'true_offsets_deg':np.rad2deg(truth).tolist(),'estimated_offsets_deg':np.rad2deg(est.x[:12]).tolist(),'applied_offsets_deg':np.rad2deg(applied).tolist(),'before_offset_rmse_deg':rmse(np.zeros(12)),'after_offset_rmse_deg':rmse(est.x[:12]),'static_only_offset_rmse_deg':rmse(static_x[:12]) if static_x is not None else None,'max_offset_error_deg':float(np.max(abs(np.rad2deg(est.x[:12]-truth)))),'heldout_before':held_metric(zero_x),'heldout_after':held_metric(est.x),'history':history,'observability':fit,'qp_failures':controller.failures,'max_torque_nm':float(np.max(np.array(logs)[:,5])),'min_foot_load_after_settle_N':float(np.min(np.array(logs)[500:,2:4])),'external_base_forces':False,'estimator_reads_truth':False,'controller_root_translation':'ideal simulator state; estimator does not receive it'}
    name=f'{selection}_seed{seed}';(OUT/f'{name}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/f'{name}.npz',qpos=states,log=logs,sensor_packets=packets_log,truth=truth,estimate=est.x,candidates=np.array(candidates))
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','true_offsets_deg','estimated_offsets_deg','applied_offsets_deg']},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=7);p.add_argument('--probes',type=int,default=7);p.add_argument('--selection',choices=['active','fixed'],default='active');a=p.parse_args();run(a.seed,a.probes,a.selection)
