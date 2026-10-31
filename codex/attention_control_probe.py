"""Diagnostic closed-loop action filter, not a trained walking/get-up policy."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import torch
import mujoco as mj
import experiment
from experiment import Controller,kinematics
from attention_data import OUT,STEP,HISTORY,model,standing,observe,load_reference,motor_envelope
from attention_train import BodyAttention

def run(filtered,seed=4001):
    experiment.DT=STEP
    m=model();nom=model();d=mj.MjData(m);q0=standing(m);d.qpos[:]=q0;mj.mj_forward(m,d)
    ctrl=Controller(nom);nd=mj.MjData(nom);nd.qpos[:]=q0;mj.mj_forward(nom,nd);foot0=np.array([x[0] for x in kinematics(nom,nd)]);foot0[:,2]=0
    ref=load_reference(m);rng=np.random.default_rng(seed);lag=rng.uniform(.008,.035);actual=np.zeros(24);states=[];actions=[];chosen=[];hist=[];fall=None
    if filtered:
        ck=torch.load(OUT/'attention.pt',weights_only=True);net=BodyAttention();net.load_state_dict(ck['state_dict']);net.eval();norm={k:v.numpy() for k,v in ck['normalization'].items()}
    for k in range(300):
        s=observe(m,d);target=q0[7:].copy()
        if k>=25:target=.8*ref[min(len(ref)-1,90+int((k-25)*STEP*30*.8))]+.2*target
        ref_u=np.clip(45*(target-d.qpos[7:])-2.2*d.qvel[6:],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        stand_u=ctrl.solve(d.qpos,d.qvel,q0,foot0,[True,True])
        chosen_alpha=1.
        if k<25:u=stand_u
        elif filtered:
            alphas=np.array([0,.25,.5,.75,1.]);cands=alphas[:,None]*ref_u+(1-alphas[:,None])*stand_u
            past=np.array(hist[-HISTORY+1:]);hh=np.concatenate([np.repeat(past[None],5,axis=0),np.concatenate([np.repeat(s[None],5,axis=0),cands],-1)[:,None]],1)
            cost=np.zeros(5)
            for h in range(10):
                with torch.no_grad():p=net(torch.tensor((hh-norm['mean'])/norm['std'],dtype=torch.float32)).numpy()
                nxt=hh[:,-1,:67].copy();nxt[:,:64]+=p[:,:64]*norm['delta_std'];nxt[:,64:]=1/(1+np.exp(-np.clip(p[:,64:67],-30,30)));nxt[:,48:51]/=np.maximum(np.linalg.norm(nxt[:,48:51],axis=1,keepdims=True),1e-6)
                cost+=20*np.maximum(.27-nxt[:,57],0)**2+2*(nxt[:,48]**2+nxt[:,49]**2)+.02*np.mean((nxt[:,:24]-target)**2,1)
                hh=np.concatenate([hh[:,1:],np.concatenate([nxt,cands],-1)[:,None]],1)
            cost+=.002*(1-alphas)**2
            j=int(np.argmin(cost));u=cands[j];chosen_alpha=float(alphas[j])
        else:u=ref_u
        hist.append(np.r_[s,u]);states.append(d.qpos.copy());actions.append(u);chosen.append(chosen_alpha)
        for _ in range(10):actual+=(u-actual)*(1-np.exp(-.002/lag));d.ctrl[:]=motor_envelope(actual,d.qvel[6:]);mj.mj_step(m,d)
        ss=observe(m,d)
        if ss[57]<.18 or ss[50]<np.cos(1.):fall=d.time;break
    name='filtered' if filtered else 'direct';r={'mode':name,'seed':seed,'seconds':d.time,'fall_time':fall,'mean_reference_torque_fraction':float(np.mean(chosen[25:])) if len(chosen)>25 else 0,'base_xy_displacement_m':float(np.linalg.norm(d.qpos[:2])),'note':'Filter may choose standing fallback; survival is not evidence of walking. No control weights trained on these trials.'}
    np.savez_compressed(OUT/f'control_{name}_{seed}.npz',qpos=states,actions=actions,alpha=chosen)
    (OUT/f'control_{name}_{seed}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':
    for seed in [4001,4002,4003]:run(False,seed);run(True,seed)
