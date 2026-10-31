"""Part-wise exploration, paired baselines and measured impulse/proprioceptive labels."""
from goal_world import *
from attention_data import motor_envelope
import argparse, time

DEST = ROOT / 'schema_v2_results'
DEST.mkdir(exist_ok=True)
GROUPS = ['left_arm','right_arm','left_leg','right_leg','both_arms','both_legs','four_limbs','hold','left_arm_right_leg','right_arm_left_leg']
PART_JOINTS = [list(range(12,17)), list(range(17,22)), list(range(0,6)), list(range(6,12))]
GROUP_PARTS = [[0],[1],[2],[3],[0,1],[2,3],[0,1,2,3],[],[0,3],[1,2]]
GROUP_STAGE = np.array([0,0,0,0,1,2,3,0,4,4])
HORIZON = 3


def action_mask(group):
    mask = np.zeros(24)
    for part in GROUP_PARTS[group]: mask[PART_JOINTS[part]] = 1
    return mask


def sample_sequence(rng, group, amplitude=.6):
    base = rng.normal(0, amplitude, 24)
    result = np.clip(np.stack([base + rng.normal(0, amplitude*.25, 24) for _ in range(HORIZON)]), -1, 1)
    return result * action_mask(group)


class SensedWorld(GoalWorld):
    def force_only(self):
        forces = np.zeros((25,3)); ground = self.m.geom('ground').id
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if ground not in (c.geom1,c.geom2): continue
            other = c.geom2 if c.geom1 == ground else c.geom1
            node = self.body2node.get(int(self.m.geom_bodyid[other]))
            if node is None: continue
            wrench = np.zeros(6); mj.mj_contactForce(self.m,self.d,i,wrench)
            force = c.frame.reshape(3,3).T @ wrench[:3]
            forces[node] += -force if c.geom2 == ground else force
        return forces

    def sensed_step(self, action, trace=False):
        before = self.state(); mass = self.m.body_mass.sum(); gravity = self.m.opt.gravity.copy()
        com0 = self.d.subtree_com[self.base].copy(); vel0 = self.d.subtree_linvel[self.base].copy()
        start = self.d.qpos[7:].copy(); target = np.clip(start+np.clip(action,-1,1)*ACTION_LIMIT,self.qlo,self.qhi)
        dt = self.m.opt.timestep; count = round(DURATION/dt)
        impulse = np.zeros((25,3)); weighted = impulse.copy(); signed = np.zeros(24); squared = signed.copy()
        work=0.; peak=0.; contact_time=np.zeros(4); rows=[]
        for k in range(count):
            if k%5 == 0:
                t=min(1,k*dt/(.75*DURATION)); blend=t*t*(3-2*t)
                command=np.clip(self.kp*((1-blend)*start+blend*target-self.d.qpos[7:])-self.kd*self.d.qvel[6:],-self.torque_limit,self.torque_limit)
            self.actual += (command-self.actual)*(1-np.exp(-dt/.015))
            self.d.ctrl[:]=motor_envelope(self.actual,self.d.qvel[6:]); tau=self.d.ctrl.copy()
            ratio=tau/self.torque_limit; signed+=ratio*dt; squared+=ratio**2*dt
            work+=float(np.maximum(tau*self.d.qvel[6:],0).sum())*dt; peak=max(peak,float(abs(ratio).max()))
            mj.mj_step(self.m,self.d)
            force=self.force_only(); impulse+=force*dt; weighted+=force*dt*(DURATION-(k+.5)*dt)
            for part,ids in enumerate(PART_JOINTS): contact_time[part]+=(force[np.array(ids)+1,2].sum()>5)*dt
            if trace and k%10==0: rows.append(self.snapshot())
        after=self.state(); node=after['obs'][:375].reshape(25,15)
        # 14 sensation channels: mean/RMS torque, reaction load, endpoint force,
        # contact probability, impulse, time-weighted impulse, endpoint torque.
        sensation=np.zeros((25,14),np.float32)
        sensation[1:,0]=signed/DURATION; sensation[1:,1]=np.sqrt(squared/DURATION)
        sensation[:,2]=np.arcsinh(node[:,3]/10); sensation[:,3:6]=np.arcsinh(node[:,7:10]/10)
        sensation[:,6]=node[:,13]; sensation[:,7:10]=impulse/(mass*9.81*DURATION)
        sensation[:,10:13]=weighted/(mass*9.81*DURATION**2/2)
        sensation[1:,13]=node[1:,2]/self.torque_limit
        dv=self.d.subtree_linvel[self.base]-vel0; dc=self.d.subtree_com[self.base]-com0
        residual=np.r_[dv-gravity*DURATION-impulse.sum(0)/mass,dc-vel0*DURATION-gravity*DURATION**2/2-weighted.sum(0)/mass]
        return after,np.array([squared.sum(),work,peak],np.float32),sensation,contact_time/DURATION,residual,rows


def collect(shard=0,shards=3):
    w=SensedWorld(); rows={k:[] for k in ['raw','q','actions','sensory','cost','modes','physics_residual','group','stage','case','anchor','baseline','split','initial']}
    cases=[(0,s,p) for s in range(1100,1106) for p in ['supine','prone']]+[(1,s,p) for s in [1110,1111] for p in ['supine','prone']]+[(2,s,p) for s in [1120,1121] for p in ['supine','prone']]
    started=time.time(); meta=[]
    for case in range(shard,len(cases),shards):
        split,seed,pose=cases[case]; rng=np.random.default_rng(seed*10+(pose=='prone')); settle=w.passive(pose,seed)
        assert settle['settled'],(seed,pose)
        snapshot=w.snapshot(); meta.append({'case':case,'split':split,'seed':seed,'pose':pose})
        for anchor in range(5):
            baseline=len(rows['raw']); successors=[]
            specs=[(7,0.)]+[(g,amp) for g in range(7) for amp in [.35,.75]]
            if split==2: specs += [(g,amp) for g in [8,9] for amp in [.35,.75]]
            for group,amplitude in specs:
                w.restore(snapshot); s=w.state(); initial=w.snapshot(); sequence=sample_sequence(rng,group,amplitude)
                raw=[s['obs']]; q=[s['qpos']]; sens=[]; costs=[]; modes=[]; residuals=[]
                for action in sequence:
                    s,c,se,mo,re,_=w.sensed_step(action);raw.append(s['obs']);q.append(s['qpos']);sens.append(se);costs.append(c);modes.append(mo);residuals.append(re)
                values=[raw,q,sequence,sens,costs,modes,residuals,group,int(GROUP_STAGE[group]),case,anchor,baseline,split,initial]
                for key,value in zip(rows,values):rows[key].append(value)
                if group<7:successors.append(w.snapshot())
            snapshot=successors[int(rng.integers(len(successors)))]
        print('shard',shard,'case',case,'sequences',len(rows['raw']),'seconds',round(time.time()-started),flush=True)
    arrays={k:np.array(v,dtype=np.int32 if k in ['group','stage','case','anchor','baseline','split'] else (np.float64 if k=='initial' else np.float32)) for k,v in rows.items()}
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(DEST/f'data_{shard}.npz',**arrays)
    (DEST/f'data_{shard}.json').write_text(json.dumps({'cases':meta,'sequences':len(rows['raw']),'wall_seconds':time.time()-started},indent=2))


def merge(shards=3):
    arrays=[]; offset=0
    for shard in range(shards):
        a=dict(np.load(DEST/f'data_{shard}.npz')); a['baseline']+=offset; offset+=len(a['group']);arrays.append(a)
    data={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    for split in [0,1,2]:
        ids=data['split']==split
        assert not set(data['case'][ids])&set(data['case'][~ids])
    assert np.all(data['group'][data['baseline']]==7)
    for group in range(10): assert np.all(data['actions'][data['group']==group]*(1-action_mask(group))==0)
    np.savez_compressed(DEST/'exploration.npz',**data)
    coverage={}
    for group in range(10):
        ids=data['group']==group; parts=GROUP_PARTS[group]
        mode=data['modes'][ids][:,:,parts]
        coverage[GROUPS[group]]={'sequences':int(ids.sum()),'active_part_steps':int(mode.size),'suspended_part_steps':int((mode<.01).sum()),'supported_part_steps':int((mode>.5).sum()),'switching_part_steps':int(((mode>=.01)&(mode<=.5)).sum())}
    residual=data['physics_residual']; summary={'sequences':len(data['group']),'transitions':int(len(data['group'])*HORIZON),'split_sequences':{name:int((data['split']==i).sum()) for i,name in enumerate(['train','validation','test'])},'coverage':coverage,'impulse_velocity_residual_rmse_m_s':float(np.sqrt(np.mean(np.sum(residual[:,:,:3]**2,-1)))),'weighted_impulse_position_residual_rmse_m':float(np.sqrt(np.mean(np.sum(residual[:,:,3:]**2,-1)))),'sources':[json.loads((DEST/f'data_{i}.json').read_text()) for i in range(shards)]}
    (DEST/'data_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=3);p.add_argument('--merge',action='store_true');a=p.parse_args()
    merge(a.shards) if a.merge else collect(a.shard,a.shards)
