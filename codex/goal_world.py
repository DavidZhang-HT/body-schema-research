"""Uniform body-region task interface and full-joint contact exploration."""
from ground_schema import GroundWorld, ROOT, mj, np, R
from attention_data import motor_envelope
from dataclasses import dataclass
from pathlib import Path
import json
OUT=ROOT/'goal_results';OUT.mkdir(exist_ok=True)
DURATION=.32
ACTION_LIMIT=np.r_[np.full(12,.24),np.full(10,.32),np.full(2,.35)]
TARGET_NODES=[0,24,17,22,6,12]

@dataclass
class Goal:
    nodes: np.ndarray
    local_points: np.ndarray
    target_positions: np.ndarray
    position_masks: np.ndarray
    local_axes: np.ndarray
    target_directions: np.ndarray
    direction_weights: np.ndarray

class GoalWorld(GroundWorld):
    def __init__(self):
        super().__init__()
        self.points=self.m.body_ipos[self.bodies].copy()
        self.torque_limit=self.m.actuator_ctrlrange[:,1].copy()
        self.parents=np.array([self.body2node.get(int(self.m.body_parentid[b]),-1) for b in self.bodies])
        self.ancestors=[]
        for b in range(25):
            ids=[];p=b
            while p>0:ids.append(p-1);p=self.parents[p]
            self.ancestors.append(ids)
    def state(self):
        obs=super().observe()
        return {'obs':obs,'qpos':self.d.qpos.copy(),'qvel':self.d.qvel.copy(),
                'positions':self.d.xpos[self.bodies].copy(),
                'rotations':self.d.xmat[self.bodies].reshape(25,3,3).copy()}
    def point_state(self,nodes,local_points,local_axes):
        rot=self.d.xmat[np.asarray(self.bodies)[nodes]].reshape(-1,3,3)
        pos=self.d.xpos[np.asarray(self.bodies)[nodes]]+np.einsum('bij,bj->bi',rot,local_points)
        dirs=np.einsum('bij,bj->bi',rot,local_axes)
        velocities=[]
        for i,n in enumerate(nodes):
            jp=np.zeros((3,self.m.nv));jr=jp.copy();mj.mj_jac(self.m,self.d,jp,jr,pos[i],self.bodies[n]);velocities.append(jp@self.d.qvel)
        return pos,dirs,np.array(velocities)
    def goal_features(self,goal):
        p,d,_=self.point_state(goal.nodes,goal.local_points,goal.local_axes)
        features=np.zeros((25,16),np.float32)
        for i,n in enumerate(goal.nodes):
            features[n]=np.r_[goal.local_points[i]/.1,(goal.target_positions[i]-p[i])/.05,goal.position_masks[i],goal.local_axes[i],goal.target_directions[i]-d[i],goal.direction_weights[i]]
        return features
    def advance_full(self,action,trace=False,duration=DURATION):
        """Action is normalized to [-1,1]; all motors share the same controller rule."""
        start=self.d.qpos[7:].copy();target=np.clip(start+np.clip(action,-1,1)*ACTION_LIMIT,self.qlo,self.qhi)
        count=round(duration/self.m.opt.timestep);command=np.zeros(24);rows=[];effort=0.;work=0.;peak=0.
        for k in range(count):
            if k%5==0:
                t=min(1,k*self.m.opt.timestep/(.75*duration));blend=t*t*(3-2*t)
                qt=(1-blend)*start+blend*target
                command=np.clip(self.kp*(qt-self.d.qpos[7:])-self.kd*self.d.qvel[6:],-self.torque_limit,self.torque_limit)
            self.actual+=(command-self.actual)*(1-np.exp(-self.m.opt.timestep/.015))
            self.d.ctrl[:]=motor_envelope(self.actual,self.d.qvel[6:]);tau=self.d.ctrl.copy();velocity=self.d.qvel[6:].copy()
            effort+=float(np.sum((tau/self.torque_limit)**2))*self.m.opt.timestep
            work+=float(np.maximum(tau*velocity,0).sum())*self.m.opt.timestep
            peak=max(peak,float(np.max(abs(tau)/self.torque_limit)))
            mj.mj_step(self.m,self.d)
            if trace and k%10==0:rows.append(self.snapshot())
        return self.state(),np.array([effort,work,peak],np.float32),rows
    def sample_action(self,rng):
        a=np.zeros(24);r=rng.random()
        if r<.12:return a
        if r<.67:
            node=int(rng.integers(1,25));ids=self.ancestors[node]
            if rng.random()<.45:ids=[int(rng.choice(ids))]
            a[ids]=rng.normal(0,.55,len(ids))
        else:
            a=rng.normal(0,.35,24);a[rng.random(24)<.35]=0
        return np.clip(a,-1,1)

def make_goal(w,node,kind='lift',amount=.025):
    point=w.points[node].copy();axes=np.eye(3);rot=w.d.xmat[w.bodies[node]].reshape(3,3)
    # Generic axis selection, identical for every body region.
    axis=axes[np.argmin(abs(rot[2]))];p,d,_=w.point_state(np.array([node]),point[None],axis[None]);target=p.copy();mask=np.zeros((1,3));direction=d.copy();dw=np.zeros(1)
    if kind=='lift':target[:,2]+=amount;mask[:,2]=1
    elif kind=='direction':
        up=np.array([0.,0.,1.]);axis_world=np.cross(d[0],up);axis_world/=max(np.linalg.norm(axis_world),1e-8)
        direction[0]=R.from_rotvec(axis_world*amount).apply(d[0]);dw[:]=1
    else:raise ValueError(kind)
    return Goal(np.array([node]),point[None],target,mask,axis[None],direction,dw)
