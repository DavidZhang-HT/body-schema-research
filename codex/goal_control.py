"""One point/direction controller for every body node, with learned imagination."""
from goal_models import *
import argparse,time,json

class GoalController:
    def __init__(self,mode='efficient',iterations=24,restarts=4):
        torch.set_num_threads(1)
        self.mode=mode;self.predictor=Predictor(mode=='blind');self.fk=self.predictor.model.fk;self.iterations=iterations;self.restarts=restarts
    def action(self,w,goal,seed=0):
        if self.mode=='hold':return np.zeros(24),{}
        state=w.state();x,q=self.predictor.inputs(state['obs'][None],state['qpos'][None]);p0,r0=self.fk(q);features=torch.tensor(w.goal_features(goal)[None])
        with torch.no_grad():
            proposal=self.predictor.actor(x,r0,features) if self.predictor.actor is not None else torch.zeros((1,24))
        if self.mode=='policy':return proposal[0].numpy(),{'method':'single shared goal-conditioned attention policy'}
        count=self.restarts;rng=np.random.default_rng(seed)
        start=np.repeat(proposal.numpy(),count,axis=0) if self.mode!='kinematic' else np.zeros((count,24),np.float32)
        start[0]=0
        if count>2:start[2:]+=rng.normal(0,.16,start[2:].shape)
        z=nn.Parameter(torch.tensor(np.arctanh(np.clip(start,-.95,.95)),dtype=torch.float32));optimizer=torch.optim.Adam([z],lr=.13)
        xx=x.expand(count,-1);qq=q.expand(count,-1);rr=r0.expand(count,-1,-1,-1)
        nodes=torch.tensor(goal.nodes,dtype=torch.long);local=torch.tensor(goal.local_points,dtype=torch.float32);target=torch.tensor(goal.target_positions,dtype=torch.float32);pmask=torch.tensor(goal.position_masks,dtype=torch.float32);axes=torch.tensor(goal.local_axes,dtype=torch.float32);directions=torch.tensor(goal.target_directions,dtype=torch.float32);dweight=torch.tensor(goal.direction_weights,dtype=torch.float32)
        candidates=[];stats=[]
        for iteration in range(self.iterations):
            a=torch.tanh(z)
            if self.mode=='kinematic':
                joints=torch.clamp(qq[:,7:]+a*self.fk.limit,self.fk.qlo,self.fk.qhi);p,r=self.fk.from_parts(qq[:,:3],rr[:,0],joints);effort=torch.zeros(count)
            else:
                out=self.predictor.model(xx,qq,a,rr);p,r=out['p'],out['r'];effort=out['cost'][:,0]
            point=p[:,nodes]+torch.einsum('bgij,gj->bgi',r[:,nodes],local);direction=torch.einsum('bgij,gj->bgi',r[:,nodes],axes)
            distance=torch.sqrt((((point-target)*pmask)**2).sum(-1)+1e-12);d_error=torch.sqrt(((direction-directions)**2).sum(-1)+1e-12)*dweight
            goalcost=(distance/.025).square().sum(1)+(d_error/.25).square().sum(1)
            regularizer=.001*a.abs().sum(1)
            if self.mode=='efficient':
                # Minimize effort within a tight predicted goal neighborhood.
                loss=(torch.relu(distance-.003)/.008).square().sum(1)+(torch.relu(d_error-.025)/.08).square().sum(1)+.15*effort/.03+regularizer
            else:loss=goalcost+regularizer
            feasible=(distance.max(1).values<.006)&(d_error.max(1).values<.052)
            candidates.append(a.detach().numpy());stats.append(torch.stack([goalcost,effort,feasible.float(),distance.max(1).values,d_error.max(1).values],1).detach().numpy())
            optimizer.zero_grad();loss.sum().backward();optimizer.step()
        candidates=np.concatenate(candidates);stats=np.concatenate(stats);feasible=np.where(stats[:,2]>.5)[0]
        if self.mode=='efficient' and len(feasible):chosen=feasible[np.argmin(stats[feasible,1])]
        else:chosen=int(np.argmin(stats[:,0]+.001*np.sum(abs(candidates),axis=1)))
        return candidates[chosen],{'predicted_goal_cost':float(stats[chosen,0]),'predicted_effort':float(stats[chosen,1]),'predicted_position_error_m':float(stats[chosen,3]),'predicted_direction_chord_error':float(stats[chosen,4]),'predicted_feasible':bool(stats[chosen,2])}

def goal_json(goal):return {key:getattr(goal,key).tolist() for key in goal.__dataclass_fields__}

def trial(w,controller,pose,seed,node,kind='lift',steps=10,save=True,goal_factory=None):
    settle=w.passive(pose,seed);initial=w.snapshot();goal=goal_factory(w) if goal_factory is not None else make_goal(w,node,kind,.025 if kind=='lift' else np.deg2rad(15));initial_position=w.point_state(goal.nodes,goal.local_points,goal.local_axes)[0].copy()
    trace=[];actions=[];costs=[];predictions=[];started=time.time()
    hold_target=None
    for step in range(steps):
        if step>=steps-2:
            if hold_target is None:hold_target=w.d.qpos[7:].copy()
            a=np.clip((hold_target-w.d.qpos[7:])/ACTION_LIMIT,-1,1);log={'phase':'common terminal joint hold; no future physics query'}
        else:a,log=controller.action(w,goal,seed=10000+step)
        _,cost,frames=w.advance_full(a,trace=True);trace+=frames;actions.append(a);costs.append(cost);predictions.append(log)
    end=w.snapshot();metrics=w.metrics();task=[]
    for snap in trace:
        w.restore(snap);point,direction,velocity=w.point_state(goal.nodes,goal.local_points,goal.local_axes)
        e=float(np.max(np.linalg.norm((point-goal.target_positions)*goal.position_masks,axis=1)));de=float(np.max(np.arccos(np.clip(np.sum(direction*goal.target_directions,axis=1),-1,1))*goal.direction_weights));speed=float(np.max(np.linalg.norm(velocity,axis=1)))
        task.append([float(w.d.time-initial[-1]),e,de,speed,*point[0]])
    task=np.asarray(task);tail=task[-16:];success=bool(np.all(tail[:,1]<=.008)&np.all(tail[:,2]<=np.deg2rad(5))&np.all(tail[:,3]<=.04));w.restore(end);costs=np.array(costs)
    from attention_data import VELOCITY_LIMIT
    framearray=np.array(trace);qlimit=np.maximum(framearray[:,7:31]-w.m.jnt_range[1:,1],w.m.jnt_range[1:,0]-framearray[:,7:31])
    result={'mode':controller.mode,'pose':pose,'seed':seed,'node':node,'body':w.m.body(w.bodies[node]).name,'kind':kind,'success':success,'settled':settle['settled'],'goal':goal_json(goal),'initial_point':initial_position.tolist(),'steps':steps,'duration_s':steps*DURATION,'final_position_error_mm':1000*float(task[-1,1]),'terminal_max_position_error_mm':1000*float(tail[:,1].max()),'final_direction_error_deg':float(np.rad2deg(task[-1,2])),'terminal_max_direction_error_deg':float(np.rad2deg(tail[:,2].max())),'terminal_max_point_speed_m_s':float(tail[:,3].max()),'effort_integral':float(costs[:,0].sum()),'positive_mechanical_work_J':float(costs[:,1].sum()),'peak_torque_fraction':float(costs[:,2].max()),'max_speed_limit_ratio':float(np.max(abs(framearray[:,37:61])/VELOCITY_LIMIT)),'max_soft_joint_limit_penetration_rad':float(np.maximum(qlimit,0).max()),'final_body_metrics':metrics,'wall_seconds':time.time()-started,'predictions':predictions}
    if save:
        name=f'{controller.mode}_{pose}_{seed}_{node}_{kind}';np.savez_compressed(OUT/f'trial_{name}.npz',initial=initial,trace=trace,actions=actions,costs=costs,task_trace=task);(OUT/f'trial_{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['mode','pose','seed','node','kind','success','final_position_error_mm','final_direction_error_deg','effort_integral','wall_seconds']}),flush=True)
    return result

def suite(mode,split='validation',kind='lift',nodes=None):
    controller=GoalController(mode);world=GoalWorld();results=[]
    seeds=[212] if split=='validation' else [900,901,902]
    selected=nodes or TARGET_NODES
    for seed in seeds:
        for pose in ['prone','supine']:
            for node in selected:results.append(trial(world,controller,pose,seed,node,kind))
    (OUT/f'control_{split}_{mode}_{kind}.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['hold','kinematic','policy','learned','efficient','blind'],default='efficient');p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--kind',choices=['lift','direction'],default='lift');p.add_argument('--nodes',nargs='+',type=int);a=p.parse_args();suite(a.mode,a.split,a.kind,a.nodes)
