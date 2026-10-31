"""Frozen learned body model, explicit roll/sit task goals, and causal arrival holding."""
from goal_control import GoalController,goal_json
from goal_challenges import challenge_goal
from goal_world import *
import argparse,time,hashlib
OUT2=ROOT/'roll_sit_results';OUT2.mkdir(exist_ok=True)
SEAT_NODES=[0,1,2,3,7,8,9]

def verify_frozen():
    record=json.loads((OUT/'test_freeze.json').read_text())
    for p,expected in record['source_and_checkpoint_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==expected,p
    return record['source_and_checkpoint_sha256']

def rotation_goal(w,target):
    nodes=np.array([0,0]);points=np.repeat(w.points[0][None],2,0);axes=np.array([[0.,0.,1.],[1.,0.,0.]])
    positions=np.repeat(w.d.xipos[w.base][None],2,0)
    directions=np.stack([target[:,2],target[:,0]])
    return Goal(nodes,points,positions,np.zeros((2,3)),axes,directions,np.ones(2))

def measurements(w,initial_rotation,final_rotation):
    m=w.metrics();current=w.d.xmat[w.base].reshape(3,3);_,_,weight,_=w.contacts();total=float(weight.sum());seat=float(weight[SEAT_NODES].sum());weight_N=float(w.m.body_mass.sum()*9.81)
    full_error=float(np.rad2deg(R.from_matrix(final_rotation@current.T).magnitude()))
    axis=initial_rotation[:,2];front=initial_rotation[:,0];now=current[:,0]-axis*np.dot(current[:,0],axis);now/=max(np.linalg.norm(now),1e-8)
    roll=float(np.rad2deg(np.arctan2(np.dot(axis,np.cross(front,now)),np.dot(front,now))))
    return {**m,'tilt_deg':float(np.rad2deg(np.arccos(np.clip(current[2,2],-1,1)))),'full_rotation_error_deg':full_error,'signed_roll_deg':roll,'angular_speed':float(np.linalg.norm(w.d.qvel[3:6])),'seat_load_N':seat,'seat_fraction':seat/max(total,1),'weight_N':weight_N}

def geometrically_arrived(task,r):
    if task=='roll':return r['full_rotation_error_deg']<=15 and abs(r['up_z'])<=np.sin(np.deg2rad(20)) and r['total_load']>=.6*r['weight_N']
    return r['tilt_deg']<=10 and .07<=r['base_height']<=.20 and r['seat_load_N']>=.4*r['weight_N'] and r['seat_fraction']>=.5 and r['total_load']>=.6*r['weight_N']

def valid(task,r):return geometrically_arrived(task,r) and r['com_speed']<=.04 and r['angular_speed']<=.2

def run_one(task,pose,seed,sign=1):
    w=GoalWorld();settle=w.passive(pose,seed);w.state();initial=w.snapshot();r0=w.d.xmat[w.base].reshape(3,3).copy();final_rotation=r0@R.from_rotvec([0,0,sign*np.pi]).as_matrix() if task=='roll' else np.eye(3)
    controller=GoalController('efficient');stage=1;hold=0;hold_target=None;trace=[];records=[];actions=[];costs=[];decisions=[];streak=0;success=False;start=time.time();t0=w.d.time
    sit_goal=challenge_goal(w,'trunk_upright') if task=='sit' else None
    for block in range(40):
        before=measurements(w,r0,final_rotation)
        if hold==0 and geometrically_arrived(task,before):hold=4;hold_target=w.d.qpos[7:].copy()
        if hold>0:
            action=np.clip((hold_target-w.d.qpos[7:])/ACTION_LIMIT,-1,1);hold-=1;decision={'block':block,'phase':'arrival_hold','remaining':hold}
        else:
            if task=='roll':
                target=r0@R.from_rotvec([0,0,sign*np.deg2rad(30*stage)]).as_matrix();current=w.d.xmat[w.base].reshape(3,3)
                if np.rad2deg(R.from_matrix(target@current.T).magnitude())<=10 and stage<6:
                    stage+=1;target=r0@R.from_rotvec([0,0,sign*np.deg2rad(30*stage)]).as_matrix()
                goal=rotation_goal(w,target)
            else:goal=sit_goal
            action,info=controller.action(w,goal,seed=15000+block);decision={'block':block,'phase':'goal_planning','stage':stage,**info}
        _,cost,frames=w.advance_full(action,trace=True);endpoint=w.snapshot();trace+=frames;actions.append(action);costs.append(cost);decisions.append(decision)
        for frame in frames:
            w.restore(frame);r=measurements(w,r0,final_rotation);r['t']=float(w.d.time-t0);r['valid']=bool(valid(task,r));records.append(r);streak=streak+1 if r['valid'] else 0
        w.restore(endpoint);end=measurements(w,r0,final_rotation)
        if streak>=51 and valid(task,end):success=True;break
    label=f'{task}_{pose}_{seed}_{sign:+d}';last=measurements(w,r0,final_rotation)
    progress=np.unwrap(np.deg2rad([r['signed_roll_deg'] for r in records]))*180/np.pi
    result={'task':task,'pose':pose,'seed':seed,'sign':sign,'success':success,'passively_settled':settle['settled'],'blocks':len(actions),'duration_s':len(actions)*DURATION,'max_intended_roll_deg':float(np.max(sign*progress)),'minimum_final_orientation_error_deg':min(r['full_rotation_error_deg'] for r in records),'minimum_trunk_tilt_deg':min(r['tilt_deg'] for r in records),'geometric_arrival_samples':sum(geometrically_arrived(task,r) for r in records),'hold_blocks':sum(d['phase']=='arrival_hold' for d in decisions),'total_effort':float(np.array(costs)[:,0].sum()),'final':last,'wall_seconds':time.time()-start,'records':records,'decisions':decisions,'initial_rotation':r0.tolist(),'final_rotation':final_rotation.tolist(),'sit_goal':goal_json(sit_goal) if sit_goal is not None else None,'scope':'Frozen attention model + existing optimizer + new task-level waypoint/arrival holding. No physics oracle or standing QP.'}
    np.savez_compressed(OUT2/f'{label}.npz',initial=initial,final=w.snapshot(),trace=trace,actions=actions,costs=costs)
    (OUT2/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['records','decisions','initial_rotation','final_rotation','sit_goal']}),flush=True)
    return result

def run(task):
    verify_frozen();results=[]
    for seed in [910,911,912]:
        for pose in ['supine','prone']:
            for sign in ([1,-1] if task=='roll' else [1]):results.append(run_one(task,pose,seed,sign))
    (OUT2/f'{task}_summary.json').write_text(json.dumps({'task':task,'trials':len(results),'successes':sum(r['success'] for r in results),'results':[{k:r[k] for k in ['pose','seed','sign','success','duration_s','max_intended_roll_deg','minimum_final_orientation_error_deg','minimum_trunk_tilt_deg','hold_blocks','final']} for r in results]},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['roll','sit'],required=True);a=p.parse_args();run(a.task)
