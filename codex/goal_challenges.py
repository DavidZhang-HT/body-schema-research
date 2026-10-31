"""Larger and composed goals using exactly the same learned controller, no QP."""
from goal_control import *
from experiment import standing,FOOT

def challenge_goal(w,kind):
    if kind=='pelvis_raise':return make_goal(w,0,'lift',.08)
    nodes=np.array([0] if kind=='trunk_upright' else [0,6,12]);local=w.points[nodes].copy()
    if kind=='stand':local[1:]=FOOT
    axis=np.tile([0.,0.,1.],(len(nodes),1));p,d,_=w.point_state(nodes,local,axis);target=p.copy();mask=np.zeros_like(p);mask[:,2]=1;directions=axis.copy()
    if kind=='trunk_upright':target[0,2]=.16
    else:
        reference=mj.MjData(w.m);reference.qpos[:]=standing(w.m);mj.mj_forward(w.m,reference);target[0,2]=reference.xipos[w.base,2];target[1:,2]=0
    return Goal(nodes,local,target,mask,axis,directions,np.ones(len(nodes)))

def run(mode='efficient',seed=900):
    w=GoalWorld();controller=GoalController(mode);results=[]
    for pose in ['prone','supine']:
        for kind in ['pelvis_raise','trunk_upright','stand']:
            r=trial(w,controller,pose,seed,0,kind,steps=32,goal_factory=lambda world:challenge_goal(world,kind))
            if kind=='stand':
                m=r['final_body_metrics'];r['goal_geometry_success']=r['success'];r['success']=r['success'] and m['foot_load']>80 and m['nonfoot_load']<5
            r['interpretation']='Large-goal challenge outside the small local-goal evaluation; no standing QP or future physics validation. Trunk-upright does not by itself specify a seated pelvis contact.';results.append(r)
    (OUT/f'challenges_{mode}_{seed}.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='efficient');p.add_argument('--seed',type=int,default=900);a=p.parse_args();run(a.mode,a.seed)
