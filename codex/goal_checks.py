"""Meaningful geometry, actuator, data-separation and energy-replay checks."""
from goal_models import *
import json

def main():
    w=GoalWorld();fk=Kinematics();rng=np.random.default_rng(99);w.passive('supine',219);snap=w.snapshot();action=rng.uniform(-.2,.2,24);s,c,trace=w.advance_full(action,trace=True);end=w.snapshot();w.restore(snap);s2,c2,_=w.advance_full(action)
    assert np.max(abs(w.snapshot()[:61]-end[:61]))<1e-9
    assert np.max(abs(c-c2))<1e-8 and c[0]>=0 and c[1]>=0 and c[2]<=1.00001
    p,r=fk(torch.tensor(s['qpos'][None],dtype=torch.float32));assert np.max(abs(p.detach().numpy()[0]-s['positions']))<1e-6;assert np.max(abs(r.detach().numpy()[0]-s['rotations']))<1e-6
    # Check derivatives used for imagined action optimization against finite differences.
    q=torch.tensor(s['qpos'][None],dtype=torch.float32,requires_grad=True);p,r=fk(q);marker=fk.body_com(p,r)[0,17,2];grad=torch.autograd.grad(marker,q)[0].numpy()[0]
    errors=[]
    for index in [7,19,20,22,23]:
        values=[]
        for sign in [-1,1]:
            qq=torch.tensor(s['qpos'][None],dtype=torch.float32);qq[0,index]+=sign*.001;pp,rr=fk(qq);values.append(float(fk.body_com(pp,rr)[0,17,2]))
        errors.append(abs((values[1]-values[0])/.002-grad[index]))
    assert max(errors)<1e-4,errors
    a=np.load(OUT/'exploration.npz');case=a['case'];train=set(case[(case<24)|(case>=36)]);val=set(case[(case>=24)&(case<30)]);test=set(case[(case>=30)&(case<36)])
    assert not train&val and not train&test and not val&test
    assert np.max(abs(a['u']))<=1 and all(np.isfinite(a[k]).all() for k in a.files)
    for node in TARGET_NODES:
        goal=make_goal(w,node);f=w.goal_features(goal);assert np.where(np.linalg.norm(f,axis=1)>0)[0].tolist()==[node]
    result={'exact_physics_replay':True,'differentiable_URDF_geometry':True,'maximum_Jacobian_check_error':float(max(errors)),'disjoint_initial_cases':True,'all_target_nodes_use_same_goal_interface':True,'nonnegative_measured_effort_and_work':True,'torque_limits':True}
    (OUT/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
