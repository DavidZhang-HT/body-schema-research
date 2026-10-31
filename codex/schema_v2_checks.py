"""Physical and leakage checks for internal motion versus external support."""
from schema_v2_models import *
from roll_sit_probe import verify_frozen
import json


def run():
    torch.set_num_threads(1);w=SensedWorld();rng=np.random.default_rng(8500)
    old=verify_frozen();a=dict(np.load(DEST/'exploration.npz'))
    c=np.load(DEST/'presentations_curriculum.npy');m=np.load(DEST/'presentations_mixed.npy')
    assert np.array_equal(np.sort(c.ravel()),np.sort(m.ravel()))
    cases=[set(a['case'][a['split']==i]) for i in [0,1,2]]
    assert not any(cases[i]&cases[j] for i in range(3) for j in range(i))
    assert not np.any(a['group'][a['split']==0]>=8)
    for group in range(10):assert np.all(a['actions'][a['group']==group]*(1-action_mask(group))==0)
    # Independent flight experiment: no ground contact, no root force, only
    # different internal actuator commands from an identical high initial pose.
    mj.mj_resetData(w.m,w.d);w.actual[:]=0;w.d.qpos[:]=standing_pose(w);w.d.qpos[2]=2.
    mj.mj_forward(w.m,w.d);state=w.state();snapshot=w.snapshot();out=[]
    for action in [np.zeros(24),sample_sequence(rng,6,.7)[0]]:
        w.restore(snapshot);w.sensed_step(action);out.append({'com':w.d.subtree_com[w.base].copy(),'points':w.d.xipos[w.bodies].copy(),'ground_force':w.force_only().sum(0)})
    flight_difference=float(np.linalg.norm(out[1]['com']-out[0]['com']))
    internal_change=float(np.sqrt(np.mean(np.sum((out[1]['points']-out[0]['points'])**2,-1))))
    assert all(np.linalg.norm(o['ground_force'])<1e-8 for o in out)
    assert flight_difference<.0003 and internal_change>.001,(flight_difference,internal_change)
    model=load_model();idx=np.where(a['split']==1)[0][:8]
    with torch.no_grad():pred=model.imagine(torch.tensor(a['raw'][idx,0]),torch.tensor(a['q'][idx,0]),torch.tensor(a['actions'][idx]))
    errors=[]
    for o in pred:
        represented=(o['bodycom']*model.fk.mass[None,:,None]).sum(1)/model.fk.mass.sum()
        errors.append(float(abs(represented-o['com']).max()))
        sense=o['sensory']
        assert torch.all(sense[:,:,9]>=0) and torch.all(sense[:,:,12]>=0)
        for sl in [slice(7,10),slice(10,13)]:
            f=sense[:,:,sl];assert torch.all(torch.linalg.vector_norm(f[:,:,:2],dim=-1)<=.8*f[:,:,2]+1e-5)
        assert float(abs(o['cost'][:,0]-DURATION*sense[:,1:,1].square().sum(1)).max())<1e-6
    assert max(errors)<1e-6
    result={'previous_frozen_files_verified':len(old),'matched_training_presentations':int(c.size),'disjoint_initial_case_splits':[len(z) for z in cases],
            'unseen_diagonal_masks_excluded_from_training':True,'command_masks_verified':True,
            'free_flight_internal_motion':{'whole_com_difference_m':flight_difference,'body_point_motion_difference_m':internal_change,'no_external_contact_force':True},
            'max_model_mass_weighted_com_error_m':max(errors),'friction_cone_checks':True,'effort_from_imagined_torque_rms_verified':True,
            'future_sensation_input':'Model API accepts only current observation, current configuration and candidate actions; targets appear only in loss functions.'}
    controls=sorted(DEST.glob('control_effort_*.npz'))
    if controls:
        path=controls[0];d=np.load(path);w=SensedWorld();w.restore(d['initial']);trace=[];costs=[]
        for action in d['actions']:
            _,cost,_,_,_,frames=w.sensed_step(action,trace=True);trace+=frames;costs.append(cost)
        error=float(abs(np.array(trace)[:,:109]-d['trace'][:,:109]).max());assert error<1e-8,error
        assert np.max(abs(np.array(trace)[:,85:109])-w.torque_limit)<1e-8
        result['physical_control_replay']={'file':path.name,'maximum_state_motor_error':error,'maximum_cost_error':float(abs(np.array(costs)-d['costs']).max()),'torque_limits_verified':True}
    (DEST/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


def standing_pose(w):
    from experiment import standing
    return standing(w.m)

if __name__=='__main__':run()
