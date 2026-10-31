"""Re-execute every selected control trajectory and audit imagined velocity state."""
from schema_v2_evaluate import *


def run():
    torch.set_num_threads(1);freeze_worlds();results=json.loads((DEST/'control_report.json').read_text())
    w=SensedWorld();checks=[];sensations=[];names=[]
    for report in results['trials']:
        stem=f"control_{report['mode']}_{report['pose']}_{report['seed']}_{report['node']}";a=np.load(DEST/f'{stem}.npz')
        w.restore(a['initial']);frames=[];costs=[];sense=[]
        for action in a['actions']:
            _,c,s,_,_,f=w.sensed_step(action,trace=True);frames.extend(f);costs.append(c);sense.append(s)
        endpoint=w.snapshot();frames=np.array(frames);error=float(np.max(abs(frames[:,:109]-a['trace'][:,:109])))
        assert error<1e-8,(stem,error)
        assert np.max(abs(frames[:,85:109])-w.torque_limit)<1e-8
        goal=Goal(**{k:np.array(v) for k,v in report['goal'].items()});tail=[]
        for row in frames[-16:]:
            w.restore(row);p,_,v=w.point_state(goal.nodes,goal.local_points,goal.local_axes)
            tail.append(np.linalg.norm((p[0]-goal.target_positions[0])*goal.position_masks[0])<=.008 and np.linalg.norm(v[0])<=.04)
        assert bool(all(tail))==report['success']
        checks.append({'trial':stem,'maximum_state_motor_error':error,'maximum_cost_error':float(abs(np.array(costs)-a['costs']).max()),'success_recomputed':bool(all(tail))})
        sensations.append(sense);names.append(stem)
    np.savez_compressed(DEST/'control_sensations.npz',sensory=sensations,names=names)
    # Newton's CoM path is constrained; generalized endpoint velocities were
    # learned separately. Quantify their remaining consistency error explicitly.
    data=dict(np.load(DEST/'exploration.npz'));ids=np.where(data['split']==2)[0];model=load_model();velocity_errors=[]
    with torch.no_grad():
        for batch in np.array_split(ids,6):
            outputs=model.imagine(torch.tensor(data['raw'][batch,0]),torch.tensor(data['q'][batch,0]),torch.tensor(data['actions'][batch]))
            for out in outputs:
                q=out['q'].numpy();raw=out['raw'].numpy();cv=out['cv'].numpy()
                for i in range(len(batch)):
                    mj.mj_resetData(w.m,w.d);w.d.qpos[:]=q[i];w.d.qvel[:6]=raw[i,381:387];w.d.qvel[6:]=raw[i,:375].reshape(25,15)[1:,1]
                    mj.mj_forward(w.m,w.d);mj.mj_subtreeVel(w.m,w.d)
                    velocity_errors.append(w.d.subtree_linvel[w.base]-cv[i])
    report={'trajectories_replayed':len(checks),'maximum_state_motor_error':max(c['maximum_state_motor_error'] for c in checks),
            'all_success_decisions_recomputed':True,'all_torque_limits_verified':True,'checks':checks,
            'predicted_generalized_velocity_vs_impulse_com_velocity_rmse_m_s':rmse3(np.array(velocity_errors)),
            'velocity_audit_note':'Geometry and Newton CoM integration are constrained, but root/joint velocity heads are not yet projected onto the same centroidal velocity. This limits multi-step physical consistency.'}
    (DEST/'replay_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':run()
