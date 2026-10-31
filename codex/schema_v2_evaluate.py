"""Frozen held-out predictions, cross-part effects and sensory-path diagnostics."""
from schema_v2_train import *
from scipy.stats import spearmanr


def freeze_worlds():
    from roll_sit_probe import verify_frozen
    old=verify_frozen()
    paths=['schema_v2_data.py','schema_v2_models.py','schema_v2_train.py','schema_v2_evaluate.py','schema_v2_control.py','schema_v2_composition.py','schema_v2_checks.py','SCHEMA_CURRICULUM_PROTOCOL.md',
           'schema_v2_results/world_curriculum.pt','schema_v2_results/world_mixed.pt']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    f=DEST/'prediction_freeze.json'
    if f.exists():assert json.loads(f.read_text())['sha256']==hashes
    else:f.write_text(json.dumps({'sha256':hashes,'previous_frozen_files_verified':len(old)},indent=2))
    c=np.load(DEST/'presentations_curriculum.npy');m=np.load(DEST/'presentations_mixed.npy')
    assert np.array_equal(np.sort(c.ravel()),np.sort(m.ravel()))


def rmse3(a):return float(np.sqrt(np.mean(np.sum(np.square(a),axis=-1))))


def predict(model,data,ids,intervention=None):
    collected={k:[] for k in ['bodycom','com','cv','sensory','cost','part_attention','sensory_attention']}
    with torch.no_grad():
        for batch in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
            b=batch_data(data,batch);out=model.imagine(b['raw'][:,0],b['q'][:,0],b['actions'],intervention)
            for key in collected:collected[key].append(torch.stack([o[key] for o in out],1).numpy())
    return {k:np.concatenate(v) for k,v in collected.items()}


def metrics(a,data,ids,pred):
    truth=data['bodycom'][ids,1:].numpy();raw=a['raw'][ids,1:];q=a['q'][ids,1:]
    com=q[:,:,:3]+raw[:,:,388:391];v=raw[:,:,391:394];body0=data['bodycom'][ids,0].numpy()
    q0=a['q'][ids,0];raw0=a['raw'][ids,0];com0=q0[:,:3]+raw0[:,388:391];v0=raw0[:,391:394]
    force0=raw0[:,:375].reshape(-1,25,15)[:,:,7:10].sum(1);mass=float(GoalWorld().m.body_mass.sum())
    acceleration=force0/mass+np.array([0,0,-9.81]);horizons=[]
    for step in range(HORIZON):
        t=(step+1)*DURATION;constant=com0+v0*t+.5*acceleration*t*t
        horizons.append({'seconds':t,'body_point_rmse_mm':rmse3(pred['bodycom'][:,step]-truth[:,step])*1000,
            'stationary_body_point_rmse_mm':rmse3(body0-truth[:,step])*1000,'whole_com_rmse_mm':rmse3(pred['com'][:,step]-com[:,step])*1000,
            'stationary_whole_com_rmse_mm':rmse3(com0-com[:,step])*1000,'constant_current_force_com_rmse_mm':rmse3(constant-com[:,step])*1000,
            'com_velocity_rmse_m_s':rmse3(pred['cv'][:,step]-v[:,step])})
    cost=a['cost'][ids,:,0].sum(1);pcost=pred['cost'][:,:,0].sum(1)
    sensory=a['sensory'][ids];force=10*np.sinh(sensory[:,:,:,3:6]);pforce=10*np.sinh(pred['sensory'][:,:,:,3:6]);active=force[:,:,:,2]>1
    base_ids=a['baseline'][ids];local={int(i):j for j,i in enumerate(ids)};effects=[];zero_effects=[];com_effect=[]
    for j,idx in enumerate(ids):
        if a['group'][idx]==7:continue
        baseline=local[int(base_ids[j])];parts=GROUP_PARTS[a['group'][idx]]
        active_nodes=[joint+1 for part in parts for joint in PART_JOINTS[part]]
        inactive=np.array([node for node in range(25) if node not in active_nodes])
        observed=truth[j,-1]-truth[baseline,-1];imagined=pred['bodycom'][j,-1]-pred['bodycom'][baseline,-1]
        effects.append((imagined-observed)[inactive]);zero_effects.append(observed[inactive])
        com_effect.append((pred['com'][j,-1]-pred['com'][baseline,-1])-(com[j,-1]-com[baseline,-1]))
    by_group={}
    for group,name in enumerate(GROUPS):
        mask=a['group'][ids]==group
        if mask.any():by_group[name]={'sequences':int(mask.sum()),'body_point_3step_rmse_mm':rmse3(pred['bodycom'][mask,-1]-truth[mask,-1])*1000,'whole_com_3step_rmse_mm':rmse3(pred['com'][mask,-1]-com[mask,-1])*1000}
    modes={'suspended':[],'supported':[],'switching':[]}
    for j,idx in enumerate(ids):
        parts=GROUP_PARTS[a['group'][idx]]
        if not parts:continue
        contact=a['modes'][idx,0,parts]
        name='suspended' if np.all(contact<.01) else ('supported' if np.any(contact>.5) else 'switching')
        modes[name].append(j)
    by_mode={name:{'sequences':len(ix),'one_step_com_rmse_mm':rmse3(pred['com'][ix,0]-com[ix,0])*1000 if len(ix) else None} for name,ix in modes.items()}
    return {'sequences':len(ids),'independent_initial_cases':int(len(np.unique(a['case'][ids]))),'horizons':horizons,'by_group':by_group,'by_initial_action_contact_mode':by_mode,
            'effort_total_rmse':float(np.sqrt(np.mean((cost-pcost)**2))),'effort_spearman':float(spearmanr(cost,pcost).statistic),
            'torque_rms_normalized_rmse':float(np.sqrt(np.mean((pred['sensory'][:,:,1:,1]-sensory[:,:,1:,1])**2))),
            'active_contact_fz_rmse_N':float(np.sqrt(np.mean((pforce[:,:,:,2][active]-force[:,:,:,2][active])**2))),
            'inactive_body_action_effect_rmse_mm':rmse3(np.concatenate(effects))*1000,'zero_effect_baseline_rmse_mm':rmse3(np.concatenate(zero_effects))*1000,
            'whole_com_action_effect_rmse_mm':rmse3(np.array(com_effect))*1000}


def run():
    torch.set_num_threads(2);freeze_worlds();a,data,_,_=dataset();ids=np.where(a['split']==2)[0];results={}
    for label in ['curriculum','mixed']:
        model=load_model(label);pred=predict(model,data,ids);report=metrics(a,data,ids,pred)
        np.savez_compressed(DEST/f'prediction_{label}.npz',ids=ids,**pred)
        if label=='curriculum':
            interventions={}
            for name in ['zero','swap']:
                altered=predict(model,data,ids,name)
                truth=data['bodycom'][ids,1:].numpy()
                interventions[name]={'body_point_3step_rmse_mm':rmse3(altered['bodycom'][:,-1]-truth[:,-1])*1000,
                                    'mean_prediction_change_mm':float(np.linalg.norm(altered['bodycom'][:,-1]-pred['bodycom'][:,-1],axis=-1).mean()*1000)}
            report['sensory_path_interventions']=interventions
            report['intervention_note']='Inference-time diagnostics, not separately trained causal ablations; analytic COM impulse path remains active.'
        results[label]=report
        print(label,json.dumps({k:v for k,v in report.items() if k not in ['by_group','by_initial_action_contact_mode']},indent=2),flush=True)
    (DEST/'prediction_report.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':run()
