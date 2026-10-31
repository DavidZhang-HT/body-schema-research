"""Independent paired A/B/A+B physics interventions; never used for planning."""
from schema_v2_evaluate import *


def run():
    torch.set_num_threads(1);freeze_worlds();model=load_model();w=SensedWorld();records=[];artifacts={k:[] for k in ['initial','actions','actual_bodycom','predicted_bodycom','actual_com','predicted_com','costs','predicted_costs']}
    pairs=[('arms',0,1),('legs',2,3),('four_limbs',4,5),('diagonal',0,3)]
    for seed in [1200,1201]:
        for pose in ['supine','prone']:
            w.passive(pose,seed);initial=w.snapshot()
            for label,ga,gb in pairs:
                rng=np.random.default_rng(seed*10+ga+(pose=='prone'))
                aa=sample_sequence(rng,ga,.55);ab=sample_sequence(rng,gb,.55)
                actions=np.stack([np.zeros_like(aa),aa,ab,aa+ab]);w.restore(initial);state=w.state()
                with torch.no_grad():
                    predicted=model.imagine(torch.tensor(np.repeat(state['obs'][None],4,0)),torch.tensor(np.repeat(state['qpos'][None],4,0),dtype=torch.float32),torch.tensor(actions,dtype=torch.float32))
                pbody=predicted[-1]['bodycom'].numpy();pc=predicted[-1]['com'].numpy();pcost=sum(o['cost'][:,0] for o in predicted).numpy()
                bodies=[];coms=[];costs=[]
                for sequence in actions:
                    w.restore(initial);cost=0.
                    for action in sequence:state,c,_,_,_,_=w.sensed_step(action);cost+=c[0]
                    bodies.append(w.d.xipos[w.bodies].copy());coms.append(w.d.subtree_com[w.base].copy());costs.append(float(cost))
                bodies=np.array(bodies);coms=np.array(coms)
                observed=bodies[3]-bodies[1]-bodies[2]+bodies[0]
                imagined=pbody[3]-pbody[1]-pbody[2]+pbody[0]
                cm=coms[3]-coms[1]-coms[2]+coms[0];pcm=pc[3]-pc[1]-pc[2]+pc[0]
                record={'seed':seed,'pose':pose,'combination':label,'nonadditive_body_motion_mm':rmse3(observed)*1000,'predicted_interaction_error_mm':rmse3(imagined-observed)*1000,'nonadditive_com_motion_mm':float(np.linalg.norm(cm)*1000),'predicted_com_interaction_error_mm':float(np.linalg.norm(pcm-cm)*1000)}
                records.append(record);print(json.dumps(record),flush=True)
                for k,v in zip(artifacts,[initial,actions,bodies,pbody,coms,pc,costs,pcost]):artifacts[k].append(v)
    arrays={k:np.array(v) for k,v in artifacts.items()};np.savez_compressed(DEST/'composition.npz',**arrays)
    obs=arrays['actual_bodycom'];pred=arrays['predicted_bodycom'];obs=obs[:,3]-obs[:,1]-obs[:,2]+obs[:,0];pred=pred[:,3]-pred[:,1]-pred[:,2]+pred[:,0]
    result={'paired_sets':len(records),'physical_sequences':4*len(records),'independent_initial_states':4,'scope':'Offline counterfactual validation only; physical outcomes do not select policy actions.',
            'interaction_prediction_rmse_mm':rmse3(pred-obs)*1000,'zero_interaction_additive_baseline_rmse_mm':rmse3(obs)*1000,'by_case':records}
    (DEST/'composition_report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='by_case'},indent=2),flush=True)

if __name__=='__main__':run()
