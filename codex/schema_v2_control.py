"""Shared body-point proposal and three-step learned-only effort selection."""
from schema_v2_models import *
import hashlib,time


class ImaginingPolicy:
    def __init__(self):
        torch.set_num_threads(1);self.model=load_model();self.actor=GoalProposal()
        self.actor.load_state_dict(torch.load(DEST/'proposal.pt',weights_only=True)['model']);self.actor.eval()
    def choose(self,w,goal,mode='effort',seed=0):
        s=w.state();raw=torch.tensor(s['obs'][None]);q=torch.tensor(s['qpos'][None],dtype=torch.float32)
        r=torch.tensor(s['rotations'][None],dtype=torch.float32);features=torch.tensor(w.goal_features(goal)[None])
        rng=np.random.default_rng(seed)
        with torch.no_grad():proposal=self.actor(self.model.normalized(raw),r,features)[0].numpy()
        if mode=='actor':return proposal[0],{'mode':'direct attention proposal'}
        candidates=[np.zeros_like(proposal),proposal]
        for i in range(62):
            if i<12:
                a=np.clip(proposal+rng.normal(0,.15,proposal.shape),-1,1);a[:,-2:]=0
            else:a=sample_sequence(rng,(i-12)%7,.25+.5*rng.random())
            candidates.append(a)
        candidates=np.array(candidates,np.float32)
        with torch.no_grad():out=self.model.imagine(raw.expand(len(candidates),-1),q.expand(len(candidates),-1),torch.tensor(candidates))
        node=int(goal.nodes[0]);target=goal.target_positions[0];mask=goal.position_masks[0]
        final=out[-1]['bodycom'][:,node].numpy();prior=out[-2]['bodycom'][:,node].numpy()
        error=np.linalg.norm((final-target)*mask,axis=1);speed=np.linalg.norm(final-prior,axis=1)/DURATION
        effort=sum(x['cost'][:,0] for x in out).numpy();feasible=np.where((error<=.008)&(speed<=.04))[0]
        chosen=int(feasible[np.argmin(effort[feasible])]) if mode=='effort' and len(feasible) else int(np.argmin(error+.05*speed))
        sense=out[0]['sensory'][chosen].numpy()
        return candidates[chosen,0],{'mode':mode,'candidate':chosen,'predicted_error_m':float(error[chosen]),'predicted_point_speed':float(speed[chosen]),'predicted_effort':float(effort[chosen]),'predicted_feasible_candidates':len(feasible),
             'imagined_first_step_torque_rms':sense[1:,1].tolist(),'imagined_first_step_ground_impulse_normalized':sense[:,7:10].tolist()}


def run_trial(controller,pose,seed,node,mode):
    w=SensedWorld();settled=w.passive(pose,seed);w.state();initial=w.snapshot()
    goal=make_goal(w,node,'lift',.015);actions=[];traces=[];costs=[];decisions=[];target_hold=None
    for step in range(8):
        if step>=6:
            if target_hold is None:target_hold=w.d.qpos[7:].copy()
            action=np.clip((target_hold-w.d.qpos[7:])/ACTION_LIMIT,-1,1);log={'mode':'common terminal hold'}
        else:action,log=controller.choose(w,goal,mode,seed=17000+seed*10+step)
        _,cost,_,_,_,frames=w.sensed_step(action,trace=True);actions.append(action);traces.extend(frames);costs.append(cost);decisions.append(log)
    final=w.snapshot();tail=[]
    for state in traces[-16:]:
        w.restore(state);p,_,v=w.point_state(goal.nodes,goal.local_points,goal.local_axes)
        tail.append([float(np.linalg.norm((p[0]-goal.target_positions[0])*goal.position_masks[0])),float(np.linalg.norm(v[0]))])
    tail=np.array(tail);w.restore(final)
    result={'pose':pose,'seed':seed,'node':node,'mode':mode,'passively_settled':settled['settled'],
            'success':bool(np.all(tail[:,0]<=.008)&np.all(tail[:,1]<=.04)),
            'final_window_max_error_mm':float(tail[:,0].max()*1000),'final_window_max_speed_m_s':float(tail[:,1].max()),
            'measured_effort':float(np.array(costs)[:,0].sum()),'positive_mechanical_work_J':float(np.array(costs)[:,1].sum()),'duration_s':8*DURATION,
            'goal':{k:getattr(goal,k).tolist() for k in goal.__dataclass_fields__},'decisions':decisions}
    stem=f'control_{mode}_{pose}_{seed}_{node}'
    np.savez_compressed(DEST/f'{stem}.npz',initial=initial,final=final,actions=actions,trace=traces,costs=costs)
    (DEST/f'{stem}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['goal','decisions']}),flush=True)
    return result


def run():
    from schema_v2_evaluate import freeze_worlds
    freeze_worlds()
    paths=['schema_v2_control.py','schema_v2_results/proposal.pt']
    (DEST/'control_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2))
    controller=ImaginingPolicy();results=[]
    for seed in [1200,1201]:
        for pose in ['supine','prone']:
            for node in [17,22,6,12]:
                for mode in ['actor','goal','effort']:results.append(run_trial(controller,pose,seed,node,mode))
    summary={'goal_cases_per_mode':16,'independent_initial_states':4,'modes':{},'trials':results}
    for mode in ['actor','goal','effort']:
        rows=[r for r in results if r['mode']==mode]
        summary['modes'][mode]={'successes':sum(r['success'] for r in rows),'trials':len(rows),'median_final_window_error_mm':float(np.median([r['final_window_max_error_mm'] for r in rows]))}
    pairs=[]
    for goal in [r for r in results if r['mode']=='goal']:
        efficient=next(r for r in results if r['mode']=='effort' and all(r[k]==goal[k] for k in ['pose','seed','node']))
        if goal['success'] and efficient['success']:
            pairs.append({'pose':goal['pose'],'seed':goal['seed'],'node':goal['node'],'goal_effort':goal['measured_effort'],'effort_mode_effort':efficient['measured_effort'],
                          'fraction_reduction':1-efficient['measured_effort']/max(goal['measured_effort'],1e-8)})
    summary['paired_successful_effort_comparisons']=pairs
    (DEST/'control_report.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='trials'},indent=2),flush=True)

if __name__=='__main__':run()
