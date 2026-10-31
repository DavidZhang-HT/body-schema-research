"""Independent continuous replays of fixed successful and failed body-goal trials."""
from goal_world import *
import json

def main():
    w=GoalWorld();results=[]
    cases=[('efficient',24),('efficient',17),('efficient',6),('kinematic',24)]
    for mode,node in cases:
        stem=f'trial_{mode}_prone_900_{node}_lift';a=np.load(OUT/f'{stem}.npz');w.restore(a['initial']);trace=[];costs=[]
        for action in a['actions']:
            _,cost,frames=w.advance_full(action,trace=True);trace+=frames;costs.append(cost)
        trace=np.array(trace);err=float(np.max(abs(trace[:,:109]-a['trace'][:,:109])));costerr=float(np.max(abs(np.array(costs)-a['costs'])))
        assert err<1e-8 and costerr<1e-7,(stem,err,costerr)
        assert np.max(abs(trace[:,85:109])-w.torque_limit)<1e-8
        r=json.loads((OUT/f'{stem}.json').read_text());g=Goal(**{k:np.array(v) for k,v in r['goal'].items()});tail=[]
        for row in trace[-16:]:
            w.restore(row);p,d,v=w.point_state(g.nodes,g.local_points,g.local_axes);pe=np.linalg.norm((p-g.target_positions)*g.position_masks,axis=1).max();de=(np.arccos(np.clip((d*g.target_directions).sum(1),-1,1))*g.direction_weights).max();speed=np.linalg.norm(v,axis=1).max();tail.append(pe<=.008 and de<=np.deg2rad(5) and speed<=.04)
        assert bool(all(tail))==r['success']
        results.append({'trial':stem,'success':r['success'],'max_replay_state_and_motor_error':err,'max_measured_cost_error':costerr,'success_recomputed':True,'torque_limits_verified':True})
    (OUT/'replay_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
