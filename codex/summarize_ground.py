"""Collect reproducible experiment outcomes without converting failures into successes."""
from ground_schema import *
from attention_data import VELOCITY_LIMIT
import hashlib

def main():
 w=GroundWorld();results=[]
 for pose in ['prone','supine']:
  r=json.loads((OUT/f'getup_replay_{pose}.json').read_text());a=np.load(OUT/f'getup_replay_{pose}.npz')['trace'];q=a[:,7:31]
  results.append({'method':'physics_search_then_QP','pose':pose,'seed':0,'success':r['success'],'macro_blocks':r['schema_blocks'],'macro_duration_s':DURATION,'continuous_duration_s':DURATION*r['schema_blocks']+5,'final':r['standing_history'][-1],'max_speed_to_URDF_limit_ratio':float(np.max(abs(a[:,37:61])/VELOCITY_LIMIT)),'max_soft_joint_limit_penetration_rad':float(np.maximum(0,np.maximum(q-w.m.jnt_range[1:,1],w.m.jnt_range[1:,0]-q)).max()),'peak_applied_motor_torque_Nm':float(np.max(abs(a[:,85:109])))})
 for mode in ['learned','memory','hybrid']:
  for p in sorted(OUT.glob(mode+'_*.json')):
   r=json.loads(p.read_text())
   if 'success' not in r:continue
   row={'method':mode,'pose':r['pose'],'seed':r['seed'],'success':r['success'],'final':r['final'],**({k:r[k] for k in ['physics_evaluations','search_wall_seconds','continuous_replay_max_state_error'] if k in r})}
   if mode=='hybrid':
    frames=np.load(OUT/f"hybrid_{r['pose']}_{r['seed']}_replay.npz")['trace'];q=frames[:,7:31]
    row['max_speed_to_URDF_limit_ratio']=float(np.max(abs(frames[:,37:61])/VELOCITY_LIMIT))
    row['max_soft_joint_limit_penetration_rad']=float(np.maximum(0,np.maximum(q-w.m.jnt_range[1:,1],w.m.jnt_range[1:,0]-q)).max())
   results.append(row)
 sources=['ground_schema.py','ground_model.py','ground_balance.py','ground_search.py','ground_hybrid.py','build_model.py','experiment.py','upstream/GMR/assets/pi_plus_24dof_250826/urdf/pi_plus_24dof_250826.urdf']
 files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
 r={'results':results,'fixed_sequence_transfer':json.loads((OUT/'transfer_results.json').read_text()),'prediction':json.loads((OUT/'evaluation.json').read_text()),'source_sha256':files,'source_repository':'https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic','source_commit':'df3fcf16bf2cf645b8e3c24ee1e395d453f78f80','physical_scope':'Nominal URDF, flat ground, assumed friction 0.8 and motor lag 15ms, ideal simulated state/tactile proxies. Floating base unassisted, applied motor torques bounded. Torque-speed envelope is not a hard velocity constraint; passive/backdriven speeds can exceed URDF velocity numbers.'}
 (OUT/'summary.json').write_text(json.dumps(r,indent=2));print('Saved',len(results),'experiment outcomes and source hashes.')
if __name__=='__main__':main()
