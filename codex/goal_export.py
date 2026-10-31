"""Export every trial and verify that frozen sources/checkpoints were unchanged."""
from goal_world import *
import csv,hashlib,sys

def export():
    paths=sorted(OUT.glob('control_test_*.json'));rows=[]
    for path in paths:
        for r in json.loads(path.read_text()):
            fields=['mode','pose','seed','node','body','kind','success','duration_s','final_position_error_mm','terminal_max_position_error_mm','final_direction_error_deg','terminal_max_direction_error_deg','terminal_max_point_speed_m_s','effort_integral','positive_mechanical_work_J','peak_torque_fraction','max_speed_limit_ratio','max_soft_joint_limit_penetration_rad','wall_seconds']
            row={k:r[k] for k in fields};row['target_failed']=r['terminal_max_position_error_mm']>8 or r['terminal_max_direction_error_deg']>5;row['settling_failed']=r['terminal_max_point_speed_m_s']>.04;rows.append(row)
    if rows:
        with (OUT/'all_test_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    freeze=json.loads((OUT/'test_freeze.json').read_text());mismatch=[]
    for path,expected in freeze['source_and_checkpoint_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:mismatch.append(path)
    assert not mismatch,mismatch
    checkpoints={}
    import torch
    for path in sorted(OUT.glob('*.pt')):
        ck=torch.load(path,weights_only=True);checkpoints[path.name]={'best_epoch':ck['epoch'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    r={'completed_test_trials':len(rows),'source_and_checkpoint_freeze_verified':True,'checkpoints':checkpoints,'python':sys.version,'robot_source':'https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic','robot_commit':'df3fcf16bf2cf645b8e3c24ee1e395d453f78f80','urdf_sha256':hashlib.sha256((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826/urdf/pi_plus_24dof_250826.urdf').read_bytes()).hexdigest(),'physical_scope':'Nominal URDF and flat ground, ideal state/contact observations, no external base force. Torque-bounded motor lag, speed-dependent drive envelope, soft joint limits. Goal planning never calls future MuJoCo rollouts.'}
    (OUT/'reproducibility.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':export()
