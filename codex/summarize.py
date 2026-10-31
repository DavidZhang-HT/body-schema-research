from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
modes=['direct','projected','adapted'];colors=['#e76f51','#e9c46a','#2a9d8f'];summary=[]
fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True)
for mode,color in zip(modes,colors):
 a=np.load(ROOT/f'results/{mode}.npz');log=a['log'];r=json.loads((ROOT/f'results/{mode}.json').read_text());summary.append(r)
 for ax,col,label in zip(axs,[3,4,7],['Base height (m)','Torso tilt (rad)','Vertical ground force (N)']):
  ax.plot(log[:,0],log[:,col],label=mode,color=color);ax.set_ylabel(label);ax.grid(alpha=.2);ax.axvline(6,color='gray',ls=':');ax.axvline(8,color='gray',ls=':')
 np.savetxt(ROOT/f'results/{mode}_log.csv',log,delimiter=',',header='time,x,y,z,tilt,joint_rmse,peak_torque,vertical_grf,cop_x,cop_y,planned_left_support,planned_right_support,max_left_leg_speed',comments='')
axs[0].axhline(.18,ls='--',color='red',alpha=.5,label='fall height threshold');axs[0].legend(loc='lower left');axs[-1].set_xlabel('Simulation time (s); 0-6 calibration, 6-8 transition, >8 reference')
fig.suptitle('Pi Plus: all three variants fail near first support transfer\nSingle deterministic trial per variant; no sustained walking demonstrated');fig.tight_layout();fig.savefig(ROOT/'results/comparison.png',dpi=160)
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
files=['GMR/assets/pi_plus_24dof_250826/urdf/pi_plus_24dof_250826.urdf','GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_walk1_subject1.csv']
provenance={'repository':'https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic','commit':'df3fcf16bf2cf645b8e3c24ee1e395d453f78f80','source_fps':30,'source_frames_used':'[30:390] (zero based, end exclusive)','files':{f:hashlib.sha256((ROOT/'upstream'/f).read_bytes()).hexdigest() for f in files}}
(ROOT/'results/provenance.json').write_text(json.dumps(provenance,indent=2))
print([(r['mode'],r['fall_time']) for r in summary])
