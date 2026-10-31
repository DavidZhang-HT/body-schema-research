import os
os.environ['MPLCONFIGDIR']='/tmp/pi_plus_mpl'
from zero_identification import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
reports=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_seed*.json'))]
# Exclude generated calibration config files.
reports=[r for r in reports if 'history' in r]
m=model();names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(1,13)]
for r in reports:
 name=f'{r["selection"]}_seed{r["seed"]}';a=np.load(OUT/f'{name}.npz')
 header=['block']+[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(1,25)]+['imu_w','imu_x','imu_y','imu_z']
 np.savetxt(OUT/f'{name}_sensors.csv',a['sensor_packets'],delimiter=',',header=','.join(header),comments='')
 config={'scope':'simulation validation only','convention':'q_corrected = q_encoder - offset_rad','joint_offsets_rad':dict(zip(names,a['estimate'].tolist())),'data_rank':r['observability']['relative_rank_1e3'],'condition':r['observability']['condition'],'aborted':r['aborted']}
 (OUT/f'{name}_calibration.json').write_text(json.dumps(config,indent=2))
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
x=np.arange(len(reports));axs[0].bar(x-.18,[r['before_offset_rmse_deg'] for r in reports],.36,label='Uncorrected');axs[0].bar(x+.18,[r['after_offset_rmse_deg'] for r in reports],.36,label='Identified');axs[0].set_xticks(x,[f'{r["selection"]}\nseed {r["seed"]}' for r in reports]);axs[0].set_ylabel('12-joint offset RMSE (deg)');axs[0].legend();axs[0].grid(axis='y',alpha=.2)
r=next(r for r in reports if r['selection']=='active' and r['seed']==7);x=np.arange(12)
axs[1].plot(x,r['true_offsets_deg'],'o-',label='Injected (evaluation only)');axs[1].plot(x,r['estimated_offsets_deg'],'x--',label='Estimated');axs[1].set_xticks(x,[n.removesuffix('_joint').replace('ankle_','ank_').replace('hip_','hip_') for n in names],rotation=65,ha='right');axs[1].set_ylabel('Joint zero offset (deg)');axs[1].legend();axs[1].grid(alpha=.2)
fig.suptitle('Pi Plus: active zero-offset identification in floating-base physics');fig.tight_layout();fig.savefig(OUT/'zero_comparison.png',dpi=160)
(OUT/'summary.json').write_text(json.dumps(reports,indent=2))
for r in reports:print(r['selection'],r['seed'],r['before_offset_rmse_deg'],r['after_offset_rmse_deg'],r['heldout_before'],r['heldout_after'])
