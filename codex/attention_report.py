import os
os.environ['MPLCONFIGDIR']='/tmp/pi_attention_mpl'
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from attention_data import OUT,model
import mujoco as mj
r={k:json.loads((OUT/f'{k}_evaluation.json').read_text()) for k in ['attention','mlp']}
fig,axs=plt.subplots(1,3,figsize=(13,4))
for kind,color in [('attention','#277da8'),('mlp','#e07a5f')]:
 h=json.loads((OUT/f'{kind}_training.json').read_text())['history'];axs[0].plot([x['epoch'] for x in h],[x['validation'] for x in h],label=kind,color=color)
 horizons=[1,5,10,20];ts=np.array(horizons)*.02
 axs[1].plot(ts,[r[kind]['rollout'][str(i)]['joint_angle_rad']*180/np.pi for i in horizons],'o-',label=kind,color=color)
 axs[2].plot(ts,[r[kind]['rollout'][str(i)]['base_height_m']*1000 for i in horizons],'o-',label=kind,color=color)
axs[1].plot(ts,[r['attention']['rollout_baselines'][str(i)]['persistence']['joint_angle_rad']*180/np.pi for i in horizons],'--',label='persistence',color='gray')
axs[2].plot(ts,[r['attention']['rollout_baselines'][str(i)]['persistence']['base_height_m']*1000 for i in horizons],'--',label='persistence',color='gray')
axs[0].set(xlabel='Epoch',ylabel='3-step validation objective');axs[1].set(xlabel='Open-loop prediction horizon (s)',ylabel='Joint angle RMSE (deg)');axs[2].set(xlabel='Open-loop prediction horizon (s)',ylabel='Base height RMSE (mm)')
for ax in axs:ax.legend();ax.grid(alpha=.2)
fig.suptitle('Pi Plus learned dynamics: held-out episodes, autoregressive rollout');fig.tight_layout();fig.savefig(OUT/'prediction_comparison.png',dpi=160)
w=np.load(OUT/'attention_weights.npy');m=model();labels=['base']+[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).replace('_joint','') for j in range(1,25)]
fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(w,cmap='viridis');ax.set_xticks(range(25),labels,rotation=90,fontsize=7);ax.set_yticks(range(25),labels,fontsize=7);ax.set(xlabel='Key node',ylabel='Query node',title='Mean spatial attention: selected held-out episode\nIncludes a topology prior; weights are not causal force explanations');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'attention_map.png',dpi=150)
print('Saved prediction comparison and spatial attention map.')
