import os
os.environ['MPLCONFIGDIR']='/tmp/ground_plot_cache'
from ground_schema import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=GroundWorld();fig,axs=plt.subplots(3,2,figsize=(12,9))
for col,pose in enumerate(['prone','supine']):
 a=np.load(OUT/f'getup_replay_{pose}.npz')['trace'];vals=[];loads=[];t=[]
 for row in a[::5]:
  w.restore(row);r=w.metrics();f,p,weight,load=w.contacts();t.append(w.d.time-a[0,-1]);vals.append([r['com_height'],r['base_height'],r['up_z']]);loads.append([weight[[6,12]].sum(),weight[[16,17,21,22]].sum(),weight.sum()-weight[[6,12,16,17,21,22]].sum(),load[9],load[12]])
 t=np.array(t);v=np.array(vals);f=np.array(loads)
 axs[0,col].plot(t,v[:,0],label='CoM height');axs[0,col].plot(t,v[:,1],label='Pelvis height');axs[0,col].set_title(pose+' -> stand (physics-discovered sequence)');axs[0,col].set_ylabel('Height (m)')
 axs[1,col].plot(t,f[:,0],label='Feet');axs[1,col].plot(t,f[:,1],label='Elbows / wrists');axs[1,col].plot(t,f[:,2],label='Other links');axs[1,col].axhline(108,color='gray',ls=':',label='Approx. body weight');axs[1,col].set_ylabel('Vertical ground force (N)')
 axs[2,col].plot(t,f[:,3],label='Left knee');axs[2,col].plot(t,f[:,4],label='Right hip pitch');axs[2,col].set_ylabel('Ground-induced joint load (Nm)');axs[2,col].set_xlabel('Time from passive equilibrium (s)')
 for ax in axs[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'getup_contact_trace.png',dpi=150)
if (OUT/'test_predictions.npz').exists():
 a=np.load(OUT/'test_predictions.npz');x,y,p=a['x'],a['y'],a['prediction'];fig,axs=plt.subplots(1,2,figsize=(10,4))
 actual=(y[:,394]-x[:,394])*1000;pred=(p[:,394]-x[:,394])*1000;axs[0].scatter(actual,pred,s=9,alpha=.5);lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max());axs[0].plot([lo,hi],[lo,hi],'k--');axs[0].set(xlabel='Observed CoM height change (mm)',ylabel='Predicted change (mm)',title='Held-out passive starts, 0.6 s schema effects')
 nodes=y[:,:375].reshape(-1,25,15);pn=p[:,:375].reshape(-1,25,15);mask=nodes[:,:,13]>.5;axs[1].scatter(nodes[:,:,9][mask],pn[:,:,9][mask],s=7,alpha=.4);axs[1].plot([0,200],[0,200],'k--');axs[1].set(xlabel='Observed active-contact Fz (N)',ylabel='Predicted Fz (N)',title='Contact force prediction; active contacts only');fig.tight_layout();fig.savefig(OUT/'contact_prediction.png',dpi=150)
print('Saved contact/CoM traces and available learned-model diagnostics.')
