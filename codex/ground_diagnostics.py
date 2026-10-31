"""Held-out action/force-input diagnostics, recursive predictions and schema probes."""
import os
os.environ['MPLCONFIGDIR']='/tmp/ground_plot_cache'
from ground_model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rmse(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))

def main():
 net=Ensemble();data=np.load(OUT/'exploration.npz');ids=data['case']>=20
 x,u,y=[data[k][ids] for k in ['x','u','y']];p,unc=net.predict(x,u)
 yn=y[:,:375].reshape(-1,25,15);pn=p[:,:375].reshape(-1,25,15);xn=x[:,:375].reshape(-1,25,15);active=yn[:,:,13]>.5
 result={'test_pairs':len(x),'independent_test_initial_cases':4,'active_contact_nodes':int(active.sum()),'active_contact_force_vector_component_rmse_N':rmse((pn[:,:,7:10]-yn[:,:,7:10])[active]),'active_contact_vertical_force_rmse_N':rmse((pn[:,:,9]-yn[:,:,9])[active]),'active_contact_vertical_force_persistence_rmse_N':rmse((xn[:,:,9]-yn[:,:,9])[active]),'contact_point_3d_rmse_mm':1000*rmse(np.linalg.norm((pn[:,:,10:13]-yn[:,:,10:13])[active],axis=1))}
 modes={}
 for name in ['normal','zero_action','shuffled_action','zero_force_inputs']:
  xx=x.copy();uu=u.copy()
  if name=='zero_action':uu[:]=0
  if name=='shuffled_action':uu=uu[np.random.default_rng(4321).permutation(len(uu))]
  if name=='zero_force_inputs':xx[:,np.r_[FORCES,LOADS]]=0
  pp,_=net.predict(xx,uu);modes[name]={'com_height_rmse_mm':1000*rmse(pp[:,394]-y[:,394]),'ground_joint_load_rmse_Nm':rmse(pp[:,LOADS]-y[:,LOADS])}
 result['input_interventions']=modes;result['intervention_note']='Diagnostic only. Removing inputs after training is out of distribution, not a retrained ablation or a causal attention explanation.'
 ep=data['episode'][ids];errors=[[] for _ in range(6)];baseline=[[] for _ in range(6)]
 for e in np.unique(ep):
  ix=np.where(ep==e)[0];pred=x[ix[:1]].copy();start=pred.copy()
  for h,i in enumerate(ix):
   pred,_=net.predict(pred,u[i:i+1]);errors[h].append(float(pred[0,394]-y[i,394]));baseline[h].append(float(start[0,394]-y[i,394]))
 result['recursive_com_prediction']=[{'horizon_s':round((h+1)*DURATION,1),'episodes':len(e),'rmse_mm':1000*rmse(e),'persistence_rmse_mm':1000*rmse(baseline[h])} for h,e in enumerate(errors)]
 probes=[];w=GroundWorld();fig,axes=plt.subplots(2,2,figsize=(14,8))
 for row,pose in enumerate(['prone','supine']):
  w.passive(pose,101);initial=w.snapshot();state=w.observe();actions=np.repeat(np.eye(12),2,axis=0)*np.tile([-.5,.5],12)[:,None]*BOUNDS
  predicted,_=net.predict(np.repeat(state[None],24,0),actions);actual=[]
  for action in actions:
   w.restore(initial);after,_=w.advance(action);actual.append(after)
  actual=np.array(actual);ix=np.arange(24);com_a=1000*(actual[:,394]-state[394]);com_p=1000*(predicted[:,394]-state[394]);pn=predicted[:,:375].reshape(-1,25,15);an=actual[:,:375].reshape(-1,25,15)
  axes[row,0].plot(ix,com_a,'o-',label='Observed');axes[row,0].plot(ix,com_p,'x--',label='Predicted');axes[row,0].set_ylabel(pose+' CoM change (mm)')
  axes[row,1].plot(ix,an[:,[6,12],9].sum(1),'o-',label='Observed feet');axes[row,1].plot(ix,pn[:,[6,12],9].sum(1),'x--',label='Predicted feet');axes[row,1].set_ylabel(pose+' feet vertical force (N)')
  for ax in axes[row]:ax.set_xticks(ix,[str(i+1)+s for i in range(12) for s in ['-','+']],rotation=70);ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Schema index and sign; half amplitude')
  probes.append({'pose':pose,'seed':101,'com_delta_rmse_mm':rmse(com_p-com_a),'schemas':SCHEMAS,'observed_com_delta_mm':com_a.tolist(),'predicted_com_delta_mm':com_p.tolist()})
 fig.suptitle('Independent probes from the same unpowered equilibrium; 0.6 s actions');fig.tight_layout();fig.savefig(OUT/'schema_effects.png',dpi=150)
 result['independent_schema_probes']=probes
 (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='independent_schema_probes'},indent=2))
if __name__=='__main__':main()
