"""Learned-model CEM: desired CoM/body/support state -> schema actions -> physics."""
from ground_model import *
from ground_balance import balance

def value(s,unc):
 n=s[:,:375].reshape(-1,25,15);g=s[:,375:];fz=n[:,:,9]*np.clip(n[:,:,13],0,1);total=fz.sum(1);foot=fz[:,6]+fz[:,12]
 com=g[:,19];base=g[:,12];up=np.clip(g[:,22],-1,1);vel=np.linalg.norm(g[:,16:19],axis=1);footheight=n[:,[6,12],14].mean(1)
 cop=(n[:,:,10:12]*fz[:,:,None]).sum(1)/np.maximum(total[:,None],1);margin=np.maximum(np.linalg.norm(g[:,13:15]-cop,axis=1)-.03,0)
 return 4*np.minimum(com,.34)+6*np.minimum(base,.37)+1.2*up+.6*np.clip(foot/108,0,1)-2*np.maximum(footheight-com,0)-.8*margin-.3*vel-.25*np.abs(total-108)/108-4*unc[:,394]-6*unc[:,387]-unc[:,-1]

def run(pose='prone',seed=101,steps=20):
 w=GroundWorld();settle=w.passive(pose,seed);initial=w.snapshot();net=Ensemble();rng=np.random.default_rng(seed);horizon=3;mean=np.zeros((horizon,12));actions=[];trace=[];log=[];success=False
 for step in range(steps):
  state=w.observe();sigma=np.tile(BOUNDS*.65,(horizon,1))
  for iteration in range(4):
   seq=np.clip(mean+rng.normal(size=(192,horizon,12))*sigma,-BOUNDS,BOUNDS);seq[0]=0
   s=np.repeat(state[None],len(seq),0);cost=np.zeros(len(seq))
   for h in range(horizon):
    s,unc=net.predict(s,seq[:,h]);cost+=(1 if h==horizon-1 else .15)*value(s,unc)
   cost-=.01*np.sum((seq/BOUNDS)**2,axis=(1,2))
   elite=np.argsort(cost)[-24:];mean=seq[elite].mean(0);sigma=np.maximum(seq[elite].std(0),BOUNDS*.08)
  action=seq[np.argmax(cost),0];prediction,pred_unc=net.predict(state[None],action[None]);after,frames=w.advance(action,trace=True);trace+=frames;actions.append(action);r=w.metrics()
  log.append({'step':step,'predicted_com_height':float(prediction[0,394]),'predicted_base_height':float(prediction[0,387]),'predicted_com_std':float(pred_unc[0,394]),**r})
  print(pose,'learned step',step+1,log[-1],flush=True)
  mean=np.concatenate([mean[1:],np.zeros((1,12))])
  if r['base_height']>.14 and r['up_z']>.85 and r['foot_load']>70 and r['nonfoot_load']<15:
   success,frames,balance_records=balance(w);trace+=frames;log.append({'handover_success':success,'final_balance':balance_records[-1]});break
 np.savez_compressed(OUT/f'learned_{pose}_{seed}.npz',initial=initial,actions=actions,trace=trace)
 result={'pose':pose,'seed':seed,'passively_settled':settle['settled'],'success':success,'planner':'3-member contact-attention ensemble, CEM, 3x0.6s horizon; no physics oracle during selection','log':log,'final':w.metrics()}
 (OUT/f'learned_{pose}_{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps({'pose':pose,'success':success,'final':w.metrics()}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pose',default='prone');p.add_argument('--seed',type=int,default=101);p.add_argument('--steps',type=int,default=20);a=p.parse_args();run(a.pose,a.seed,a.steps)
