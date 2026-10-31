"""Explore-memory-guided neural action selection; never queries future physics."""
from ground_model import *
from ground_balance import balance

def run(pose='prone',seed=101):
 w=GroundWorld();source=np.load(OUT/f'oracle_{pose}_best.npz');w.restore(source['initial']);references=[w.observe()]
 # Build once from the already discovered experience, not the test initial state.
 for action in source['actions']:w.advance(action);references.append(w.observe())
 refs=np.array(references);w.passive(pose,seed);initial=w.snapshot();net=Ensemble();rng=np.random.default_rng(801);pinv=np.linalg.pinv(B)
 trace=[];log=[];chosen=[]
 for idx,stored in enumerate(source['actions']):
  state=w.observe();goal=refs[idx+1];q=state[:375].reshape(25,15)[1:,0];qgoal=goal[:375].reshape(25,15)[1:,0]
  correction=np.clip(pinv@(qgoal-q),-BOUNDS,BOUNDS)
  candidates=np.clip(correction+rng.normal(0,.07,(64,12))*BOUNDS,-BOUNDS,BOUNDS);candidates[0]=stored;candidates[1]=correction
  p,unc=net.predict(np.repeat(state[None],64,0),candidates)
  cost=150*(p[:,394]-goal[394])**2+150*(p[:,387]-goal[387])**2+2*np.sum((p[:,-3:]-goal[-3:])**2,1)
  cost+=.05*np.mean((p[:,:375].reshape(-1,25,15)[:,1:,0]-qgoal)**2,1)+5*unc[:,394]
  # Trust-region term keeps actions near the demonstrated reachable neighborhood.
  cost+=.1*np.mean(((candidates-correction)/BOUNDS)**2,1)
  j=int(np.argmin(cost));after,frames=w.advance(candidates[j],trace=True);trace+=frames;chosen.append(candidates[j]);log.append({'step':idx,'candidate':j,'predicted_com_height':float(p[j,394]),**w.metrics()})
 success,frames,history=balance(w);trace+=frames
 result={'pose':pose,'seed':seed,'success':success,'method':'Learned contact model selects local corrections around a physics-explored successful schema memory; not model-only discovery','log':log,'final':w.metrics()}
 (OUT/f'memory_{pose}_{seed}.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/f'memory_{pose}_{seed}.npz',initial=initial,trace=trace,actions=chosen);print(json.dumps({'pose':pose,'seed':seed,'success':success,'final':w.metrics()}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pose',default='prone');p.add_argument('--seed',type=int,default=101);a=p.parse_args();run(a.pose,a.seed)
