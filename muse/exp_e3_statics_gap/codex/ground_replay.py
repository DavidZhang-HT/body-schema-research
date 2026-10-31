"""Replay a discovered schema sequence continuously, then verify sustained standing."""
from ground_schema import *
from ground_balance import balance

def replay(pose,hybrid=False,seed=101):
 source=f'hybrid_{pose}_{seed}_best.npz' if hybrid else f'oracle_{pose}_best.npz'
 output=f'reproduced_hybrid_{pose}_{seed}' if hybrid else f'getup_replay_{pose}'
 w=GroundWorld();a=np.load(OUT/source);w.restore(a['initial']);trace=[w.snapshot()];log=[w.metrics()]
 for delta in a['actions']:
  _,frames=w.advance(delta,trace=True);trace+=frames;log.append(w.metrics())
 endpoint_error=float(np.max(abs(w.d.qpos-a['final'][:31])))
 success,frames,records=balance(w);trace+=frames
 r={'pose':pose,'source':source,'method':('Attention-proposed, physics-validated exploration' if hybrid else 'Physics-guided schema exploration')+' followed by torque-only balance controller; not learned-model-only control','success':success,'schema_blocks':len(a['actions']),'macro_seconds':DURATION,'replay_endpoint_max_qpos_error':endpoint_error,'macro_history':log,'standing_history':records,'final':w.metrics()}
 np.savez_compressed(OUT/f'{output}.npz',trace=trace,actions=a['actions']);(OUT/f'{output}.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['macro_history','standing_history']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pose',default='prone');p.add_argument('--hybrid',action='store_true');p.add_argument('--seed',type=int,default=101);a=p.parse_args();replay(a.pose,a.hybrid,a.seed)
