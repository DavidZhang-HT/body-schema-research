"""Resume incomplete frozen evaluations after executor interruption; no retuning."""
from goal_control import *
import hashlib

def verify_freeze():
    frozen=json.loads((OUT/'test_freeze.json').read_text())
    for path,digest in frozen['source_and_checkpoint_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path

def resume(mode,kind):
    verify_freeze();controller=GoalController(mode);world=GoalWorld();rows=[];reused=0
    for seed in [900,901,902]:
        for pose in ['prone','supine']:
            for node in TARGET_NODES:
                stem=f'trial_{mode}_{pose}_{seed}_{node}_{kind}';path=OUT/f'{stem}.json';arraypath=OUT/f'{stem}.npz'
                if path.exists() and arraypath.exists():
                    old=json.loads(path.read_text());data=np.load(arraypath);assert data['actions'].shape==(10,24) and np.isfinite(data['trace']).all();rows.append(old);reused+=1
                else:rows.append(trial(world,controller,pose,seed,node,kind))
    (OUT/f'control_test_{mode}_{kind}.json').write_text(json.dumps(rows,indent=2));print('completed',mode,kind,'reused',reused,'of',len(rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',required=True);p.add_argument('--kinds',nargs='+',default=['lift','direction']);a=p.parse_args()
    for mode in a.modes:
        for kind in a.kinds:resume(mode,kind)
