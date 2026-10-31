"""Branched full-joint exploration; no task-specific action rules or demonstrations."""
from goal_world import *
import argparse,time

def collect(shard=0,shards=3,anchors=12,branches=16):
    w=GoalWorld();rng=np.random.default_rng(5200+shard);rows={k:[] for k in ['x','q','u','y','next_q','cost','case','anchor']};meta=[];start=time.time()
    def probe(case,anchor,snapshot):
        successors=[]
        for b in range(branches):
            w.restore(snapshot);before=w.state();a=np.zeros(24) if b==0 else w.sample_action(rng);after,cost,_=w.advance_full(a)
            values=[before['obs'],before['qpos'],a,after['obs'],after['qpos'],cost,case,anchor]
            for k,v in zip(rows,values):rows[k].append(v)
            successors.append(w.snapshot())
        return successors[int(rng.integers(1,len(successors)))]
    for case in range(shard,36,shards):
        pose=['supine','prone'][case%2];seed=200+case//2;settled=w.passive(pose,seed)
        if not settled['settled']:raise RuntimeError((pose,seed,'not settled'))
        snapshot=w.snapshot();meta.append({'case':case,'pose':pose,'seed':seed,'settled':True})
        for anchor in range(anchors):snapshot=probe(case,anchor,snapshot)
        print('shard',shard,'case',case,'pairs',len(rows['x']),'elapsed',round(time.time()-start),flush=True)
    if shard==0:
        # Existing physically achieved support states, used only as training initializations.
        for pose in ['prone','supine']:
            source=ROOT/'ground_results'/f'getup_replay_{pose}.npz';states=np.load(source)['trace']
            for i,ix in enumerate(np.linspace(0,len(states)-1,24,dtype=int)):
                case=36+(0 if pose=='prone' else 24)+i;probe(case,0,states[ix]);meta.append({'case':case,'source':str(source.relative_to(ROOT)),'frame':int(ix),'training_only':True})
    arrays={k:np.asarray(v,dtype=np.int32 if k in ['case','anchor'] else np.float32) for k,v in rows.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(OUT/f'data_shard_{shard}.npz',**arrays)
    (OUT/f'data_shard_{shard}.json').write_text(json.dumps({'shard':shard,'pairs':len(rows['x']),'cases':meta,'wall_seconds':time.time()-start},indent=2))

def merge(shards):
    all_data=[np.load(OUT/f'data_shard_{s}.npz') for s in range(shards)]
    a={k:np.concatenate([d[k] for d in all_data]) for k in all_data[0].files};np.savez_compressed(OUT/'exploration.npz',**a)
    cases=a['case'];train=(cases<24)|(cases>=36);val=(cases>=24)&(cases<30);test=(cases>=30)&(cases<36)
    (OUT/'data_manifest.json').write_text(json.dumps({'training_pairs':int(train.sum()),'validation_pairs':int(val.sum()),'test_pairs':int(test.sum()),'test_initial_cases':6,'macro_seconds':DURATION,'full_joint_action_limit_rad':ACTION_LIMIT.tolist(),'sources':[json.loads((OUT/f'data_shard_{i}.json').read_text()) for i in range(shards)],'note':'All cases >=36 are training-only existing physically achieved get-up support states. Branches share initial states; samples are not independent trials.'},indent=2))
    print({k:v.shape for k,v in a.items()},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=3);p.add_argument('--merge',action='store_true');a=p.parse_args();merge(a.shards) if a.merge else collect(a.shard,a.shards)
