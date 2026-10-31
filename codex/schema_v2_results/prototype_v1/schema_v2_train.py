"""Matched sample-exposure curriculum and shuffled-order world-model training."""
from schema_v2_models import *
import argparse,time,hashlib


def dataset():
    a=dict(np.load(DEST/'exploration.npz'));train=a['split']==0
    flat=transform(a['raw'][train,:-1].reshape(-1,398));mean=flat.mean(0)
    floor=np.r_[np.tile([.1,.2,.2,.2,.02,.02,.02,.2,.2,.2,.02,.02,.02,.25,.02],25),np.full(23,.03)]
    std=np.maximum(flat.std(0),floor).astype(np.float32)
    data={k:torch.tensor(a[k],dtype=torch.float32) for k in ['raw','q','actions','sensory','cost']}
    fk=Kinematics()
    with torch.no_grad():
        shape=data['q'].shape; p,r=fk(data['q'].reshape(-1,31));bc=fk.body_com(p,r)
        data['rot']=r.reshape(shape[0],shape[1],25,3,3);data['bodycom']=bc.reshape(shape[0],shape[1],25,3)
    return a,data,torch.tensor(mean),torch.tensor(std)


def presentation_schedule(a,steps_per_stage=80,batch=96):
    rng=np.random.default_rng(8301);schedule=[]
    for stage in range(4):
        new=np.where((a['split']==0)&(a['stage']==stage))[0]
        old=np.where((a['split']==0)&(a['stage']<stage))[0]
        for step in range(steps_per_stage):
            nold=round(batch*.35) if len(old) else 0
            ids=np.r_[rng.choice(new,batch-nold),rng.choice(old,nold) if nold else np.array([],dtype=int)]
            rng.shuffle(ids);schedule.append(ids)
    schedule=np.array(schedule);mixed=schedule.ravel().copy();rng.shuffle(mixed);mixed=mixed.reshape(schedule.shape)
    assert np.array_equal(np.sort(schedule.ravel()),np.sort(mixed.ravel()))
    return schedule,mixed


def batch_data(data,ids):return {k:v[ids] for k,v in data.items()}


def loss_step(out,b,index):
    raw=b['raw'][:,index+1];target=b['sensory'][:,index];p=out['sensory'];node=raw[:,:375].reshape(-1,25,15)
    com=b['q'][:,index+1,:3]+raw[:,388:391]
    loss=2*F.smooth_l1_loss(out['bodycom']/.03,b['bodycom'][:,index+1]/.03)
    loss+=4*F.smooth_l1_loss(out['com']/.02,com/.02)+.5*F.smooth_l1_loss(out['cv']/.15,raw[:,391:394]/.15)
    loss+=F.smooth_l1_loss(out['q'][:,7:]/.2,b['q'][:,index+1,7:]/.2)
    loss+=F.mse_loss(out['r'][:,0],b['rot'][:,index+1,0])
    loss+=.2*F.smooth_l1_loss(out['raw'][:,:375].reshape(-1,25,15)[:,1:,1]/5,node[:,1:,1]/5)
    loss+=.2*F.smooth_l1_loss(out['raw'][:,381:387],raw[:,381:387])
    loss+=F.smooth_l1_loss(p[:,:,:2]*5,target[:,:,:2]*5)
    loss+=.15*F.smooth_l1_loss(p[:,:,2:6],target[:,:,2:6])+.1*F.binary_cross_entropy(p[:,:,6].clamp(1e-6,1-1e-6),target[:,:,6])
    loss+=.5*F.smooth_l1_loss(p[:,:,7:10]/.05,target[:,:,7:10]/.05)
    loss+=.8*F.smooth_l1_loss(p[:,:,10:13]/.05,target[:,:,10:13]/.05)
    loss+=.2*F.smooth_l1_loss(p[:,:,13],target[:,:,13])
    loss+=.5*F.smooth_l1_loss(torch.log1p(out['cost'][:,0]/.03),torch.log1p(b['cost'][:,index,0]/.03))
    loss+=.2*F.smooth_l1_loss(torch.log1p(out['cost'][:,1]),torch.log1p(b['cost'][:,index,1]))
    return loss


def sequence_loss(model,b):
    outputs=model.imagine(b['raw'][:,0],b['q'][:,0],b['actions'])
    return sum(loss_step(out,b,i) for i,out in enumerate(outputs))/HORIZON


def validation_loss(model,data,ids):
    model.eval();values=[]
    with torch.no_grad():
        for batch in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
            values.append((float(sequence_loss(model,batch_data(data,batch))),len(batch)))
    return sum(v*n for v,n in values)/sum(n for v,n in values)


def fit(label='curriculum',steps_per_stage=80):
    torch.set_num_threads(2);torch.manual_seed(8300)
    a,data,mean,std=dataset();schedules=presentation_schedule(a,steps_per_stage)
    np.save(DEST/f'presentations_{label}.npy',schedules[label=='mixed'])
    schedule=schedules[label=='mixed'];net=SchemaWorld(mean,std)
    optimizer=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4)
    val=np.where(a['split']==1)[0];history=[];best=float('inf');start=time.time()
    for step,ids in enumerate(schedule):
        net.train();loss=sequence_loss(net,batch_data(data,ids));optimizer.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),3);optimizer.step()
        if (step+1)%20==0:
            vl=validation_loss(net,data,val);row={'step':step+1,'stage':step//steps_per_stage,'training_loss':float(loss.detach()),'validation_loss':vl,'wall_seconds':time.time()-start};history.append(row);print(label,row,flush=True)
            if step>=3*steps_per_stage and vl<best:
                best=vl
    # The primary order comparison uses the same final update count and exact
    # sample multiset. Validation is diagnostic, not an unequal early stop.
    torch.save({'model':net.state_dict(),'mean':mean,'std':std,'step':len(schedule),'label':label},DEST/f'world_{label}.pt')
    (DEST/f'training_{label}.json').write_text(json.dumps({'history':history,'parameters':sum(p.numel() for p in net.parameters()),'presentations':int(schedule.size),'optimizer_steps':len(schedule),'sample_exposure_sha256':hashlib.sha256(np.sort(schedule.ravel()).tobytes()).hexdigest(),'best_validation_loss':best,'wall_seconds':time.time()-start},indent=2))


def proposal_goals(b,model,rng):
    batch=len(b['q']);features=torch.zeros(batch,25,16);nodes=rng.choice([0,24,17,22,6,12],batch)
    targets=b['bodycom'][:,-1].clone();rows=torch.arange(batch);local=model.fk.points[nodes]
    features[rows,nodes,:3]=local/.1;features[rows,nodes,3:6]=(targets[rows,nodes]-b['bodycom'][:,0][rows,nodes])/.05
    features[rows,nodes,6:9]=1
    vertical=torch.tensor(rng.random(batch)<.5)
    features[rows[vertical],nodes[vertical],6:8]=0
    return features,torch.tensor(nodes),targets[rows,nodes]


def fit_policy():
    torch.set_num_threads(2);torch.manual_seed(8400);rng=np.random.default_rng(8401)
    a,data,_,_=dataset();model=load_model();policy=GoalProposal();opt=torch.optim.AdamW(policy.parameters(),lr=.0006,weight_decay=.0001)
    schedule=presentation_schedule(a,steps_per_stage=30,batch=64)[0];history=[];start=time.time()
    for step,ids in enumerate(schedule):
        b=batch_data(data,ids);features,nodes,target=proposal_goals(b,model,rng)
        actions=policy(model.normalized(b['raw'][:,0]),b['rot'][:,0],features)
        loss=F.mse_loss(actions,b['actions'])
        if step>=30:
            predicted=model.imagine(b['raw'][:,0],b['q'][:,0],actions)
            end=predicted[-1]['bodycom'][torch.arange(len(ids)),nodes]
            mask=features[torch.arange(len(ids)),nodes,6:9]
            loss+=.2*F.smooth_l1_loss((end-target)*mask/.03,torch.zeros_like(target))
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),3);opt.step()
        if (step+1)%20==0:
            row={'step':step+1,'stage':step//30,'loss':float(loss.detach()),'wall_seconds':time.time()-start};history.append(row);print('policy',row,flush=True)
    torch.save({'model':policy.state_dict(),'steps':len(schedule)},DEST/'proposal.pt')
    (DEST/'training_proposal.json').write_text(json.dumps({'history':history,'note':'Goal-conditioned proposal plus learned-model search; direct actor success is evaluated separately.'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=['curriculum','mixed','policy'],default='curriculum');p.add_argument('--steps-per-stage',type=int,default=80);args=p.parse_args()
    fit_policy() if args.label=='policy' else fit(args.label,args.steps_per_stage)
