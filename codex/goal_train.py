"""Train dynamics and a shared hindsight-goal attention policy on disjoint starts."""
from goal_models import *
import argparse,time,json

def dataset():
    a=np.load(OUT/'exploration.npz');case=a['case'];tr=(case<24)|(case>=36);va=(case>=24)&(case<30);te=(case>=30)&(case<36)
    raw=transform(a['x']);mean=raw[tr].mean(0);floor=np.r_[np.tile([.1,.2,.2,.2,.02,.02,.02,.2,.2,.2,.02,.02,.02,.25,.02],25),np.full(23,.03)];std=np.maximum(raw[tr].std(0),floor).astype(np.float32)
    out={k:torch.tensor(a[k],dtype=torch.float32) for k in ['q','next_q','u','cost']};out['x']=torch.tensor((raw-mean)/std)
    fk=Kinematics()
    with torch.no_grad():
        p0,r0=fk(out['q']);p1,r1=fk(out['next_q']);out.update(p0=p0,r0=r0,p1=p1,r1=r1)
        out['com1']=fk.body_com(p1,r1);out['com0']=fk.body_com(p0,r0)
    rootrot=Rotation.from_quat(a['next_q'][:,[4,5,6,3]])*Rotation.from_quat(a['q'][:,[4,5,6,3]]).inv()
    out['dp']=torch.tensor((a['next_q'][:,:3]-a['q'][:,:3])/.05);out['dr']=torch.tensor(rootrot.as_rotvec()/.3,dtype=torch.float32)
    node=a['y'][:,:375].reshape(-1,25,15);out['force']=torch.tensor(np.arcsinh(node[:,:,7:10]/10));out['load']=torch.tensor(np.arcsinh(node[:,:,3]/10));out['contact']=torch.tensor(node[:,:,13]);out['velocity']=torch.tensor(node[:,1:,1]/5)
    out['cost_log']=torch.log1p(out['cost']/torch.tensor(COST_SCALE))
    return out,[np.where(z)[0] for z in [tr,va,te]],torch.tensor(mean),torch.tensor(std),fk,a

def slice_batch(data,ids):return {k:v[ids] for k,v in data.items()}

def dynamics_loss(model,b):
    pred=model(b['x'],b['q'],b['u'],b['r0']);n=pred['nodes'];g=pred['global'];target=(b['next_q'][:,7:]-torch.clamp(b['q'][:,7:]+b['u']*model.fk.limit,model.fk.qlo,model.fk.qhi))/.1
    loss=F.smooth_l1_loss(n[:,1:,0],target)+3*F.smooth_l1_loss(g[:,:3],b['dp'])+F.smooth_l1_loss(g[:,3:6],b['dr'])
    loss+=2*F.smooth_l1_loss(model.fk.body_com(pred['p'],pred['r'])/.03,b['com1']/.03)
    loss+=.2*F.smooth_l1_loss(n[:,:,1:4],b['force'])+.2*F.smooth_l1_loss(n[:,:,4],b['load'])+.1*F.binary_cross_entropy_with_logits(n[:,:,5],b['contact'])
    loss+=.3*F.smooth_l1_loss(g[:,6:9],b['cost_log'])+.03*F.smooth_l1_loss(n[:,1:,6],b['velocity'])
    return loss

def fit_dynamics(blind=False,epochs=35):
    torch.manual_seed(6100);rng=np.random.default_rng(6100);data,splits,mean,std,fk,raw=dataset();tr,va,te=splits;net=Dynamics(blind);opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001);history=[];best=1e9;start=time.time();label='blind' if blind else 'full'
    for ep in range(epochs):
        net.train();total=0
        for ids in np.array_split(rng.permutation(tr),max(1,len(tr)//128)):
            loss=dynamics_loss(net,slice_batch(data,ids));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3);opt.step();total+=float(loss.detach())*len(ids)
        net.eval()
        with torch.no_grad():vl=float(dynamics_loss(net,slice_batch(data,va)))
        history.append([ep+1,total/len(tr),vl])
        if vl<best:
            best=vl;torch.save({'model':net.state_dict(),'mean':mean,'std':std,'epoch':ep+1,'blind':blind},OUT/f'dynamics_{label}.pt')
        if ep%5==0 or ep==epochs-1:print(label,'dynamics epoch',ep+1,'train',round(total/len(tr),4),'val',round(vl,4),'wall',round(time.time()-start),flush=True)
    (OUT/f'training_dynamics_{label}.json').write_text(json.dumps({'history':history,'wall_seconds':time.time()-start,'parameters':sum(p.numel() for p in net.parameters()),'train_pairs':len(tr),'validation_pairs':len(va),'test_pairs':len(te)},indent=2));evaluate(blind)

def hindsight_goals(b,fk,rng):
    batch=len(b['q']);goals=torch.zeros((batch,25,16));local=fk.points[None].expand(batch,-1,-1).clone();local+=torch.tensor(rng.normal(0,.005,(batch,25,3)),dtype=torch.float32)
    p0=fk.markers(b['p0'],b['r0'],local);p1=fk.markers(b['p1'],b['r1'],local);movement=torch.linalg.vector_norm(p1-p0,dim=-1).numpy()
    axis_index=rng.integers(0,3,(batch,25));axis=torch.eye(3)[axis_index];d0=torch.einsum('bnij,bnj->bni',b['r0'],axis);d1=torch.einsum('bnij,bnj->bni',b['r1'],axis)
    pmask=torch.zeros((batch,25,3));dmask=torch.zeros((batch,25))
    for i in range(batch):
        options=np.where(movement[i]>.002)[0]
        if not len(options):options=np.arange(25)
        selected=rng.choice(options,min(len(options),2 if rng.random()<.2 else 1),replace=False)
        for n in selected:
            mode=rng.random()
            if mode<.5:pmask[i,n,2]=1
            elif mode<.8:pmask[i,n]=1
            else:dmask[i,n]=1
            goals[i,n]=torch.cat([local[i,n]/.1,(p1[i,n]-p0[i,n])/.05,pmask[i,n],axis[i,n],d1[i,n]-d0[i,n],dmask[i,n,None]])
    return goals,local,p1,pmask,axis,d1,dmask

def policy_loss(actor,world,b,targets,pretrain=False):
    goals,local,target,pmask,axis,dtarget,dmask=targets;action=actor(b['x'],b['r0'],goals);bc=F.mse_loss(action,b['u'])
    if pretrain:return bc
    prediction=world(b['x'],b['q'],action,b['r0']);points=world.fk.markers(prediction['p'],prediction['r'],local);dirs=torch.einsum('bnij,bnj->bni',prediction['r'],axis)
    position=(((points-target)/.03)**2*pmask).sum()/torch.clamp(pmask.sum(),min=1)
    direction=((((dirs-dtarget)/.2)**2).sum(-1)*dmask).sum()/torch.clamp(dmask.sum(),min=1)
    return position+direction+.1*bc+.002*action.square().mean()

def fit_policy(blind=False,epochs=30):
    torch.manual_seed(6200);rng=np.random.default_rng(6200);data,splits,mean,std,fk,raw=dataset();tr,va,te=splits;predictor=Predictor(blind);world=predictor.model;actor=GoalPolicy(blind);opt=torch.optim.AdamW(actor.parameters(),lr=.0006,weight_decay=.0001);label='blind' if blind else 'full';best=1e9;history=[];start=time.time()
    validation=slice_batch(data,va);valgoals=hindsight_goals(validation,fk,np.random.default_rng(6201))
    for ep in range(epochs):
        actor.train();total=0
        for ids in np.array_split(rng.permutation(tr),max(1,len(tr)//128)):
            b=slice_batch(data,ids);targets=hindsight_goals(b,fk,rng);loss=policy_loss(actor,world,b,targets,pretrain=ep<3);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),3);opt.step();total+=float(loss.detach())*len(ids)
        actor.eval()
        with torch.no_grad():vl=float(policy_loss(actor,world,validation,valgoals))
        history.append([ep+1,total/len(tr),vl])
        if vl<best:
            best=vl;torch.save({'model':actor.state_dict(),'epoch':ep+1,'blind':blind},OUT/f'policy_{label}.pt')
        if ep%5==0 or ep==epochs-1:print(label,'policy epoch',ep+1,'train',round(total/len(tr),4),'val',round(vl,4),'wall',round(time.time()-start),flush=True)
    (OUT/f'training_policy_{label}.json').write_text(json.dumps({'history':history,'wall_seconds':time.time()-start,'parameters':sum(p.numel() for p in actor.parameters())},indent=2))

def evaluate(blind=False):
    data,splits,mean,std,fk,raw=dataset();te=splits[2];b=slice_batch(data,te);model=Predictor(blind).model
    with torch.no_grad():
        out=model(b['x'],b['q'],b['u'],b['r0']);marker=fk.body_com(out['p'],out['r']);kinp,kinr=fk.from_parts(b['q'][:,:3],b['r0'][:,0],torch.clamp(b['q'][:,7:]+b['u']*fk.limit,fk.qlo,fk.qhi));kinmarker=fk.body_com(kinp,kinr)
    e=(marker-b['com1']).numpy();ke=(kinmarker-b['com1']).numpy();forces=10*torch.sinh(b['force']);active=b['contact']>.5
    perbody={str(n):{'learned_point_rmse_mm':float(np.sqrt(np.mean(np.sum(e[:,n]**2,-1)))*1000),'fixed_base_kinematic_rmse_mm':float(np.sqrt(np.mean(np.sum(ke[:,n]**2,-1)))*1000)} for n in TARGET_NODES}
    costs=out['cost'].numpy();truth=b['cost'].numpy();r={'condition':'no explicit contact features' if blind else 'full contact features','test_pairs':len(te),'independent_initial_cases':6,'point_3d_rmse_mm':float(np.sqrt(np.mean(np.sum(e**2,-1)))*1000),'fixed_base_kinematic_point_rmse_mm':float(np.sqrt(np.mean(np.sum(ke**2,-1)))*1000),'active_contact_fz_rmse_N':float(torch.sqrt(torch.mean((out['forces'][:,:,2][active]-forces[:,:,2][active])**2))),'effort_integral_rmse':float(np.sqrt(np.mean((costs[:,0]-truth[:,0])**2))),'effort_correlation':float(np.corrcoef(costs[:,0],truth[:,0])[0,1]),'by_body_node':perbody}
    label='blind' if blind else 'full';(OUT/f'prediction_{label}.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/f'prediction_{label}.npz',marker=marker.numpy(),truth=b['com1'].numpy(),kinematic=kinmarker.numpy(),cost=costs,cost_truth=truth,forces=out['forces'].numpy(),force_truth=forces.numpy());print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dynamics','policy','evaluate'],default='dynamics');p.add_argument('--blind',action='store_true');p.add_argument('--epochs',type=int);a=p.parse_args()
    if a.kind=='dynamics':fit_dynamics(a.blind,a.epochs or 35)
    elif a.kind=='policy':fit_policy(a.blind,a.epochs or 30)
    else:evaluate(a.blind)
