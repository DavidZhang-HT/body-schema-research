"""Train action-conditioned spatial/temporal attention, with held-out episode splits."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,json,time,argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import rankdata
from attention_data import OUT,STEP,HISTORY,model

torch.set_num_threads(3)
class BodyAttention(nn.Module):
    def __init__(self):
        super().__init__();d=48;m=model()
        self.joint=nn.Linear(3,d);self.base=nn.Linear(19,d);self.identity=nn.Parameter(torch.randn(25,d)*.02)
        props=np.zeros((25,8),np.float32);mask=np.full((25,25),-1.5,np.float32);np.fill_diagonal(mask,0);mask[0,:]=0;mask[:,0]=0
        for j in range(1,25):
            b=m.jnt_bodyid[j];p=m.body_parentid[b];pj=int(m.body_jntadr[p]);pj=max(0,pj)
            mask[j,pj]=mask[pj,j]=0
            props[j]=np.r_[m.body_mass[b],m.body_inertia[b]*100,m.jnt_axis[j],np.linalg.norm(m.body_pos[b])]
        self.register_buffer('properties',torch.from_numpy(props));self.register_buffer('graph_bias',torch.from_numpy(mask))
        self.prop=nn.Linear(8,d);self.spatial=nn.MultiheadAttention(d,4,batch_first=True,dropout=.02)
        self.sn=nn.LayerNorm(d);self.sf=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Linear(96,d))
        self.temporal_in=nn.Linear(91,d);self.time_embed=nn.Parameter(torch.randn(HISTORY,d)*.02)
        self.temporal=nn.TransformerEncoderLayer(d,4,96,.02,batch_first=True,norm_first=True)
        self.head=nn.Sequential(nn.Linear(d*3,128),nn.GELU(),nn.Linear(128,68))
    def forward(self,x,weights=False):
        last=x[:,-1];joint=torch.stack([last[:,:24],last[:,24:48],last[:,67:91]],-1)
        z=torch.cat([self.base(last[:,48:67]).unsqueeze(1),self.joint(joint)],1)+self.identity+self.prop(self.properties)
        attended,w=self.spatial(z,z,z,attn_mask=self.graph_bias,need_weights=weights)
        z=self.sn(z+attended);z=z+self.sf(z)
        t=self.temporal(self.temporal_in(x)+self.time_embed)
        out=self.head(torch.cat([z[:,0],z[:,1:].mean(1),t[:,-1]],-1))
        return (out,w) if weights else out
class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(HISTORY*91,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,68))
    def forward(self,x):return self.net(x)

def split_ids(n):
    # Stratified by episode family. Never split overlapping windows across sets.
    train=[];val=[];test=[]
    rng=np.random.default_rng(911)
    for f in range(6):
        ids=np.arange(f,n,6);rng.shuffle(ids);test.extend(ids[:3]);val.extend(ids[3:5]);train.extend(ids[5:])
    return np.array(train),np.array(val),np.array(test)

def data():
    a=np.load(OUT/'transitions.npz');s=a['states'];u=a['actions'];tr,va,te=split_ids(len(s))
    raw=np.concatenate([s[:,:-1],u],-1)
    mean=raw[tr].reshape(-1,91).mean(0);std=np.maximum(raw[tr].reshape(-1,91).std(0),.05)
    delta=s[:,1:,:64]-s[:,:-1,:64];ds=np.maximum(delta[tr].reshape(-1,64).std(0),.002)
    norm={'mean':mean,'std':std,'delta_std':ds};np.savez(OUT/'normalization.npz',**norm)
    (OUT/'split.json').write_text(json.dumps({'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()},indent=2))
    x=np.lib.stride_tricks.sliding_window_view(raw,HISTORY,axis=1).transpose(0,1,3,2).copy()
    idx=np.arange(HISTORY-1,u.shape[1]-19);x=x[:,:len(idx)];fallen=(s[:,:,57]<.18)|(s[:,:,50]<np.cos(1.))
    risk=np.stack([fallen[:,t+1:min(t+21,s.shape[1])].any(1) for t in idx],1)
    target=np.concatenate([delta[:,idx]/ds,s[:,idx+1,64:67],risk[:,:,None]],-1).astype(np.float32)
    x=(x-mean)/std
    return s,u,x.astype(np.float32),target,norm,(tr,va,te)

def loss(out,y):
    return F.smooth_l1_loss(out[:,:64],y[:,:64])+.12*F.binary_cross_entropy_with_logits(out[:,64:67],y[:,64:67])+.12*F.binary_cross_entropy_with_logits(out[:,67],y[:,67])

def train(kind,epochs=20):
    torch.manual_seed(42);rng=np.random.default_rng(42);s,u,x,y,norm,splits=data();tr,va,te=splits
    xt=torch.from_numpy(x[tr].reshape(-1,HISTORY,91));yt=torch.from_numpy(y[tr].reshape(-1,68))
    xv=torch.from_numpy(x[va].reshape(-1,HISTORY,91));yv=torch.from_numpy(y[va].reshape(-1,68))
    tids=np.arange(HISTORY-1,HISTORY-1+x.shape[1])
    def futures(ids):
        fu=np.stack([u[ids][:,tids+j] for j in [1,2]],2).reshape(-1,2,24)
        fs=np.stack([s[ids][:,tids+j+1,:64] for j in [1,2]],2).reshape(-1,2,64)
        return torch.from_numpy(fu),torch.from_numpy(fs)
    fut, fst=futures(tr);fuv,fsv=futures(va)
    mean=torch.tensor(norm['mean']);std=torch.tensor(norm['std']);ds=torch.tensor(norm['delta_std'])
    def objective(net,xx,yy,fu,fs,steps):
        out=net(xx);value=loss(out,yy);hh=xx
        for j in range(steps-1):
            raw=hh[:,-1,:67]*std[:67]+mean[:67]
            nxt=raw[:,:64]+out[:,:64]*ds
            nxt=torch.cat([nxt[:,:48],F.normalize(nxt[:,48:51],dim=1),nxt[:,51:],out[:,64:67].sigmoid()],-1)
            row=(torch.cat([nxt,fu[:,j]],-1)-mean)/std
            hh=torch.cat([hh[:,1:],row[:,None]],1);out=net(hh)
            future=nxt[:,:64]+out[:,:64]*ds
            value=value+.2*F.smooth_l1_loss((future-fs[:,j])/ds,torch.zeros_like(future))
        return value
    net=BodyAttention() if kind=='attention' else MLP();opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-4)
    best=float('inf');history=[];start=time.time()
    for epoch in range(epochs):
        net.train();ix=torch.from_numpy(rng.permutation(len(xt)));total=0
        for ids in ix.split(128):
            l=objective(net,xt[ids],yt[ids],fut[ids],fst[ids],1 if epoch<8 else 3);opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();total+=l.item()*len(ids)
        net.eval()
        with torch.no_grad():v=sum(objective(net,xx,yy,fu,fs,3).item()*len(xx) for xx,yy,fu,fs in zip(xv.split(256),yv.split(256),fuv.split(256),fsv.split(256)))/len(xv)
        history.append({'epoch':epoch+1,'train':total/len(xt),'validation':v})
        if v<best:
            best=v;torch.save({'state_dict':net.state_dict(),'kind':kind,'epoch':epoch+1,'normalization':{k:torch.tensor(v) for k,v in norm.items()}},OUT/f'{kind}.pt')
        if epoch%2==0 or epoch==epochs-1:print(kind,epoch+1,'train',round(total/len(xt),5),'val',round(v,5),'seconds',round(time.time()-start),flush=True)
    (OUT/f'{kind}_training.json').write_text(json.dumps({'parameters':sum(p.numel() for p in net.parameters()),'wall_seconds':time.time()-start,'training_objective':'8 epochs one-step warmup then 3-step autoregressive loss; checkpoint selected by 3-step validation loss','history':history},indent=2))

def auc(y,p):
    y=np.asarray(y,dtype=bool);n=y.sum();m=len(y)-n
    return None if n==0 or m==0 else float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m))

def evaluate(kind):
    s,u,x,y,norm,splits=data();test=splits[2];ck=torch.load(OUT/f'{kind}.pt',weights_only=True);net=BodyAttention() if kind=='attention' else MLP();net.load_state_dict(ck['state_dict']);net.eval()
    with torch.no_grad():p=torch.cat([net(xx) for xx in torch.from_numpy(x[test].reshape(-1,HISTORY,91)).split(256)]).numpy()
    real=y[test].reshape(-1,68);ds=norm['delta_std'];delta_error=(p[:,:64]-real[:,:64])*ds
    idx=np.arange(HISTORY-1,HISTORY-1+x.shape[1]);current=s[test][:,idx].reshape(-1,67);nextstate=s[test][:,idx+1].reshape(-1,67)
    persist=current[:,:64]-nextstate[:,:64]
    groups={'joint_angle_rad':slice(0,24),'joint_velocity_rad_s':slice(24,48),'up_vector':slice(48,51),'base_height_m':slice(57,58),'com_velocity_m_s':slice(61,64)}
    metric=lambda e:{k:float(np.sqrt(np.mean(e[:,sl]**2))) for k,sl in groups.items()}
    prob=1/(1+np.exp(-np.clip(p[:,67],-30,30)));upright=(current[:,57]>.22)&(current[:,50]>.7)
    result={'kind':kind,'selected_epoch':ck['epoch'],'test_episodes':test.tolist(),'one_step':metric(delta_error),'persistence':metric(persist),'contact_accuracy':float(np.mean((p[:,64:67]>0)==(real[:,64:67]>.5))),'fall_auc_all':auc(real[:,67],prob),'fall_auc_currently_upright':auc(real[upright,67],prob[upright]),'upright_samples':int(upright.sum()),'upright_positive_samples':int(real[upright,67].sum()),'rollout':{},'rollout_baselines':{},'one_step_by_family':{}}
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    families=np.repeat([manifest['episodes'][int(ep)]['family'] for ep in test],x.shape[1])
    for fam in sorted(set(families)):
        result['one_step_by_family'][fam]=metric(delta_error[families==fam])
    # Open-loop autoregression: only the initial history and recorded future actions are supplied.
    pairs=[(ep,t) for ep in test for t in range(HISTORY-1,u.shape[1]-20,20)]
    hist=np.stack([np.concatenate([s[e,t-HISTORY+1:t+1],u[e,t-HISTORY+1:t+1]],-1) for e,t in pairs]);preds=[]
    for h in range(1,21):
        with torch.no_grad():pr=net(torch.tensor((hist-norm['mean'])/norm['std'],dtype=torch.float32)).numpy()
        nxt=hist[:,-1,:67].copy();nxt[:,:64]+=pr[:,:64]*ds;nxt[:,64:67]=1/(1+np.exp(-np.clip(pr[:,64:67],-30,30)))
        nxt[:,48:51]/=np.maximum(np.linalg.norm(nxt[:,48:51],axis=1,keepdims=True),1e-6)
        nxt=np.clip(nxt,-100,100)
        if h in [1,5,10,20]:
            truth=np.stack([s[e,t+h] for e,t in pairs]);result['rollout'][str(h)]={'seconds':h*STEP,**metric(nxt[:,:64]-truth[:,:64])}
            initial=np.stack([s[e,t] for e,t in pairs]);cv=initial.copy();cv[:,:24]+=initial[:,24:48]*h*STEP;cv[:,57]+=np.sum(initial[:,48:51]*initial[:,54:57],axis=1)*h*STEP
            result['rollout_baselines'][str(h)]={'persistence':metric(initial[:,:64]-truth[:,:64]),'constant_velocity':metric(cv[:,:64]-truth[:,:64])}
        act=np.stack([u[e,t+h] for e,t in pairs]);hist=np.concatenate([hist[:,1:],np.concatenate([nxt,act],-1)[:,None]],1)
    if kind=='attention':
        with torch.no_grad():_,weights=net(torch.from_numpy(x[test[0],:128]),weights=True)
        np.save(OUT/'attention_weights.npy',weights.mean(0).numpy())
    (OUT/f'{kind}_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['attention','mlp'],default='attention');p.add_argument('--epochs',type=int,default=20);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if not a.evaluate:train(a.kind,a.epochs)
    evaluate(a.kind)
