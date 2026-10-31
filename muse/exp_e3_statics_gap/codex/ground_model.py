"""Body/contact attention predictor for 0.6 s bounded schema actions."""
from ground_schema import *
import torch,time
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(3)
CONTACT=np.arange(25)*15+13
FORCES=np.concatenate([np.arange(15*i+7,15*i+10) for i in range(25)])
LOADS=np.arange(1,25)*15+3
ASINH=np.concatenate([np.array([15*i+j for i in range(25) for j in [2,3,7,8,9]])])
def transform(s):
 z=np.array(s,dtype=np.float32,copy=True);z[...,ASINH]=np.arcsinh(z[...,ASINH]/10);return z
def untransform(s):
 z=np.array(s,dtype=np.float32,copy=True);z[...,ASINH]=10*np.sinh(np.clip(z[...,ASINH],-6,6));return z

class ContactAttention(nn.Module):
 def __init__(self):
  super().__init__();d=48;self.node=nn.Linear(16,d);self.global_=nn.Linear(23+12,d);self.identity=nn.Parameter(torch.randn(27,d)*.02);self.ground=nn.Parameter(torch.randn(1,1,d)*.02)
  self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,4,96,.02,batch_first=True,norm_first=True) for _ in range(2)])
  self.node_out=nn.Linear(d,15);self.global_out=nn.Linear(d,23);self.register_buffer('basis',torch.tensor(B,dtype=torch.float32))
  w=GroundWorld();mask=np.full((27,27),-1.5,np.float32);np.fill_diagonal(mask,0);mask[25:,:]=0;mask[:,25:]=0
  prop=np.zeros((27,8),np.float32)
  for j,b in enumerate(w.bodies):
   prop[j,:4]=np.r_[w.m.body_mass[b],w.m.body_inertia[b]*100]
   if j:
    par=max(0,int(w.m.body_jntadr[w.m.body_parentid[b]]));mask[j,par]=mask[par,j]=0;prop[j,4:7]=w.m.jnt_axis[j];prop[j,7]=np.linalg.norm(w.m.body_pos[b])
  self.register_buffer('mask',torch.tensor(mask));self.register_buffer('properties',torch.tensor(prop));self.prop=nn.Linear(8,d)
 def forward(self,s,a):
  nodes=s[:,:375].reshape(-1,25,15);delta=torch.cat([torch.zeros_like(a[:,:1]),a@self.basis.T],1)
  z=self.node(torch.cat([nodes,delta[:,:,None]],-1));g=self.global_(torch.cat([s[:,375:],a],-1))[:,None]
  z=torch.cat([z,self.ground.expand(len(s),-1,-1),g],1)+self.identity+self.prop(self.properties)
  for layer in self.layers:z=layer(z,src_mask=self.mask)
  return torch.cat([self.node_out(z[:,:25]).flatten(1),self.global_out(z[:,26])],-1)

def load_data():
 a=np.load(OUT/'exploration.npz');cases=a['case'];train=cases<16;val=(cases>=16)&(cases<20);test=cases>=20
 arrays=[a['x'][train],a['u'][train],a['y'][train]]
 oracles=[]
 for pose in ['supine','prone']:
  path=OUT/f'oracle_{pose}_data.npz'
  if path.exists():
   o=np.load(path);oracles.append(path.name)
   arrays=[np.concatenate([v,o[k]],0) for v,k in zip(arrays,['x','u','y'])]
 x,u,y=arrays;tx=transform(x);ty=transform(y);mean=np.concatenate([tx,ty]).mean(0);floor=np.tile([.1,.2,.2,.2,.02,.02,.02,.2,.2,.2,.02,.02,.02,.25,.02],25);floor=np.r_[floor,np.full(23,.03)];std=np.maximum(np.concatenate([tx,ty]).std(0),floor).astype(np.float32)
 np.savez(OUT/'normalization.npz',mean=mean,std=std)
 (OUT/'split.json').write_text(json.dumps({'train_cases':list(range(16)),'validation_cases':list(range(16,20)),'test_cases':list(range(20,24)),'oracle_training_files':oracles,'training_pairs':len(x),'validation_pairs':int(sum(val)),'test_pairs':int(sum(test))},indent=2))
 def tensors(xx,uu,yy):
  txx=transform(xx);tyy=transform(yy);target=(tyy-txx)/std;target[:,CONTACT]=yy[:,CONTACT]
  return torch.tensor((txx-mean)/std),torch.tensor(uu,dtype=torch.float32),torch.tensor(target)
 return tensors(x,u,y),tensors(a['x'][val],a['u'][val],a['y'][val]),(a['x'][test],a['u'][test],a['y'][test]),(mean,std)

def objective(p,y):
 mask=np.ones(DIM,dtype=bool);mask[CONTACT]=False
 weights=torch.ones(DIM);weights[375:]=3;weights[FORCES]=.6;weights[LOADS]=.6
 return (F.smooth_l1_loss(p[:,mask],y[:,mask],reduction='none')*weights[mask]).mean()+.25*F.binary_cross_entropy_with_logits(p[:,CONTACT],y[:,CONTACT])

def train(epochs=55,members=3):
 tr,va,te,norm=load_data();logs=[]
 for member in range(members):
  torch.manual_seed(800+member);rng=np.random.default_rng(800+member);net=ContactAttention();opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.0001);best=1e9;history=[];start=time.time()
  for ep in range(epochs):
   net.train();ids=rng.permutation(len(tr[0]));total=0
   for block in np.array_split(ids,max(1,len(ids)//128)):
    i=torch.tensor(block);p=net(tr[0][i],tr[1][i]);loss=objective(p,tr[2][i]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();total+=float(loss.detach())*len(i)
   net.eval()
   with torch.no_grad():v=float(objective(net(va[0],va[1]),va[2]))
   history.append([ep+1,total/len(ids),v])
   if v<best:
    best=v;torch.save({'weights':net.state_dict(),'mean':torch.tensor(norm[0]),'std':torch.tensor(norm[1]),'epoch':ep+1},OUT/f'contact_model_{member}.pt')
   if ep%10==0 or ep==epochs-1:print('contact member',member,'epoch',ep+1,'train',round(total/len(ids),4),'val',round(v,4),'wall',round(time.time()-start),flush=True)
  logs.append({'member':member,'history':history,'wall_seconds':time.time()-start,'parameters':sum(p.numel() for p in net.parameters())})
 (OUT/'training.json').write_text(json.dumps(logs,indent=2));evaluate()

class Ensemble:
 def __init__(self,members=3):
  self.models=[]
  for member in range(members):
   ck=torch.load(OUT/f'contact_model_{member}.pt',weights_only=True);net=ContactAttention();net.load_state_dict(ck['weights']);net.eval();self.models.append(net)
  self.mean=ck['mean'].numpy();self.std=ck['std'].numpy()
 def predict(self,s,a):
  trans=transform(s);xx=torch.tensor((trans-self.mean)/self.std);aa=torch.tensor(a,dtype=torch.float32);pred=[]
  with torch.no_grad():
   for net in self.models:
    o=net(xx,aa).numpy();y=untransform(trans+o*self.std);y[:,CONTACT]=1/(1+np.exp(-np.clip(o[:,CONTACT],-30,30)));y[:,np.arange(25)*15+9]=np.maximum(y[:,np.arange(25)*15+9],0);pred.append(y)
  pred=np.array(pred);return pred.mean(0),pred.std(0)

def evaluate():
 a=np.load(OUT/'exploration.npz');test=a['case']>=20;x,u,y=[a[k][test] for k in ['x','u','y']];net=Ensemble();pred,unc=net.predict(x,u);err=pred-y
 def metrics(e):return {'com_height_rmse_mm':float(np.sqrt(np.mean(e[:,-4]**2))*1000),'base_height_rmse_mm':float(np.sqrt(np.mean(e[:,375+12]**2))*1000),'contact_force_component_rmse_N':float(np.sqrt(np.mean(e[:,FORCES]**2))),'ground_joint_load_rmse_Nm':float(np.sqrt(np.mean(e[:,LOADS]**2))),'joint_angle_rmse_rad':float(np.sqrt(np.mean(e[:,np.arange(1,25)*15]**2)))}
 tp=np.sum((pred[:,CONTACT]>.5)&(y[:,CONTACT]>.5));fp=np.sum((pred[:,CONTACT]>.5)&(y[:,CONTACT]<.5));fn=np.sum((pred[:,CONTACT]<.5)&(y[:,CONTACT]>.5))
 delta_true=y[:,-4]-x[:,-4];delta_pred=pred[:,-4]-x[:,-4]
 r={'heldout_pairs':len(x),'horizon_seconds':DURATION,'prediction':metrics(err),'persistence':metrics(x-y),'contact_f1':float(2*tp/max(2*tp+fp+fn,1)),'com_delta_correlation':float(np.corrcoef(delta_true,delta_pred)[0,1]),'com_delta_sign_accuracy_over_2mm':float(np.mean(np.sign(delta_pred[abs(delta_true)>.002])==np.sign(delta_true[abs(delta_true)>.002]))),'scope':'Test starts from held-out unpowered equilibria. Oracle search transitions used only for training.'}
 (OUT/'evaluation.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'test_predictions.npz',x=x,u=u,y=y,prediction=pred,uncertainty=unc);print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=55);p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else train(a.epochs)
