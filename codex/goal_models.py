"""URDF-consistent learned dynamics and a shared goal-conditioned attention policy."""
from goal_world import *
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

torch.set_num_threads(2)
CONTACT_FIELDS=np.array([15*i+j for i in range(25) for j in [3,7,8,9,10,11,12,13]])
ASINH=np.array([15*i+j for i in range(25) for j in [2,3,7,8,9]])
COST_SCALE=np.array([.03,1.,1.],np.float32)

def transform(obs):
    x=np.array(obs,dtype=np.float32,copy=True);x[...,ASINH]=np.arcsinh(x[...,ASINH]/10);return x

def quaternion_matrix(q):
    q=F.normalize(q,dim=-1);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def exp_rotation(v):
    angle=torch.linalg.vector_norm(v,dim=-1);x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    k=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(-1,3,3)
    a=torch.sinc(angle/torch.pi);b=.5*torch.sinc(angle/(2*torch.pi))**2
    return torch.eye(3,device=v.device,dtype=v.dtype)[None]+a[:,None,None]*k+b[:,None,None]*(k@k)

class Kinematics(nn.Module):
    def __init__(self):
        super().__init__();w=GoalWorld();m=w.m;self.parents=w.parents.tolist()
        self.register_buffer('offset',torch.tensor(m.body_pos[w.bodies],dtype=torch.float32))
        self.register_buffer('fixed_rot',torch.tensor(Rotation.from_quat(m.body_quat[w.bodies][:,[1,2,3,0]]).as_matrix(),dtype=torch.float32))
        self.register_buffer('axis',torch.tensor(np.r_[np.zeros((1,3)),m.jnt_axis[1:]],dtype=torch.float32))
        self.register_buffer('joint_pos',torch.tensor(np.r_[np.zeros((1,3)),m.jnt_pos[1:]],dtype=torch.float32))
        self.register_buffer('points',torch.tensor(w.points,dtype=torch.float32));self.register_buffer('mass',torch.tensor(m.body_mass[w.bodies],dtype=torch.float32))
        self.register_buffer('qlo',torch.tensor(w.qlo,dtype=torch.float32));self.register_buffer('qhi',torch.tensor(w.qhi,dtype=torch.float32));self.register_buffer('limit',torch.tensor(ACTION_LIMIT,dtype=torch.float32))
    def from_parts(self,rootp,rootr,joints):
        positions=[rootp];rotations=[rootr]
        for n in range(1,25):
            parent=self.parents[n];pre=rotations[parent]@self.fixed_rot[n];joint=exp_rotation(joints[:,n-1,None]*self.axis[n])
            rotations.append(pre@joint)
            local=self.offset[n]+torch.einsum('ij,bj->bi',self.fixed_rot[n],self.joint_pos[n]-(joint@self.joint_pos[n,:,None]).squeeze(-1))
            positions.append(positions[parent]+torch.einsum('bij,bj->bi',rotations[parent],local))
        return torch.stack(positions,1),torch.stack(rotations,1)
    def forward(self,q):return self.from_parts(q[:,:3],quaternion_matrix(q[:,3:7]),q[:,7:])
    def body_com(self,p,r):return p+torch.einsum('bnij,nj->bni',r,self.points)
    def markers(self,p,r,points):return p+torch.einsum('bnij,bnj->bni',r,points)

class BodyEncoder(nn.Module):
    def __init__(self,extra=1,blind=False):
        super().__init__();self.blind=blind;d=64;self.node=nn.Linear(21+extra,d);self.global_=nn.Linear(23,d);self.ground=nn.Parameter(torch.randn(1,1,d)*.02);self.identity=nn.Parameter(torch.randn(27,d)*.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,4,128,0.,batch_first=True,norm_first=True) for _ in range(2)])
        w=GoalWorld();prop=np.zeros((27,12),np.float32);mask=np.full((27,27),-.8,np.float32);np.fill_diagonal(mask,0);mask[25:,:]=0;mask[:,25:]=0
        for n,b in enumerate(w.bodies):
            prop[n,:4]=np.r_[w.m.body_mass[b],100*w.m.body_inertia[b]];prop[n,4:7]=w.m.body_pos[b]
            if n:
                prop[n,7:10]=w.m.jnt_axis[n];prop[n,10]=w.torque_limit[n-1]/20;prop[n,11]=ACTION_LIMIT[n-1]
                parent=w.parents[n];mask[n,parent]=mask[parent,n]=0
        self.prop=nn.Linear(12,d);self.register_buffer('properties',torch.tensor(prop));self.register_buffer('mask',torch.tensor(mask));self.register_buffer('blind_indices',torch.tensor(CONTACT_FIELDS))
    def forward(self,x,r,extra):
        if self.blind:
            x=x.clone();x[:,self.blind_indices]=0
        orient=r[:,:,:,:2].transpose(-1,-2).reshape(-1,25,6)
        nodes=torch.cat([x[:,:375].reshape(-1,25,15),orient,extra],-1)
        z=torch.cat([self.node(nodes),self.ground.expand(len(x),-1,-1),self.global_(x[:,375:])[:,None]],1)+self.identity+self.prop(self.properties)
        for layer in self.layers:z=layer(z,src_mask=self.mask)
        return z

class Dynamics(nn.Module):
    def __init__(self,blind=False):
        super().__init__();self.encoder=BodyEncoder(1,blind);self.fk=Kinematics();self.node_out=nn.Linear(64,7);self.global_out=nn.Linear(64,15)
        nn.init.zeros_(self.node_out.weight);nn.init.zeros_(self.node_out.bias);nn.init.zeros_(self.global_out.weight);nn.init.zeros_(self.global_out.bias)
    def forward(self,x,q,action,rotations=None):
        r=self.fk(q)[1] if rotations is None else rotations
        a=torch.cat([torch.zeros_like(action[:,:1]),action],1)[:,:,None];z=self.encoder(x,r,a);n=self.node_out(z[:,:25]);g=self.global_out(z[:,26])
        joints=torch.clamp(q[:,7:]+action*self.fk.limit+.1*n[:,1:,0],self.fk.qlo,self.fk.qhi)
        rootp=q[:,:3]+.05*g[:,:3];rootr=exp_rotation(.3*g[:,3:6])@r[:,0]
        pnext,rnext=self.fk.from_parts(rootp,rootr,joints)
        return {'nodes':n,'global':g,'p':pnext,'r':rnext,'joints':joints,'forces':10*torch.sinh(torch.clamp(n[:,:,1:4],-4,4)),'cost':torch.expm1(torch.clamp(g[:,6:9],0,6))*torch.tensor(COST_SCALE,device=x.device)}

class GoalPolicy(nn.Module):
    def __init__(self,blind=False):
        super().__init__();self.encoder=BodyEncoder(0,blind);self.goal=nn.Linear(16,64);self.cross=nn.MultiheadAttention(64,4,batch_first=True)
        self.action=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.action[-1].weight);nn.init.zeros_(self.action[-1].bias)
    def forward(self,x,rotations,goals):
        # Explicit goal-to-body cross attention avoids diluting a single target in 25 state tokens.
        z=self.encoder(x,rotations,goals[:,:,:0]);g=self.goal(goals)+z[:,:25]
        active=goals.abs().sum(-1)>0;g=torch.cat([g,z[:,26:27]],1)
        padding=torch.cat([~active,torch.zeros((len(x),1),dtype=torch.bool,device=x.device)],1)
        context,_=self.cross(z[:,1:25],g,g,key_padding_mask=padding,need_weights=False)
        return torch.tanh(self.action(torch.cat([z[:,1:25],context],-1)).squeeze(-1))

class Predictor:
    def __init__(self,blind=False):
        label='blind' if blind else 'full';ck=torch.load(OUT/f'dynamics_{label}.pt',weights_only=True);self.model=Dynamics(blind);self.model.load_state_dict(ck['model']);self.model.eval()
        self.mean=ck['mean'];self.std=ck['std'];self.actor=None
        path=OUT/f'policy_{label}.pt'
        if path.exists():
            actorck=torch.load(path,weights_only=True);self.actor=GoalPolicy(blind);self.actor.load_state_dict(actorck['model']);self.actor.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        if self.actor:
            for p in self.actor.parameters():p.requires_grad_(False)
    def inputs(self,obs,q):return (torch.tensor(transform(obs))-self.mean)/self.std,torch.tensor(q,dtype=torch.float32)

if __name__=='__main__':
    w=GoalWorld();fk=Kinematics();rng=np.random.default_rng(43);worst=0.
    for pose in ['prone','supine']:
        w.passive(pose,200)
        for _ in range(4):
            w.advance_full(w.sample_action(rng));state=w.state();p,r=fk(torch.tensor(state['qpos'][None],dtype=torch.float32));worst=max(worst,float(np.max(abs(p.detach().numpy()[0]-state['positions']))),float(np.max(abs(r.detach().numpy()[0]-state['rotations']))))
    assert worst<2e-6,worst
    print('Differentiable URDF kinematics matches MuJoCo; max position/rotation component error',worst)
