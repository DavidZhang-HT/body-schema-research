"""Hierarchical part attention, imagined sensations and impulse-consistent dynamics."""
from goal_models import BodyEncoder,Kinematics,transform,ASINH,exp_rotation,quaternion_matrix,torch,nn,F
from schema_v2_data import *

PART_OF_NODE=torch.tensor([0]+[3]*6+[4]*6+[1]*5+[2]*5+[5]*2)


def quaternion_step(q, rotation_vector):
    angle=torch.linalg.vector_norm(rotation_vector,dim=-1,keepdim=True)
    d=torch.cat([torch.cos(angle/2),.5*torch.sinc(angle/(2*torch.pi))*rotation_vector],-1)
    a,b=d[:,:1],d[:,1:];c,e=q[:,:1],q[:,1:]
    return F.normalize(torch.cat([a*c-(b*e).sum(-1,keepdim=True),a*e+c*b+torch.cross(b,e,dim=-1)],-1),dim=-1)


class SchemaWorld(nn.Module):
    def __init__(self,mean=None,std=None):
        super().__init__();self.encoder=BodyEncoder(1);self.fk=Kinematics()
        self.register_buffer('mean',torch.zeros(398) if mean is None else mean)
        self.register_buffer('std',torch.ones(398) if std is None else std)
        self.register_buffer('part_of_node',PART_OF_NODE)
        pooling=F.one_hot(PART_OF_NODE,6).float().T;pooling=pooling/pooling.sum(1,keepdim=True)
        self.register_buffer('pooling',pooling)
        self.parts_to_body=nn.MultiheadAttention(64,4,batch_first=True)
        self.parts_combine=nn.TransformerEncoderLayer(64,4,128,0.,batch_first=True,norm_first=True)
        self.sensation_head=nn.Linear(64,14);self.sensation_embed=nn.Linear(14,64)
        self.total_impulse=nn.Linear(64,6)
        self.sensory_identity=nn.Parameter(torch.randn(25,64)*.02)
        self.sensation_to_motion=nn.MultiheadAttention(64,4,batch_first=True)
        self.motion_node=nn.Linear(64,2);self.motion_global=nn.Linear(64,10)
        self.register_buffer('motor_limits',torch.tensor(np.r_[0.,GoalWorld().torque_limit],dtype=torch.float32))
        for layer in [self.sensation_head,self.total_impulse,self.motion_node,self.motion_global]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def normalized(self,raw):
        x=raw.clone();x[:,ASINH]=torch.asinh(x[:,ASINH]/10)
        return (x-self.mean)/self.std

    @staticmethod
    def force_cone(value):
        normal=torch.clamp(value[...,2:3],min=0)
        tangent=value[...,:2];length=torch.linalg.vector_norm(tangent,dim=-1,keepdim=True)
        tangent=tangent*torch.clamp(.8*normal/torch.clamp(length,min=1e-8),max=1)
        return torch.cat([tangent,normal],-1)

    def allocate_impulse(self,total,local,old_force):
        # Separate the net external impulse from opposing local shear forces.
        weight=torch.softmax(torch.log(torch.clamp(old_force[:,:,2],min=0)/100+.01)+local[:,:,2],dim=1)
        normal=total[:,None,2:3]*weight[:,:,None]
        ratio=total[:,:2]/torch.clamp(.8*total[:,2:3],min=1e-8)
        shear=torch.tanh(local[:,:,:2]);shear=shear-(shear*weight[:,:,None]).sum(1,keepdim=True)
        shear=shear/torch.clamp(torch.linalg.vector_norm(shear,dim=-1).max(1).values[:,None,None],min=1)
        margin=torch.clamp(1-torch.linalg.vector_norm(ratio,dim=-1),min=0)[:,None,None]
        tangent=.8*normal*(ratio[:,None]+.5*margin*shear)
        return torch.cat([tangent,normal],-1)

    def forward(self,raw,q,action,rotations=None,intervention=None):
        r=self.fk(q)[1] if rotations is None else rotations
        x=self.normalized(raw);a=torch.cat([torch.zeros_like(action[:,:1]),action],1)[:,:,None]
        z=self.encoder(x,r,a)
        parts=torch.einsum('pn,bnd->bpd',self.pooling,z[:,:25])
        context,part_attention=self.parts_to_body(parts,z,z,need_weights=True,average_attn_weights=True)
        parts=self.parts_combine(parts+context)
        body=z[:,:25]+parts[:,self.part_of_node]
        sp=self.sensation_head(body);old=raw[:,:375].reshape(-1,25,15)
        # Nominal supported-body total is gravity-balanced, not an extrapolation
        # of the possibly impulsive instantaneous force at the current sample.
        net=self.total_impulse(z[:,26]+parts.mean(1)).reshape(-1,2,3)
        total=self.force_cone(torch.tensor([0.,0.,1.],device=q.device)+.04*net)
        j=self.allocate_impulse(total[:,0],sp[:,:,7:10],old[:,:,7:10])
        k=self.allocate_impulse(total[:,1],sp[:,:,10:13],old[:,:,7:10])
        motor_mask=(self.motor_limits>0)[None]
        mean=torch.tanh(sp[:,:,0])*motor_mask;rms=torch.sigmoid(sp[:,:,1]-3)*motor_mask
        contact=torch.sigmoid(sp[:,:,6]+4*old[:,:,13]-2)
        endforce=10*torch.sinh(torch.clamp(torch.asinh(old[:,:,7:10]/10)+.4*sp[:,:,3:6],-5,5))
        endforce=self.force_cone(endforce)
        sensation=torch.cat([mean[:,:,None],rms[:,:,None],sp[:,:,2:3],torch.asinh(endforce/10),contact[:,:,None],j,k,(torch.tanh(sp[:,:,13])*motor_mask)[:,:,None]],-1)
        feed=sensation
        if intervention=='zero':feed=torch.zeros_like(feed)
        elif intervention=='swap':feed=torch.roll(feed,shifts=6,dims=1)
        sensory_tokens=self.sensation_embed(feed)+self.sensory_identity
        motion_context,sensory_attention=self.sensation_to_motion(torch.cat([body,z[:,26:27]],1),sensory_tokens,sensory_tokens,need_weights=True,average_attn_weights=True)
        motion=body+motion_context[:,:25];n=self.motion_node(motion);g=self.motion_global(z[:,26]+motion_context[:,25])
        joints=torch.clamp(q[:,7:]+action*self.fk.limit+.12*n[:,1:,0],self.fk.qlo,self.fk.qhi)
        newquat=quaternion_step(q[:,3:7],.3*g[:,:3]);rootr=quaternion_matrix(newquat)
        p_relative,rnext=self.fk.from_parts(torch.zeros_like(q[:,:3]),rootr,joints)
        relative_com=self.fk.body_com(p_relative,rnext)
        global0=raw[:,375:];com0=q[:,:3]+global0[:,13:16];v0=global0[:,16:19]
        gravity=torch.tensor([0.,0.,-9.81],dtype=q.dtype,device=q.device)
        com=com0+v0*DURATION+gravity*(DURATION**2/2)+k.sum(1)*(9.81*DURATION**2/2)
        cv=v0+gravity*DURATION+j.sum(1)*(9.81*DURATION)
        rootp=com-(relative_com*self.fk.mass[None,:,None]).sum(1)/self.fk.mass.sum()
        p=p_relative+rootp[:,None];bodycom=relative_com+rootp[:,None]
        jointv=old[:,1:,1]+5*n[:,1:,1]
        rootv=global0[:,6:12]+g[:,3:9]*torch.tensor([.5,.5,.5,2.,2.,2.],device=q.device)
        nextnode=torch.zeros_like(old)
        nextnode[:,1:,0]=joints;nextnode[:,1:,1]=jointv
        nextnode[:,:,2]=sensation[:,:,13]*self.motor_limits
        nextnode[:,:,3]=10*torch.sinh(torch.clamp(sensation[:,:,2],-5,5))
        nextnode[:,:,4:7]=bodycom-rootp[:,None];nextnode[:,:,7:10]=endforce
        # Approximate future contact location at each body's x/y over the flat plane.
        points=torch.cat([bodycom[:,:,:2]-rootp[:,None,:2],-rootp[:,None,2:3].expand(-1,25,-1)],-1)
        nextnode[:,:,10:13]=points*contact[:,:,None];nextnode[:,:,13]=contact;nextnode[:,:,14]=bodycom[:,:,2]
        nextglobal=torch.cat([rootr[:,:,:2].transpose(-1,-2).reshape(-1,6),rootv,rootp[:,2:3],com-rootp,cv,com[:,2:3],rootr[:,:,2]],-1)
        nextraw=torch.cat([nextnode.reshape(-1,375),nextglobal],-1)
        nextq=torch.cat([rootp,newquat,joints],-1)
        effort=DURATION*rms.square().sum(1);work=F.softplus(g[:,9]-4)
        return {'raw':nextraw,'q':nextq,'r':rnext,'p':p,'bodycom':bodycom,'com':com,'cv':cv,'sensory':sensation,
                'cost':torch.stack([effort,work],-1),'part_attention':part_attention,'sensory_attention':sensory_attention}

    def imagine(self,raw,q,actions,intervention=None):
        result=[];r=None
        for i in range(actions.shape[1]):
            out=self(raw,q,actions[:,i],r,intervention);result.append(out)
            raw,q,r=out['raw'],out['q'],out['r']
        return result


class GoalProposal(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=BodyEncoder(0);self.goal=nn.Linear(16,64)
        self.cross=nn.MultiheadAttention(64,4,batch_first=True)
        self.decoder=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,HORIZON))
    def forward(self,x,r,goals):
        z=self.encoder(x,r,goals[:,:,:0]);g=self.goal(goals)+z[:,:25]
        active=goals.abs().sum(-1)>0;g=torch.cat([g,z[:,26:27]],1)
        padding=torch.cat([~active,torch.zeros((len(x),1),dtype=torch.bool,device=x.device)],1)
        context,_=self.cross(z[:,1:25],g,g,key_padding_mask=padding,need_weights=False)
        a=torch.tanh(self.decoder(torch.cat([z[:,1:25],context],-1))).transpose(1,2)
        mask=torch.tensor(np.r_[np.ones(22),np.zeros(2)],dtype=a.dtype,device=a.device)
        return a*mask


def load_model(label='curriculum'):
    ck=torch.load(DEST/f'world_{label}.pt',weights_only=True)
    model=SchemaWorld(ck['mean'],ck['std']);model.load_state_dict(ck['model']);model.eval()
    for param in model.parameters():param.requires_grad_(False)
    return model
