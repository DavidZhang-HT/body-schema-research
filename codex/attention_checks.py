"""Check episode isolation, normalization, model contract and action dependence."""
import json
import numpy as np
import torch
from attention_train import data,BodyAttention,OUT
s,u,x,y,norm,(tr,va,te)=data()
assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
assert len(set(tr)|set(va)|set(te))==96
raw=np.concatenate([s[:,:-1],u],-1)
assert np.allclose(norm['mean'],raw[tr].reshape(-1,91).mean(0))
assert np.isfinite(x).all() and np.isfinite(y).all()
# Window contains s[t-7:t] and u[t-7:t], never future states or future actions.
assert np.allclose(x[te[0],0]*norm['std']+norm['mean'],raw[te[0],:8],atol=1e-5)
assert np.allclose(y[te[0],0,:64]*norm['delta_std'],s[te[0],8,:64]-s[te[0],7,:64],atol=1e-6)
ck=torch.load(OUT/'attention.pt',weights_only=True);net=BodyAttention();net.load_state_dict(ck['state_dict']);net.eval()
a=torch.from_numpy(x[te].reshape(-1,8,91));target=torch.from_numpy(y[te].reshape(-1,68));rng=np.random.default_rng(2026)
with torch.no_grad():
 def mse(xx):
  pred=torch.cat([net(v) for v in xx.split(256)])
  assert pred.shape==target.shape and torch.isfinite(pred).all()
  return float(((pred[:,:64]-target[:,:64])**2).mean())
 original=mse(a);perm=torch.tensor(rng.permutation(len(a)));shuffled=a.clone();shuffled[:,:,67:91]=a[perm,:,67:91];ablated=mse(shuffled)
report={'episode_split_disjoint':True,'normalization_training_only':True,'causal_windows':True,'finite_outputs':True,'normalized_delta_mse':original,'shuffled_action_history_mse':ablated,'action_ablation_ratio':ablated/original}
(OUT/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
