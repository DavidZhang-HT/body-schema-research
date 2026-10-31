"""Predict from an 8x91 history: [67 state features,24 commanded torques]."""
import argparse,json
import numpy as np
import torch
from attention_train import BodyAttention,OUT
p=argparse.ArgumentParser();p.add_argument('history_npy',help='raw (8,91) float array, oldest first');a=p.parse_args()
h=np.load(a.history_npy)
if h.shape!=(8,91) or not np.isfinite(h).all():raise ValueError('Expected finite (8,91) history; layout in data_manifest.json')
ck=torch.load(OUT/'attention.pt',weights_only=True);net=BodyAttention();net.load_state_dict(ck['state_dict']);net.eval();n={k:v.numpy() for k,v in ck['normalization'].items()}
with torch.no_grad():o=net(torch.tensor((h-n['mean'])/n['std'],dtype=torch.float32)[None])[0].numpy()
s=h[-1,:67].copy();s[:64]+=o[:64]*n['delta_std'];s[48:51]/=max(np.linalg.norm(s[48:51]),1e-6);s[64:]=1/(1+np.exp(-np.clip(o[64:67],-30,30)))
print(json.dumps({'dt':.02,'predicted_state':s.tolist(),'behavior_policy_fall_probability':float(1/(1+np.exp(-np.clip(o[67],-30,30))))},indent=2))
