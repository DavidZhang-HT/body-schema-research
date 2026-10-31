"""Estimate offsets from sensor-only CSV. No injection truth is read."""
import argparse,json
import numpy as np
from zero_identification import ZeroEstimator,Packet,model
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--last-training-block',type=int,default=7);a=p.parse_args()
rows=np.loadtxt(a.csv,delimiter=',',skiprows=1)
rows=rows[rows[:,0]<=a.last_training_block]
packets=[Packet(r[1:25],r[25:29]) for r in rows]
e=ZeroEstimator(model());fit=e.fit(packets)
print(json.dumps({'offsets_deg':np.rad2deg(e.x).tolist(),'observability':fit},indent=2))
