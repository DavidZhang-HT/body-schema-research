"""Numerical contract tests for observability, sign convention and sensor-only replay."""
import numpy as np
from zero_identification import ZeroEstimator,Packet,model,standing,candidate_poses

def main():
 m=model();q0=standing(m);qs=candidate_poses(m,q0);bias=np.deg2rad(np.array([2,-1.2,1.8,-2.4,1.3,-1.6,-2.1,1.9,-1.4,2.2,-1.7,1.1]))
 def packet(q):
  e=q[7:].copy();e[:12]+=bias;return Packet(e,q[3:7].copy())
 e=ZeroEstimator(m);one=[packet(qs[0])]*10;f=e.fit(one)
 assert f['relative_rank_1e3']<=5, f
 e=ZeroEstimator(m);data=[packet(q) for q in qs];f=e.fit(data)
 assert f['relative_rank_1e3']==12,f
 assert np.max(abs(e.x[:12]-bias))<1e-5, np.rad2deg(e.x[:12]-bias)
 # Packet has no true angle, offset truth, world foot pose, or base translation fields.
 assert set(Packet.__dataclass_fields__)=={'encoder','imu_quat'}
 print('PASS: single posture rank <=5; diverse poses rank12; zero offsets recovered with correct sign to <1e-5 rad; sensor-only estimator API.')
if __name__=='__main__':main()
