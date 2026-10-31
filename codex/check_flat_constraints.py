from zero_identification import *
m=model();d=mj.MjData(m);a=np.load(OUT/'active_seed7.npz');p=a['sensor_packets'];p=p[p[:,0]<=7]
def fun(b):
 out=[]
 for row in p:
  d.qpos[:]=0;d.qpos[3:7]=row[25:29];d.qpos[7:]=row[1:25];d.qpos[7:19]-=b;mj.mj_kinematics(m,d);f=feet(m,d)
  out.extend(np.r_[(f[1][0][2]-f[0][0][2])/.001,f[0][1][:2,2]/.003,f[1][1][:2,2]/.003])
 return out
r=least_squares(fun,np.zeros(12),max_nfev=100,loss='soft_l1',f_scale=2,bounds=(-np.deg2rad(6),np.deg2rad(6)))
s=np.linalg.svd(r.jac,compute_uv=False)
print('sv',s,'bias',np.rad2deg(r.x),'error',np.rad2deg(r.x-a['truth']),'rmse',np.rad2deg(np.sqrt(np.mean((r.x-a['truth'])**2))))
