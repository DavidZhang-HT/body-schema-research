import os
os.environ['MUJOCO_GL']='egl';os.environ['LP_NUM_THREADS']='2';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/pi_attention_shader'
from attention_data import OUT,model
import numpy as np,mujoco as mj,imageio.v2 as imageio
from PIL import Image,ImageDraw
m=model();d=mj.MjData(m);r=mj.Renderer(m,368,480);cam=mj.MjvCamera();cam.distance=1.4;cam.azimuth=135;cam.elevation=-12
trajectories=[np.load(OUT/f'control_{k}_4001.npz')['qpos'] for k in ['direct','filtered']]
w=imageio.get_writer(OUT/'control_comparison.mp4',fps=10,quality=7)
for t in np.arange(0,6,.2):
 frames=[]
 for j,states in enumerate(trajectories):
  idx=min(int(t/.02),len(states)-1);d.qpos[:]=states[idx];mj.mj_forward(m,d);cam.lookat[:]=[d.qpos[0],d.qpos[1],.26]
  r.update_scene(d,cam);r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False;im=Image.fromarray(r.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,480,43),fill=(20,25,30))
  dr.text((8,6),f'{["DIRECT REFERENCE", "ATTENTION FILTER: STANDING FALLBACK"][j]}',fill='white')
  dr.text((8,25),f't={t:.1f}s | 2x playback'+(' | stopped at fall threshold' if j==0 and t>=1.74 else ''),fill='white');frames.append(np.asarray(im))
 w.append_data(np.concatenate(frames,1))
w.close();r.close();print('Rendered closed-loop comparison; fallback is explicitly labelled.')
