import os
os.environ['MUJOCO_GL']='egl';os.environ['LP_NUM_THREADS']='2';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/pi_zero_shader'
from zero_identification import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw
name='active_seed7';a=np.load(OUT/f'{name}.npz');m=model();d=mj.MjData(m)
r=mj.Renderer(m,368,480);cam=mj.MjvCamera();cam.distance=1.35;cam.azimuth=135;cam.elevation=-12
opt=mj.MjvOption()
w=imageio.get_writer(OUT/'active_zero_identification.mp4',fps=8,quality=7)
for idx in range(0,len(a['qpos']),100):
 d.qpos[:]=a['qpos'][idx];mj.mj_forward(m,d);cam.lookat[:]=[0,0,.26]
 r.update_scene(d,cam,scene_option=opt);r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
 im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);t,block=a['log'][idx,:2]
 stage='Initial stand' if block==0 else ('ACTIVE PROBE' if block<=7 else 'HELD-OUT VALIDATION')
 draw.rectangle((0,0,480,44),fill=(20,25,30));draw.text((8,6),f'Pi Plus | Encoder-zero ID | t={t:.1f}s | playback 3.2x',fill='white');draw.text((8,25),f'{stage} | block {int(block)} | torque-only physics',fill='white')
 w.append_data(np.asarray(im))
 if 6<t<6.5:im.save(OUT/'active_zero_preview.png')
w.close();r.close();print('Rendered zero-identification physics rollout.',flush=True)
