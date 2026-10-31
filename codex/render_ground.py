import os
os.environ['MUJOCO_GL']='egl';os.environ['LP_NUM_THREADS']='2';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/ground_shader'
from ground_schema import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['discovered','hybrid','learned'],default='discovered');p.add_argument('--seed',type=int,default=101);p.add_argument('--poses',nargs='+',default=['prone','supine']);args=p.parse_args()
w=GroundWorld();renderer=mj.Renderer(w.m,400,640);cam=mj.MjvCamera();cam.distance=1.8;cam.azimuth=125;cam.elevation=-15
for pose in args.poses:
 if args.mode=='discovered':
  path=OUT/f'getup_replay_{pose}.npz';name=f'getup_{pose}';title='PHYSICS-SEARCH DISCOVERED SCHEMAS + TORQUE BALANCE';note='No external lift. This video is NOT neural-only planning.'
 elif args.mode=='hybrid':
  path=OUT/f'hybrid_{pose}_{args.seed}_replay.npz';name=f'hybrid_getup_{pose}_{args.seed}';title='CONTACT ATTENTION + PHYSICS VALIDATION + TORQUE BALANCE';note='New passive start. Explicit physics oracle is used in planning.'
 else:
  path=OUT/f'learned_{pose}_{args.seed}.npz';name=f'learned_attempt_{pose}_{args.seed}';title='LEARNED CONTACT-ATTENTION CEM: GET-UP ATTEMPT';note='No physics oracle in action selection. Attempt did not achieve standing.'
 a=np.load(path);states=a['trace'];t0=states[0,-1];writer=imageio.get_writer(OUT/f'{name}.mp4',fps=12,quality=7)
 for idx in range(0,len(states),10):
  w.restore(states[idx]);cam.lookat[:]=[w.d.qpos[0],w.d.qpos[1],.29];renderer.update_scene(w.d,cam);renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
  im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,59),fill=(20,25,30));t=states[idx,-1]-t0
  draw.text((10,6),f'Pi Plus | {pose} get-up trial | t={t:.1f}s | playback 2.4x',fill='white');draw.text((10,24),title,fill='white');draw.text((10,42),note,fill='white')
  writer.append_data(np.asarray(im))
  if idx>=len(states)-11:im.save(OUT/f'{name}_final.png')
 writer.close();print('rendered',pose,flush=True)
renderer.close()
