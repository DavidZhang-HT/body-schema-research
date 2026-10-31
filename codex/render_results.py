"""Render saved states from actual physics rollouts, never used for control."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/mesa_cache';os.environ['LP_NUM_THREADS']='2'
from experiment import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['direct','projected','adapted']);args=p.parse_args()
for mode in args.modes:
    a=np.load(ROOT/f'results/{mode}.npz');m=model();d=mj.MjData(m)
    renderer=mj.Renderer(m,360,480);camera=mj.MjvCamera();camera.distance=1.45;camera.azimuth=135;camera.elevation=-12
    writer=imageio.get_writer(ROOT/f'results/{mode}.mp4',fps=15,quality=7)
    inds=np.linspace(0,len(a['qpos'])-1,int(a['log'][-1,0]*15)+1).astype(int)
    for ii,i in enumerate(inds):
        d.qpos[:]=a['qpos'][i];mj.mj_forward(m,d);camera.lookat[:]=[d.qpos[0],d.qpos[1],.27]
        renderer.update_scene(d,camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        t=a['log'][i,0];draw.rectangle((0,0,480,42),fill=(20,25,30));draw.text((10,6),f'Pi Plus | {mode} | PHYSICS ROLLOUT | t={t:.2f}s',fill='white');draw.text((10,23),'Calibration 0-6s | Transition 6-8s | Reference after 8s',fill='white')
        writer.append_data(np.asarray(im))
        if ii==len(inds)//2:im.save(ROOT/f'results/{mode}.png')
    writer.close();renderer.close();print('rendered',mode,flush=True)
