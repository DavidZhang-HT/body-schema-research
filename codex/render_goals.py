"""Render actual torque-driven traces, including unsuccessful goals, with task markers."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['LP_NUM_THREADS']='2';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/goal_shader'
from goal_world import *
import argparse
import imageio.v2 as imageio
from PIL import Image,ImageDraw

def add_sphere(scene,position,radius,color):
    geom=scene.geoms[scene.ngeom];mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_SPHERE,np.array([radius,0,0]),np.asarray(position),np.eye(3).ravel(),np.array(color));scene.ngeom+=1

def render(mode='efficient',pose='prone',seed=900,kind='lift',nodes=(24,17,6)):
    w=GoalWorld();renderer=mj.Renderer(w.m,320,320);camera=mj.MjvCamera();camera.distance=1.22;camera.azimuth=120;camera.elevation=-35
    records=[]
    for node in nodes:
        stem=f'trial_{mode}_{pose}_{seed}_{node}_{kind}';a=np.load(OUT/f'{stem}.npz');r=json.loads((OUT/f'{stem}.json').read_text());records.append((a,r))
    path=OUT/f'demo_{mode}_{pose}_{seed}_{kind}.mp4';writer=imageio.get_writer(path,fps=25,quality=7)
    for index in range(0,min(len(a['trace']) for a,r in records),2):
        panels=[]
        for a,report in records:
            w.restore(a['trace'][index]);g=Goal(**{k:np.array(v) for k,v in report['goal'].items()});point,direction,_=w.point_state(g.nodes,g.local_points,g.local_axes)
            camera.lookat[:]=[w.d.qpos[0],w.d.qpos[1],.11];renderer.update_scene(w.d,camera);renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype==mj.mjtObj.mjOBJ_GEOM and geom.objid>=0 and int(w.m.geom_bodyid[geom.objid])==w.bodies[int(g.nodes[0])]:geom.rgba[:]=[.05,.95,.55,1]
            add_sphere(renderer.scene,point[0],.009,[.05,1,.55,1])
            if kind=='lift':
                target=point[0].copy();target[1]+=.075;target[2]=g.target_positions[0,2];add_sphere(renderer.scene,target,.009,[1,.2,.05,1])
            else:add_sphere(renderer.scene,point[0]+g.target_directions[0]*.075,.009,[1,.2,.05,1])
            panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel);draw.rectangle((0,0,320,69),fill=(18,24,31));name=report['body'].replace('_link','');status='PASS' if report['success'] else 'FAIL'
            draw.text((7,6),name+' | '+status,fill='white');draw.text((7,23),('Height target +25 mm; x/y free' if kind=='lift' else 'Direction target +15 degrees'),fill='white');draw.text((7,40),f'{mode} | t={index*.02:.2f}s | {pose}',fill='white');draw.text((7,56),'Green: target part / Red: goal',fill='white');panels.append(np.asarray(panel))
        frame=np.concatenate(panels,axis=1);writer.append_data(frame)
        if index>=min(len(a['trace']) for a,r in records)-3:Image.fromarray(frame).save(path.with_suffix('.png'))
    writer.close();renderer.close();print(path,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',default='efficient');p.add_argument('--pose',default='prone');p.add_argument('--seed',type=int,default=900);p.add_argument('--kind',default='lift');a=p.parse_args();render(a.mode,a.pose,a.seed,a.kind)
