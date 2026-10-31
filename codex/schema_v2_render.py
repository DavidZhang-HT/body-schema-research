"""Render successful and failed actual force-driven control, not imagined motion."""
import os
os.environ['MUJOCO_GL']='egl';os.environ['LP_NUM_THREADS']='2';os.environ['MESA_SHADER_CACHE_DIR']='/tmp/schema_v2_shader'
from schema_v2_data import *
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont


def render():
    reports=json.loads((DEST/'control_report.json').read_text())['trials']
    good=next(r for r in reports if r['mode']=='effort' and r['success'])
    same=next(r for r in reports if r['mode']=='goal' and all(r[k]==good[k] for k in ['pose','seed','node']))
    bad=next(r for r in reports if r['mode']=='effort' and not r['success'] and r['node'] in [6,12])
    chosen=[same,good,bad];w=SensedWorld();renderer=mj.Renderer(w.m,320,384);cam=mj.MjvCamera();cam.distance=1.15;cam.azimuth=120;cam.elevation=-35
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    datasets=[np.load(DEST/f"control_{r['mode']}_{r['pose']}_{r['seed']}_{r['node']}.npz") for r in chosen]
    writer=imageio.get_writer(DEST/'control_examples.mp4',fps=12.5,quality=7);last=None
    for index in range(0,128,2):
        panels=[]
        for r,a in zip(chosen,datasets):
            w.restore(a['trace'][index]);cam.lookat[:]=[w.d.qpos[0],w.d.qpos[1],.12];renderer.update_scene(w.d,cam);renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
            node=r['node'];target=np.array(r['goal']['target_positions'][0]);point=w.d.xipos[w.bodies[node]]
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype==mj.mjtObj.mjOBJ_GEOM and geom.objid>=0 and w.m.geom_bodyid[geom.objid]==w.bodies[node]:geom.rgba[:]=[.1,.9,.6,1]
            for position,color in [(point,[.1,1,.5,1]),(np.r_[point[:2]+[0,.07],target[2]],[1,.25,.1,1])]:
                geom=renderer.scene.geoms[renderer.scene.ngeom];mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_SPHERE,np.array([.008,0,0]),position,np.eye(3).ravel(),np.array(color));renderer.scene.ngeom+=1
            panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel);draw.rectangle((0,0,384,76),fill=(18,24,31))
            status='PASS' if r['success'] else 'FAIL';name={17:'left wrist',22:'right wrist',6:'left foot',12:'right foot'}[node]
            draw.text((6,4),f"{r['mode']} | {name} | {status}",font=font,fill='white')
            draw.text((6,22),f"{r['pose']} {r['seed']} | t={w.d.time-a['initial'][-1]:.2f}s",font=font,fill='white')
            draw.text((6,40),f"Target +15 mm | total effort {r['measured_effort']:.4f}",font=font,fill='white')
            draw.text((6,58),'Actual physics | playback 0.5x | no base assist',font=font,fill=(210,220,230));panels.append(np.asarray(panel))
        last=np.concatenate(panels,axis=1);writer.append_data(last)
    writer.close();renderer.close();Image.fromarray(last).save(DEST/'control_examples.png')
    (DEST/'video_examples.json').write_text(json.dumps({'scope':'Illustrations selected after scoring; full rates are 0/16 actor, 1/16 goal, 1/16 effort.','cases':[{k:r[k] for k in ['mode','pose','seed','node','success']} for r in chosen]},indent=2))
    print(DEST/'control_examples.mp4')

if __name__=='__main__':render()
