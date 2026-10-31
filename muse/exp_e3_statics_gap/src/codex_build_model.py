from pathlib import Path
import xml.etree.ElementTree as E
import mujoco as mj
import json
ROOT=Path(__file__).resolve().parent
(ROOT/'assets').mkdir(exist_ok=True)
(ROOT/'results').mkdir(exist_ok=True)
SRC=ROOT/'upstream/GMR/assets/pi_plus_24dof_250826'
u=E.parse(SRC/'urdf/pi_plus_24dof_250826.urdf')
r=u.getroot(); r.find('mujoco/compiler').set('meshdir',str(SRC/'meshes')); r.find('mujoco/compiler').set('fusestatic','false')
E.SubElement(r,'link',name='world')
j=E.SubElement(r,'joint',name='floating_base',type='floating');E.SubElement(j,'parent',link='world');E.SubElement(j,'child',link='base_link')
for mesh in r.findall('.//mesh'):
    mesh.set('filename', str(SRC/'meshes'/Path(mesh.get('filename')).name))
u.write(ROOT/'assets/source.urdf')
m=mj.MjModel.from_xml_path(str(ROOT/'assets/source.urdf'));mj.mj_saveLastXML(str(ROOT/'assets/converted.xml'),m)
r=E.parse(ROOT/'assets/converted.xml').getroot()
r.find('compiler').set('meshdir',str(SRC/'meshes'))
E.SubElement(r,'option',timestep='0.002',integrator='implicitfast',cone='elliptic',iterations='60')
default=E.SubElement(r,'default');E.SubElement(default,'joint',damping='0.08',armature='0.003');E.SubElement(default,'geom',friction='0.8 0.005 0.0001',solref='0.008 1')
w=r.find('worldbody');E.SubElement(w,'geom',name='ground',type='plane',size='10 10 0.1',rgba='0.22 0.28 0.32 1',contype='1',conaffinity='1')
E.SubElement(w,'light',pos='0 -2 4',dir='0 0 -1',diffuse='0.8 0.8 0.8')
E.SubElement(r,'visual');E.SubElement(r.find('visual'),'global',offwidth='1280',offheight='720')
a=E.SubElement(r,'actuator');limits={j.attrib['name']:float(j.find('limit').attrib['effort']) for j in u.findall('joint') if j.find('limit') is not None}
for j in r.findall('.//joint'):
    if j.get('name') in limits:
        lim=limits[j.get('name')];E.SubElement(a,'motor',name=j.get('name'),joint=j.get('name'),ctrllimited='true',ctrlrange=f'{-lim} {lim}')
E.ElementTree(r).write(ROOT/'assets/scene.xml')
m=mj.MjModel.from_xml_path(str(ROOT/'assets/scene.xml'));d=mj.MjData(m);mj.mj_forward(m,d)
print('model',m.nq,m.nv,m.nu,'mass',sum(m.body_mass))
print([(mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i),int(m.jnt_qposadr[i])) for i in range(m.njnt)])
for side in ['l','r']:
 b=m.body(side+'_ankle_roll_link').id
 print(side,d.xpos[b],[(i,m.geom_type[i],m.geom_pos[i].tolist(),m.geom_size[i].tolist()) for i in range(m.ngeom) if m.geom_bodyid[i]==b])
