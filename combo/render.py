"""CPU headless front/side trajectory render, real-time playback at 25 fps."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['LP_NUM_THREADS'] = '1'
import argparse
import json
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from .plant import GroundWorld, gs, np, ROOT
from .modes import classify


def render(pose, seed, selection="robust"):
    out = ROOT/'combo/results'
    if selection != 'robust':
        out = out / selection
    tag = f'{pose}_{seed}'
    report = json.loads((out/f'{tag}_explanation.json').read_text())
    states = np.load(out/f'{tag}_trace.npz')['trace']
    w = GroundWorld()
    renderer = gs.mj.Renderer(w.m, height=384, width=512)
    cams = []
    w.restore(states[0])
    # Fixed world viewpoints show the whole motion; recenter follows translation.
    for azimuth in (90, 0):
        cam = gs.mj.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.65, azimuth, -15
        cams.append(cam)
    times = states[:, -1]
    indices = np.searchsorted(times, np.arange(times[0], times[-1]+1e-6, .04))
    writer = imageio.get_writer(out/f'{tag}_getup.mp4', fps=25, quality=7,
                                ffmpeg_params=['-threads','1'])
    try:
        for idx in np.minimum(indices, len(states)-1):
            w.restore(states[idx])
            views = []
            for cam in cams:
                cam.lookat[:] = [w.d.qpos[0], w.d.qpos[1], .25]
                renderer.update_scene(w.d, camera=cam)
                views.append(renderer.render().copy())
            im = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,1024,52), fill=(20,25,30))
            draw.text((10,5), f'24-DoF pi+ | {pose}, seed {seed} | t={times[idx]-times[0]:.2f}s | {classify(w)}', fill='white')
            draw.text((10,22), 'Physics search (16 synergies), then unchanged QP balance | no network | 1x playback', fill='white')
            draw.text((10,37), f'Replay sustained stand: {report["replay_success"]} | front / side', fill='white')
            writer.append_data(np.asarray(im))
        im.save(out/f'{tag}_final.png')
    finally:
        writer.close()
        renderer.close()
    print(out/f'{tag}_getup.mp4', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pose', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument("--selection", choices=["robust", "dynamic"], default="robust")
    a = p.parse_args()
    render(a.pose,a.seed,a.selection)
