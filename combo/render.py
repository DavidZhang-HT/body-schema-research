"""CPU headless front/side trajectory render, 12 fps, explicitly labeled 2x playback."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['LP_NUM_THREADS'] = '1'
import argparse
import fcntl
import json
import subprocess
from PIL import Image, ImageDraw
from .plant import GroundWorld, gs, np, ROOT
from .modes import classify


def render(pose, seed, selection="dynamic"):
    out = ROOT/'combo/results'
    if selection != 'robust':
        out = out / selection
    tag = f'{pose}_{seed}'
    lock = (out/f'{tag}.render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    movie = out/f'{tag}_getup.mp4'
    if movie.exists():
        print('Existing completed movie:', movie, flush=True)
        lock.close()
        return
    report = json.loads((out/f'{tag}_explanation.json').read_text())
    states = np.load(out/f'{tag}_trace.npz')['trace']
    w = GroundWorld()
    renderer = gs.mj.Renderer(w.m, height=384, width=512)
    cams = []
    w.restore(states[-1])
    rot = w.d.xmat[w.base].reshape(3, 3)
    heading = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    w.restore(states[0])
    # Orthogonal viewpoints relative to final torso heading; follow translation.
    for azimuth in (heading, heading+90):
        cam = gs.mj.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.65, azimuth, -15
        cams.append(cam)
    times = states[:, -1]
    indices = np.searchsorted(times, np.arange(times[0], times[-1]+1e-6, 2/12))
    movie = out/f'{tag}_getup.mp4'
    temporary_movie = movie.with_name(movie.stem+'.tmp.mp4')
    writer = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
                               '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x384',
                               '-r','12','-i','-','-an','-vcodec','libx264','-crf','20',
                               '-pix_fmt','yuv420p','-threads','1',str(temporary_movie)],
                              stdin=subprocess.PIPE)
    try:
        for frame_number, idx in enumerate(np.minimum(indices, len(states)-1)):
            w.restore(states[idx])
            views = []
            for cam in cams:
                cam.lookat[:] = [w.d.qpos[0], w.d.qpos[1], .25]
                renderer.update_scene(w.d, camera=cam)
                renderer.scene.flags[gs.mj.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[gs.mj.mjtRndFlag.mjRND_REFLECTION] = False
                views.append(renderer.render().copy())
            im = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(im)
            draw.rectangle((0,0,1024,52), fill=(20,25,30))
            draw.text((10,5), f'24-DoF pi+ | {pose}, seed {seed} | t={times[idx]-times[0]:.2f}s | {classify(w)}', fill='white')
            draw.text((10,22), 'Physics search (16 synergies), then unchanged QP balance | no network | 2x playback', fill='white')
            draw.text((10,37), f'Replay sustained stand: {report["replay_success"]} | front / side', fill='white')
            writer.stdin.write(np.asarray(im).tobytes())
            if frame_number % 50 == 0:
                print("render frame", frame_number, "of", len(indices), flush=True)
        im.save(out/f'{tag}_final.png')
    finally:
        writer.stdin.close()
        returncode = writer.wait()
        renderer.close()
        if returncode:
            raise RuntimeError(f"ffmpeg exited {returncode}")
    temporary_movie.replace(movie)
    lock.close()
    print(movie, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pose', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument("--selection", choices=["robust", "dynamic"], default="dynamic")
    a = p.parse_args()
    render(a.pose,a.seed,a.selection)
