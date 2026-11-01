"""Sequential replay/certification and CPU movie generation after both sweeps."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(wait=False):
    base = ROOT/'combo/results'
    expected = [base/folder/f'{pose}_{seed}.json'
                for folder in ('','dynamic') for pose in ('supine','prone') for seed in (0,1,2)]
    while not all(p.exists() for p in expected):
        if not wait:
            raise RuntimeError('Both evaluation sweeps must finish before artifacts')
        time.sleep(10)
    for selection in ('robust','dynamic'):
        out = base if selection == 'robust' else base/selection
        for pose in ('supine','prone'):
            for seed in (0,1,2):
                tag = f'{pose}_{seed}'
                if not ((out/f'{tag}_explanation.json').exists() and (out/f'{tag}_trace.npz').exists()):
                    subprocess.run([sys.executable,'-m','combo.explain','--pose',pose,'--seed',str(seed),
                                    '--selection',selection],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','combo.summarize','--selection',selection],cwd=ROOT,check=True)
    successes = []
    for p in sorted((base/'dynamic').glob('supine_*_explanation.json')):
        r = json.loads(p.read_text())
        if r['replay_success']:
            successes.append(r)
    if not successes:
        raise RuntimeError('No independently replayed supine success to render')
    r = successes[0]
    movie = base/'dynamic'/f'supine_{r["seed"]}_getup.mp4'
    if not movie.exists():
        subprocess.run(['xvfb-run','-a',sys.executable,'-m','combo.render','--pose','supine',
                        '--seed',str(r['seed']),'--selection','dynamic'],cwd=ROOT,check=True,
                       env=dict(os.environ,MUJOCO_GL='glfw'))
    metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format',
                                                   '-show_streams','-of','json',str(movie)],text=True))
    metadata['sha256'] = hashlib.sha256(movie.read_bytes()).hexdigest()
    metadata['path'] = str(movie.relative_to(ROOT))
    (base/'dynamic/video_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Artifacts complete:',movie,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait',action='store_true')
    build(p.parse_args().wait)
