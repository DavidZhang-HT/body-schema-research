"""Restart-safe two-worker sweep launcher; workers are isolated Python processes."""
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sweep(selection, after_robust=False):
    out = ROOT/'combo/results'
    if after_robust:
        while not all((out/f'{pose}_{seed}.json').exists()
                      for pose in ('supine','prone') for seed in (0,1,2)):
            time.sleep(10)
    destination = out if selection == 'robust' else out/selection
    destination.mkdir(parents=True, exist_ok=True)

    def pose_queue(pose):
        for seed in (0,1,2):
            with (destination/f'{pose}_{seed}.log').open('a') as log:
                subprocess.run([sys.executable,'-m','combo.search','--pose',pose,
                                '--seed',str(seed),'--selection',selection],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(selection, pose, seed, 'complete', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(pose_queue, pose) for pose in ('supine','prone')]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection', choices=['robust','dynamic'], default='dynamic')
    p.add_argument('--after-robust', action='store_true')
    a = p.parse_args()
    sweep(a.selection,a.after_robust)
