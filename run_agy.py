#!/usr/bin/env python3
"""Headless agy runner for the body-schema gait research task.

Runs `agy -p <brief>` via subprocess argv list (no shell=True, no shell
interpolation of the brief), streaming stdout+stderr to agy-run.log.
Exit code of agy is propagated.
"""
import subprocess
import sys

WORK = '/home/hatch/workspace/research/body-schema-gait'

brief = open(WORK + '/agy-brief-en.txt').read()

cmd = [
    '/home/hatch/.local/bin/agy',
    '--model', 'claude-opus-5-5-medium',
    '--dangerously-skip-permissions',
    '--print-timeout', '1500s',
    '-p', brief,
]

log = open(WORK + '/agy-run.log', 'w', buffering=1)
log.write('[RUNNER] launching agy headless (opus 5.5 one-off, per user order)\n')
log.flush()

p = subprocess.Popen(
    cmd,
    cwd=WORK,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in p.stdout:
    log.write(line)

rc = p.wait()
log.write(f'\n[RUNNER] agy exited with code {rc}\n')
log.close()
sys.exit(rc)
