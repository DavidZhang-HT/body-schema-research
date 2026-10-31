#!/usr/bin/env python3
"""Second headless agy run: continue the interrupted body-schema gait task."""
import subprocess
import sys

WORK = '/home/hatch/workspace/research/body-schema-gait'
brief = open(WORK + '/agy-brief-continue-en.txt').read()

cmd = [
    '/home/hatch/.local/bin/agy',
    '--model', 'claude-opus-5-5-medium',
    '--dangerously-skip-permissions',
    '--print-timeout', '1500s',
    '-p', brief,
]

log = open(WORK + '/agy-run2.log', 'w', buffering=1)
log.write('[RUNNER2] continuation run after SIGTERM interruption\n')
log.flush()

p = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    log.write(line)

rc = p.wait()
log.write(f'\n[RUNNER2] agy exited with code {rc}\n')
log.close()
sys.exit(rc)
