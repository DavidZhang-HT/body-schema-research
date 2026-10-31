"""Headless agy runner for the q0 (joint zero offset) active-ID task.

Invokes agy via subprocess argv list (no shell, no quoting hazards), streaming
stdout/stderr to agy-run-q0.log in the workflow directory.
"""
import subprocess, sys, os

WORK = "/home/hatch/workspace/research/body-schema-gait"
BRIEF = os.path.join(WORK, "agy-brief-q0-en.txt")
LOG = os.path.join(WORK, "agy-run-q0.log")

with open(BRIEF, "r") as f:
    brief = f.read()

cmd = [
    "/home/hatch/.local/bin/agy",
    "--model", "claude-opus-5-5-medium",
    "--dangerously-skip-permissions",
    "--print-timeout", "1500s",
    "-p", brief,
]

with open(LOG, "w") as log:
    log.write("=== agy q0 run: cmd argv (brief passed as argv, not shell) ===\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=WORK,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log.write(line)
        log.flush()
    rc = proc.wait()

with open(LOG, "a") as log:
    log.write(f"\n=== agy exited with rc={rc} ===\n")

print(f"done rc={rc} log={LOG}")
sys.exit(rc)
