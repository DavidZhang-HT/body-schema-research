import subprocess, sys

brief = open('/home/hatch/workspace/research/body-schema-gait/agy-brief-q0-torque-resume-en.txt').read()
cmd = ['/home/hatch/.local/bin/agy', '--model', 'claude-opus-5-5-medium',
       '--dangerously-skip-permissions', '--print-timeout', '1500s', '-p', brief]
log_path = '/home/hatch/workspace/research/body-schema-gait/agy-run-q0-torque-resume.log'
with open(log_path, 'w') as log:
    proc = subprocess.Popen(cmd, cwd='/home/hatch/workspace/research/body-schema-gait',
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        log.flush()
    rc = proc.wait()
print(f"agy exited with code {rc}", flush=True)
sys.exit(rc)
