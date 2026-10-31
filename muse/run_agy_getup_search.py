#!/usr/bin/env python3
"""Headless agy runner for the get-up strategy search task.

Reads the English brief and invokes agy with an argv list (no shell
interpolation of the brief text). Streams stdout/stderr to the log file.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BRIEF = os.path.join(HERE, "agy-brief-getup-search-en.txt")
LOG = os.path.join(HERE, "agy-run-getup-search.log")

def main():
    with open(BRIEF, "r", encoding="utf-8") as f:
        brief = f.read()
    print(f"brief bytes: {len(brief.encode('utf-8'))}", flush=True)
    cmd = [
        "/home/hatch/.local/bin/agy",
        "--model", "claude-opus-5-5-medium",
        "--dangerously-skip-permissions",
        "--print-timeout", "1500s",
        "-p", brief,
    ]
    with open(LOG, "ab") as logf:
        logf.write(f"\n===== run started =====\n".encode())
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line.encode("utf-8", errors="replace"))
            logf.flush()
        rc = proc.wait()
        logf.write(f"===== run finished rc={rc} =====\n".encode())
    print(f"agy exit code: {rc}", flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())
