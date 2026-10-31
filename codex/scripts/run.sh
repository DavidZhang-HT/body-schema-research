#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  printf '%s\n' 'Project virtual environment is missing. On Apple Silicon, run scripts/setup_macos.sh first.' >&2
  exit 1
fi
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/run_python.py" "$@"
