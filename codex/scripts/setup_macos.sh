#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'The pinned runtime supports Apple Silicon. Code, data and results can still be synced on other machines.' >&2
  exit 1
fi
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if (( MACOS_MAJOR < 14 )); then
  printf '%s\n' 'Pinned PyTorch 2.15 requires macOS 14 or newer.' >&2
  exit 1
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  printf '%s\n' 'Python 3.12 is required. Install it locally, then rerun this script.' >&2
  exit 1
fi
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  python3.12 -m venv "$PROJECT_DIR/.venv"
fi
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements-macos.txt"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/verify_snapshot.py"
"$PROJECT_DIR/scripts/run.sh" scripts/smoke_test.py
