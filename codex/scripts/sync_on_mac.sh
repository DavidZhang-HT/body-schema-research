#!/usr/bin/env bash
# Execute on the Mac; does not configure remote access or run training.
set -euo pipefail
REPOSITORY="${1:-DavidZhang-HT/body-schema-research}"
DESTINATION="${2:-$HOME/Projects/body-schema-research}"
if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  printf '%s\n' 'Repository must be owner/name.' >&2; exit 1
fi
if [[ "$(uname -s)" != Darwin ]]; then
  printf '%s\n' 'Run this synchronization script on the Mac.' >&2; exit 1
fi
if [[ -L "$DESTINATION" ]]; then
  printf '%s\n' 'Destination is a symbolic link; choose an explicit project directory.' >&2; exit 1
fi
if [[ -e "$DESTINATION" ]]; then
  if [[ ! -d "$DESTINATION/.git" ]]; then
    printf '%s\n' 'Existing destination is not a standalone Git checkout; left unchanged.' >&2; exit 1
  fi
  REMOTE="$(git -C "$DESTINATION" remote get-url origin)"
  case "$REMOTE" in
    "https://github.com/$REPOSITORY"|"https://github.com/$REPOSITORY.git"|"git@github.com:$REPOSITORY.git") ;;
    *) printf '%s\n' 'Destination belongs to another repository; left unchanged.' >&2; exit 1 ;;
  esac
  if [[ -n "$(git -C "$DESTINATION" status --porcelain)" ]]; then
    printf '%s\n' 'Local changes exist. Preserve or commit them first; no files were reset.' >&2; exit 1
  fi
  if [[ "$(git -C "$DESTINATION" branch --show-current)" != main ]]; then
    printf '%s\n' 'Checkout is not on main; no branch switch was performed.' >&2; exit 1
  fi
  git -C "$DESTINATION" pull --ff-only origin main
else
  mkdir -p -- "$(dirname -- "$DESTINATION")"
  git clone "https://github.com/$REPOSITORY.git" "$DESTINATION"
fi
if command -v python3 >/dev/null 2>&1; then
  python3 "$DESTINATION/codex/scripts/verify_snapshot.py"
fi
printf 'Synced repository: %s\nProject: %s/codex\nCommit: ' "$DESTINATION" "$DESTINATION"
git -C "$DESTINATION" rev-parse HEAD
