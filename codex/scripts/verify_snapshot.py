"""Verify published files and experimental freezes without third-party packages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(mapping):
    for relative, expected in mapping.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise SystemExit(f'Invalid manifest path: {relative}')
        if not path.is_file():
            raise SystemExit(f'Missing project file: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'File differs from published snapshot: {relative}')


def main():
    manifest = json.loads((ROOT / 'PROJECT_SNAPSHOT.json').read_text())
    verify(manifest['sha256'])
    verify(json.loads((ROOT / 'goal_results/test_freeze.json').read_text())['source_and_checkpoint_sha256'])
    verify(json.loads((ROOT / 'schema_v2_results/prediction_freeze.json').read_text())['sha256'])
    verify(json.loads((ROOT / 'schema_v2_results/control_freeze.json').read_text()))
    print(f"Verified {len(manifest['sha256'])} published files and original experiment freezes.")


if __name__ == '__main__':
    main()
