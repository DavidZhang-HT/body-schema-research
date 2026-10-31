"""Run frozen experiment sources with platform-appropriate MuJoCo rendering."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ['MUJOCO_GL'] = 'cgl' if sys.platform == 'darwin' else os.environ.get('MUJOCO_GL', 'egl')
# Preload the native context before frozen render scripts set their old EGL flag.
import mujoco
from mujoco.rendering.classic import gl_context

if len(sys.argv) < 2:
    raise SystemExit('Usage: scripts/run.sh experiment_script.py [arguments]')
target = (ROOT / sys.argv[1]).resolve()
if not target.is_file() or not target.is_relative_to(ROOT):
    raise SystemExit('Choose a Python script inside this project.')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.argv = [str(target), *sys.argv[2:]]
runpy.run_path(str(target), run_name='__main__')
