"""Build a portable source, assets, data, weights and results package."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parent
prefix='pi_plus_ground_schema/'
code=['build_model.py','experiment.py','attention_data.py','zero_identification.py','render_ground.py','predict_ground.py','summarize_ground.py','package_ground.py','requirements.txt','requirements-ground.txt','GROUND_GETUP.md']
code += [p.name for p in sorted(ROOT.glob('ground_*.py'))]
files=[ROOT/p for p in code]
files += sorted((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826').rglob('*'))
files += [ROOT/'upstream/LICENCE',ROOT/'upstream/GMR/LICENSE']
files += [p for p in sorted((ROOT/'ground_results').iterdir()) if p.is_file() and 'refine' not in p.name and 'candidate' not in p.name and not p.name.startswith('balance_')]
files=[p for p in files if p.is_file()]
manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
path=ROOT/'pi_plus_ground_schema.zip'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,prefix+str(p.relative_to(ROOT)))
 z.write(ROOT/'GROUND_GETUP.md',prefix+'README.md')
 z.writestr(prefix+'MANIFEST.json',json.dumps(manifest,indent=2))
print(json.dumps({'archive':str(path),'files':len(files)+2,'size_mb':path.stat().st_size/1e6},indent=2))
