from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
names=['build_model.py','experiment.py','zero_identification.py','test_zero_identification.py','replay_zero_sensors.py','summarize_zero.py','render_zero.py','validate_zero_results.py','package_zero.py','ZERO_IDENTIFICATION.md','requirements.txt','environment-lock.txt']
files=[ROOT/n for n in names]+list((ROOT/'zero_results').rglob('*'))+list((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826').rglob('*'))+[ROOT/'upstream/GMR/LICENSE',ROOT/'upstream/README.md',ROOT/'results/provenance.json']
with zipfile.ZipFile(ROOT/'pi_plus_zero_identification.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.writestr('pi_plus_zero/README.md',(ROOT/'ZERO_IDENTIFICATION.md').read_text())
 for p in files:
  if p.is_file():z.write(p,Path('pi_plus_zero')/p.relative_to(ROOT))
print(ROOT/'pi_plus_zero_identification.zip')
