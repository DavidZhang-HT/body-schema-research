from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
names=['build_model.py','experiment.py','zero_identification.py','attention_data.py','attention_train.py','attention_checks.py','attention_control_probe.py','attention_report.py','predict_attention.py','render_attention_probe.py','package_attention.py','ATTENTION_DYNAMICS.md','requirements.txt','requirements-attention.txt']
files=[ROOT/n for n in names]+[p for p in (ROOT/'attention_results').glob('*') if p.is_file()]+list((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826').rglob('*'))+[ROOT/'upstream/GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_walk1_subject1.csv',ROOT/'upstream/GMR/LICENSE',ROOT/'upstream/README.md']
with zipfile.ZipFile(ROOT/'pi_plus_attention_dynamics.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.writestr('pi_plus_attention/README.md',(ROOT/'ATTENTION_DYNAMICS.md').read_text())
 for p in files:
  if p.is_file():z.write(p,Path('pi_plus_attention')/p.relative_to(ROOT))
print(ROOT/'pi_plus_attention_dynamics.zip')
