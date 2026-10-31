from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
with zipfile.ZipFile(ROOT/'pi_plus_experiment.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    files=list(ROOT.glob('*.py'))+[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'environment-lock.txt']
    files+=list((ROOT/'results').glob('*.json'))+list((ROOT/'results').glob('*.npz'))+list((ROOT/'results').glob('*.csv'))+list((ROOT/'results').glob('*.png'))
    files+=list((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826').rglob('*'))
    files+=[ROOT/'upstream/GMR/RetargetData/lafan1/csv/pi_plus/pi_plus_walk1_subject1.csv',ROOT/'upstream/GMR/MotionData/lafan1/walk1_subject1.bvh',ROOT/'upstream/GMR/LICENSE',ROOT/'upstream/README.md']
    for p in files:
        if p.is_file():z.write(p,Path('pi_plus_experiment')/p.relative_to(ROOT))
print('Packaged source, required model/motion subset, logs and figures; videos supplied separately.')
