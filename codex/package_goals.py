"""Portable study package plus separate complete trace/history archive."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'goal_results';PREFIX='pi_plus_goal_schema/'

def pack():
    top=['build_model.py','experiment.py','ground_schema.py','attention_data.py','zero_identification.py','render_goals.py','resume_goal_tests.py','package_goals.py','requirements.txt','requirements-ground.txt','GOAL_PROTOCOL.md','GOAL_SCHEMA_STUDY.md']+[p.name for p in sorted(ROOT.glob('goal_*.py'))]
    files=[ROOT/p for p in top]
    files+=sorted((ROOT/'upstream/GMR/assets/pi_plus_24dof_250826').rglob('*'))
    files+=[ROOT/'upstream/LICENCE',ROOT/'upstream/GMR/LICENSE']
    files+=[ROOT/'ground_results'/f'getup_replay_{pose}.npz' for pose in ['prone','supine']]
    for p in sorted(OUT.iterdir()):
        if not p.is_file():continue
        if p.suffix in ['.json','.csv','.png','.mp4','.pt','.log']:files.append(p)
        elif p.name=='exploration.npz' or p.name.startswith('prediction_'):files.append(p)
    # Fixed illustrative trial traces used by the report and independent replay checks.
    for mode in ['efficient','learned','kinematic']:
        for node in [24,17,6]:
            for kind in ['lift','direction']:
                p=OUT/f'trial_{mode}_prone_900_{node}_{kind}.npz'
                if p.exists():files.append(p)
    files=sorted(set(p for p in files if p.is_file()));manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    package=ROOT/'pi_plus_goal_schema.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,PREFIX+str(p.relative_to(ROOT)))
        z.write(ROOT/'GOAL_SCHEMA_STUDY.md',PREFIX+'README.md');z.writestr(PREFIX+'MANIFEST.json',json.dumps(manifest,indent=2))
    traces=ROOT/'pi_plus_goal_trajectories.zip';raw=[p for p in OUT.glob('trial_*.npz')]+[p for p in (OUT/'prototype_v1').rglob('*') if p.is_file()]
    with zipfile.ZipFile(traces,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in raw:z.write(p,PREFIX+str(p.relative_to(ROOT)))
        z.writestr(PREFIX+'TRACE_MANIFEST.json',json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw},indent=2))
    print(json.dumps({'main_archive':str(package),'main_megabytes':package.stat().st_size/1e6,'main_files':len(files)+2,'raw_archive':str(traces),'raw_megabytes':traces.stat().st_size/1e6,'raw_files':len(raw)},indent=2))
if __name__=='__main__':pack()
