"""Export scientific figures from measured physics, including controls and failures."""
import os
os.environ['MPLCONFIGDIR']='/tmp/goal_plot_cache'
from goal_world import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={0:'Pelvis/base',24:'Head',17:'Left hand',22:'Right hand',6:'Left foot',12:'Right foot'}
MODES=['hold','kinematic','policy','blind','learned','efficient']

def report():
    full=json.loads((OUT/'prediction_full.json').read_text());blind=json.loads((OUT/'prediction_blind.json').read_text());fig,axes=plt.subplots(2,2,figsize=(13,9));ids=TARGET_NODES;x=np.arange(len(ids))
    for offset,(name,record,key) in enumerate([('Kinematic',full,'fixed_base_kinematic_rmse_mm'),('Attention + contacts',full,'learned_point_rmse_mm'),('No explicit contacts',blind,'learned_point_rmse_mm')]):
        axes[0,0].bar(x+(offset-1)*.25,[record['by_body_node'][str(n)][key] for n in ids],.25,label=name)
    axes[0,0].set_xticks(x,[LABELS[n] for n in ids],rotation=20);axes[0,0].set_ylabel('0.32 s point-position RMSE (mm)');axes[0,0].set_title('Prediction on six held-out passive starts');axes[0,0].legend(fontsize=8)
    for kind,style in [('lift','o-'),('direction','s--')]:
        path=OUT/f'summary_test_{kind}.json'
        if path.exists():
            r=json.loads(path.read_text())['method_results'];names=[m for m in MODES if m in r];positions=[MODES.index(m) for m in names]
            axes[0,1].plot(positions,[100*r[m]['success_rate'] for m in names],style,label=kind)
    axes[0,1].set_xticks(range(len(MODES)),MODES,rotation=25);axes[0,1].set_ylim(-3,103);axes[0,1].set_ylabel('Arrival + settling success (%)');axes[0,1].set_title('Same goals, starts, and 3.2 s duration');axes[0,1].legend()
    paths=[OUT/f'control_test_{m}_lift.json' for m in ['learned','efficient']]
    if all(p.exists() for p in paths):
        a,b=[json.loads(p.read_text()) for p in paths];lookup={(r['pose'],r['seed'],r['node']):r for r in a};limits=[]
        for r in b:
            first=lookup[(r['pose'],r['seed'],r['node'])];okay=first['success'] and r['success'];color='tab:green' if okay else 'tab:gray';axes[1,0].scatter(first['effort_integral'],r['effort_integral'],color=color,marker='o' if okay else 'x',alpha=.7);limits.extend([first['effort_integral'],r['effort_integral']])
        maximum=max(limits)*1.05;axes[1,0].plot([0,maximum],[0,maximum],'k--');axes[1,0].set(xlabel='No effort objective: measured torque-squared integral',ylabel='Effort objective: measured torque-squared integral',title='Green: both succeed; gray: excluded from savings claim')
    path=OUT/'summary_test_lift.json'
    if path.exists():
        summary=json.loads(path.read_text())['method_results']
        for k,mode in enumerate(['kinematic','learned','efficient']):
            if mode in summary:axes[1,1].bar(x+(k-1)*.25,[summary[mode]['by_body'][str(n)]['successes'] for n in ids],.25,label=mode)
        axes[1,1].set_xticks(x,[LABELS[n] for n in ids],rotation=20);axes[1,1].set_ylim(0,6.5);axes[1,1].set_ylabel('Lift successes out of six starts');axes[1,1].set_title('Every body group is reported');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'study_summary.png',dpi=170);plt.close(fig)
    # Fixed seed/pose examples, chosen before looking for successful videos.
    fig,axes=plt.subplots(3,3,figsize=(13,10));world=GoalWorld()
    for col,node in enumerate([24,17,6]):
        for mode in ['kinematic','learned','efficient']:
            stem=f'trial_{mode}_prone_900_{node}_lift';path=OUT/f'{stem}.npz'
            if not path.exists():continue
            a=np.load(path);r=json.loads((OUT/f'{stem}.json').read_text());trace=a['task_trace'];delta=(trace[:,6]-r['initial_point'][0][2])*1000
            axes[0,col].plot(trace[:,0],delta,label=mode);axes[2,col].plot(np.arange(11)*DURATION,np.r_[0,np.cumsum(a['costs'][:,0])],label=mode)
        axes[0,col].axhline(25,color='k',ls=':');axes[0,col].axhspan(17,33,color='green',alpha=.08);axes[0,col].set_title(LABELS[node]+' | prone seed 900');axes[0,col].set_ylabel('Observed height change (mm)')
        path=OUT/f'trial_efficient_prone_900_{node}_lift.npz'
        if path.exists():
            a=np.load(path);t=[];forces=[]
            for snap in a['trace'][::4]:
                world.restore(snap);_,_,weight,_=world.contacts();t.append(snap[-1]-a['initial'][-1]);forces.append([weight[[6,12]].sum(),weight[[16,17,21,22]].sum(),weight.sum()-weight[[6,12,16,17,21,22]].sum()])
            forces=np.array(forces)
            for i,label in enumerate(['Feet','Elbows/wrists','Other support']):axes[1,col].plot(t,forces[:,i],label=label)
        axes[1,col].set_ylabel('Measured support force Fz (N)');axes[2,col].set_ylabel('Measured effort integral');axes[2,col].set_xlabel('Actual physics time (s)')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'body_goal_contact_traces.png',dpi=160);plt.close(fig);print('Saved scientific figures.')
if __name__=='__main__':report()
