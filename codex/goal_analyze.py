"""Matched task outcomes and effort comparisons; no savings credited to failures."""
from goal_world import *
import argparse,collections

def bootstrap(values,groups,seed=773,n=4000):
    values=np.asarray(values,float);groups=np.asarray(groups);unique=np.unique(groups)
    if len(unique)<2:return None
    rng=np.random.default_rng(seed);means=[]
    for _ in range(n):
        selected=rng.choice(unique,len(unique),replace=True);sample=np.concatenate([values[groups==u] for u in selected]);means.append(float(sample.mean()))
    return [float(v) for v in np.quantile(means,[.025,.975])]

def summarize(split='test',kind='lift'):
    trials={}
    for p in sorted(OUT.glob(f'control_{split}_*_{kind}.json')):
        rs=json.loads(p.read_text())
        if rs:trials[rs[0]['mode']]=rs
    result={'split':split,'kind':kind,'goal_height_change_mm':25 if kind=='lift' else None,'goal_direction_change_deg':15 if kind=='direction' else None,'method_results':{},'note':'Trials share six passive starts. Confidence intervals resample initial pose/seed groups, not individual time points. Small exploratory sample; one training seed.'}
    for mode,rs in trials.items():
        groups=[r['pose']+'_'+str(r['seed']) for r in rs];success=np.array([r['success'] for r in rs],float)
        bybody={}
        for node in TARGET_NODES:
            sub=[r for r in rs if r['node']==node]
            if sub:bybody[str(node)]={'body':sub[0]['body'],'successes':sum(r['success'] for r in sub),'trials':len(sub),'mean_final_position_error_mm':float(np.mean([r['final_position_error_mm'] for r in sub])),'mean_final_direction_error_deg':float(np.mean([r['final_direction_error_deg'] for r in sub]))}
        result['method_results'][mode]={'trials':len(rs),'successes':int(sum(success)),'success_rate':float(success.mean()),'cluster_bootstrap_success_rate_95_interval':bootstrap(success,groups),'mean_final_position_error_mm':float(np.mean([r['final_position_error_mm'] for r in rs])),'mean_final_direction_error_deg':float(np.mean([r['final_direction_error_deg'] for r in rs])),'mean_effort_all_trials':float(np.mean([r['effort_integral'] for r in rs])),'by_body':bybody}
    if 'learned' in trials and 'efficient' in trials:
        def key(r):return (r['pose'],r['seed'],r['node'])
        lookup={key(r):r for r in trials['learned']};pairs=[(lookup[key(r)],r) for r in trials['efficient'] if key(r) in lookup];both=[(a,b) for a,b in pairs if a['success'] and b['success']]
        deltas=[b['effort_integral']-a['effort_integral'] for a,b in both];ratios=[1-b['effort_integral']/max(a['effort_integral'],1e-9) for a,b in both];groups=[a['pose']+'_'+str(a['seed']) for a,b in both]
        result['matched_effort']={'all_matched_tasks':len(pairs),'both_successful_tasks':len(both),'learned_only_successes':sum(a['success'] and not b['success'] for a,b in pairs),'efficient_only_successes':sum(b['success'] and not a['success'] for a,b in pairs),'mean_effort_difference_efficient_minus_learned':float(np.mean(deltas)) if both else None,'mean_relative_effort_reduction':float(np.mean(ratios)) if both else None,'median_relative_effort_reduction':float(np.median(ratios)) if both else None,'cluster_bootstrap_relative_reduction_95_interval':bootstrap(ratios,groups) if both else None,'scope':'Conditioned on both methods succeeding; not an unconditional energy/success guarantee.'}
    (OUT/f'summary_{split}_{kind}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='method_results'},indent=2))
    for mode,r in result['method_results'].items():print(mode,r['successes'],'/',r['trials'],'err mm',round(r['mean_final_position_error_mm'],2),'dir deg',round(r['mean_final_direction_error_deg'],2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='test');p.add_argument('--kind',default='lift');a=p.parse_args();summarize(a.split,a.kind)
