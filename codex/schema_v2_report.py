"""Publish the complete bounded study, including negative findings."""
from schema_v2_data import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/schema_v2_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/schema_v2_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv,zipfile
from scipy.stats import spearmanr


def generate():
    data=dict(np.load(DEST/'exploration.npz'));manifest=json.loads((DEST/'data_manifest.json').read_text())
    prediction=json.loads((DEST/'prediction_report.json').read_text());control=json.loads((DEST/'control_report.json').read_text())
    composition=json.loads((DEST/'composition_report.json').read_text());checks=json.loads((DEST/'checks.json').read_text())
    curr=prediction['curriculum'];mixed=prediction['mixed'];p=np.load(DEST/'prediction_curriculum.npz');ids=p['ids']
    replay=json.loads((DEST/'replay_report.json').read_text())
    actual_effort=data['cost'][ids,:,0].sum(1);plain=(data['actions'][ids]**2).sum((1,2));w=GoalWorld()
    pd=((data['actions'][ids]*(w.kp*ACTION_LIMIT/w.torque_limit))**2).sum((1,2))
    baseline={'scope':'Post-hoc descriptive baselines; no test-driven training or selection','squared_action_magnitude_spearman':float(spearmanr(plain,actual_effort).statistic),
              'nominal_pd_scaled_action_magnitude_spearman':float(spearmanr(pd,actual_effort).statistic),'learned_spearman':curr['effort_spearman']}
    (DEST/'effort_baselines.json').write_text(json.dumps(baseline,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label,color in [('curriculum','#166aa5'),('mixed','#db7531')]:
        h=prediction[label]['horizons'];axes[0,0].plot([r['seconds'] for r in h],[r['body_point_rmse_mm'] for r in h],'o-',label=label,color=color)
        axes[0,1].plot([r['seconds'] for r in h],[r['whole_com_rmse_mm'] for r in h],'o-',label=label,color=color)
    h=curr['horizons'];times=[r['seconds'] for r in h]
    axes[0,0].plot(times,[r['stationary_body_point_rmse_mm'] for r in h],'--',color='gray',label='stationary baseline')
    axes[0,1].plot(times,[r['stationary_whole_com_rmse_mm'] for r in h],'--',color='gray',label='stationary baseline')
    for ax,title in [(axes[0,0],'All body points'),(axes[0,1],'Whole-body centre of mass')]:
        ax.set(xlabel='Prediction horizon (s)',ylabel='3D RMSE (mm)',title=title);ax.legend();ax.grid(alpha=.2)
    actual=data['cost'][ids,:,0].sum(1);imagined=p['cost'][:,:,0].sum(1)
    axes[1,0].scatter(actual,imagined,c=data['group'][ids],cmap='tab10',s=12,alpha=.65)
    lim=max(actual.max(),imagined.max());axes[1,0].plot([0,lim],[0,lim],'--',color='gray')
    axes[1,0].set(xlabel='Measured normalized torque-squared integral',ylabel='Imagined effort',title=f"Effort ranking: Spearman {curr['effort_spearman']:.3f}")
    groups=['actor','goal','effort'];counts=[control['modes'][m]['successes'] for m in groups]
    axes[1,1].bar(groups,counts,color=['#aaa','#db7531','#166aa5']);axes[1,1].set_ylim(0,16)
    axes[1,1].set(ylabel='Successes / 16 goal cases',title='Physical lift-and-hold test (+15 mm)')
    for i,n in enumerate(counts):axes[1,1].text(i,n+.3,str(n),ha='center')
    fig.suptitle('Pi Plus: staged limb learning and imagined proprioception',fontsize=15)
    fig.savefig(DEST/'study_summary.png',dpi=160);plt.close(fig)

    a=np.load(DEST/'composition.npz');mass=GoalWorld().m.body_mass[GoalWorld().bodies]
    part_nodes=[[0],list(range(13,18)),list(range(18,23)),list(range(1,7)),list(range(7,13)),[23,24]]
    maps=[]
    for key,comkey in [('actual_bodycom','actual_com'),('predicted_bodycom','predicted_com')]:
        b=a[key];interaction=b[:,3]-b[:,1]-b[:,2]+b[:,0];values=[]
        for nodes in part_nodes:
            centre=(interaction[:,nodes]*mass[None,nodes,None]).sum(1)/mass[nodes].sum()
            values.append(np.linalg.norm(centre,axis=-1)*1000)
        b=a[comkey];values.append(np.linalg.norm(b[:,3]-b[:,1]-b[:,2]+b[:,0],axis=-1)*1000)
        values=np.array(values).T.reshape(4,4,7).mean(0);maps.append(values)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');vmax=max(x.max() for x in maps)
    for ax,values,title in zip(axes,maps,['Observed A+B interaction','Imagined A+B interaction']):
        image=ax.imshow(values,cmap='magma',vmin=0,vmax=vmax,aspect='auto')
        ax.set_xticks(range(7),['trunk','L arm','R arm','L leg','R leg','head','whole CoM'],rotation=25)
        ax.set_yticks(range(4),['both arms','both legs','four limbs','diagonal']);ax.set_title(title)
        for y in range(4):
            for x in range(7):ax.text(x,y,f'{values[y,x]:.1f}',ha='center',va='center',color='white' if values[y,x]<vmax*.6 else 'black',fontsize=9)
    fig.colorbar(image,ax=axes,shrink=.8,label='Nonadditive displacement (mm)')
    fig.suptitle('Combination minus A minus B plus hold; 4 initial states per row')
    fig.savefig(DEST/'combination_relations.png',dpi=160);plt.close(fig)
    with (DEST/'control_trials.csv').open('w') as f:
        keys=['pose','seed','node','mode','success','final_window_max_error_mm','final_window_max_speed_m_s','measured_effort','positive_mechanical_work_J']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in control['trials'])

    pairs=control['paired_successful_effort_comparisons'];pair_text='没有两种规划方式都成功的配对，因此没有可报告的同等成功条件下省力证据。'
    if pairs:
        reduction=np.mean([r['fraction_reduction'] for r in pairs])*100
        pair_text=f'两种规划方式都成功的配对有 {len(pairs)} 个；这些配对的归一化力矩平方积分平均变化为减少 {reduction:.1f}%。这只是小样本、局部候选集合中的比较，不代表全局最省力。'
    rows='\n'.join(f"| {r['seconds']:.2f} s | {r['body_point_rmse_mm']:.2f} / {mixed['horizons'][i]['body_point_rmse_mm']:.2f} / {r['stationary_body_point_rmse_mm']:.2f} | {r['whole_com_rmse_mm']:.2f} / {mixed['horizons'][i]['whole_com_rmse_mm']:.2f} / {r['stationary_whole_com_rmse_mm']:.2f} |" for i,r in enumerate(curr['horizons']))
    crows='\n'.join(f"| {label} | {control['modes'][mode]['successes']}/16 | {control['modes'][mode]['median_final_window_error_mm']:.2f} mm |" for mode,label in [('actor','直接目标策略'),('goal','三步想象，优先目标'),('effort','三步想象，增加省力选择')])
    text=f'''# Pi Plus：肢体课程与想象受力实验

已实现“单臂／单腿 → 双臂 → 双腿 → 四肢”的课程、跨部位注意力、预测受力感受、多步运动想象和候选动作的用力选择。实测有用力排序信号，但还没有证明网络掌握了可靠的跨肢体支撑组合关系，也没有证明该课程优于同数据的混合训练。

## 实现逻辑

当前本体观测与候选动作先进入 URDF 身体节点和肢体节点的注意力计算，预测每个关节的平均／RMS 力矩、末端关节负载和接触力，以及每个部位的支撑冲量。预测感受重新编码成 token，通过交叉注意力进入关节、躯干相对运动预测。全身质心由预测外力冲量和牛顿方程推进，再用质量加权的 URDF 几何重建各部位的位置。最后比较候选动作的目标误差和用力代价。

预测感受参与前向计算；训练时从未把真实未来力作为预测输入。真实未来接触和力矩仅用于监督损失。策略也没有向 MuJoCo 查询未来结果来筛选动作。连续想象 3 个 0.32 s 动作，总计 0.96 s；执行第一个动作后重新观测。

课程分阶段训练，共 320 次更新，批大小 96，每阶段 80 次，旧阶段回放比例 35%。阶段按固定训练预算切换，尚未实现按掌握程度自动晋级。混合对照使用相同初始化、完全相同的 {checks['matched_training_presentations']:,} 次样本呈现和更新预算，仅打乱顺序。目标提议策略也按相同阶段顺序训练；最终动作选择包含学习模型搜索，不能将搜索结果当作直接策略网络的能力。

## 内部运动与外部支撑

悬空肢体能改变各部分相对位置、身体姿态和角动量分配；如果其他部位接触地面，还可能间接改变支撑力。全身质心的加速度由重力和外力决定。

独立的无地面接触物理检查中，两组不同关节动作让身体各点的差异达到 {checks['free_flight_internal_motion']['body_point_motion_difference_m']*1000:.2f} mm，但全身质心的差异只有 {checks['free_flight_internal_motion']['whole_com_difference_m']*1000:.3f} mm，属于积分误差量级。这个检查验证仿真力学区别；本网络的全身自由飞行预测能力尚未验证。

## 数据与独立测试

共 {manifest['sequences']:,} 个三动作序列、{manifest['transitions']:,} 段物理运动。训练／验证／测试分别为 900／300／380 个序列，初态分别来自 12／4／4 个不同的种子与姿态组合。同一初态派生的分支不是独立重复试验。各肢体动作与保持动作从完全相同状态配对，能够测量动作对其他部分的影响。

腿部悬空覆盖明显不足：单左腿 40/600、单右腿 25/600 个部位时间段处于悬空；手臂相对充分。这里“悬空”由该部位的实测地面接触时长判定，整个身体仍可能由其他部位支撑。测试还包含训练中未单独出现的左臂＋右腿、右臂＋左腿组合。

外力积分的物理闭合误差：质心速度 RMSE {manifest['impulse_velocity_residual_rmse_m_s']:.6f} m/s，位置 RMSE {manifest['weighted_impulse_position_residual_rmse_m']*1000:.4f} mm。早期把瞬时接触力延用到未来的实现发生预测漂移，已仅依据验证集修正；初版和修订记录保存在 `prototype_v1/` 与实验协议中。独立测试前冻结了最终模型和源文件。

## 预测结果

下表每格依次为 **课程／混合训练／静止预测基线**，均为 3D RMSE（mm）：

| 预测跨度 | 全部身体点 | 全身质心 |
|---|---:|---:|
{rows}

课程模型对三步用力代价的 Spearman 排序相关为 **{curr['effort_spearman']:.3f}**，混合训练为 {mixed['effort_spearman']:.3f}。课程模型的关节 RMS 力矩误差为力矩上限的 {curr['torque_rms_normalized_rmse']*100:.2f}%（汇总 RMSE），实际接触位置的竖直力误差约 {curr['active_contact_fz_rmse_N']:.2f} N。用力代价是归一化力矩平方的时间积分，另记录正机械功；它不是实测电耗、肌肉疲劳或全局最小能耗。

事后加入的描述性简单基线：只按动作幅度平方排序，相关为 {baseline['squared_action_magnitude_spearman']:.3f}；按名义 PD 增益和力矩上限缩放动作幅度，相关为 {baseline['nominal_pd_scaled_action_magnitude_spearman']:.3f}。网络的排序信号强于这两个基线，但还没有转化成稳定的省力控制收益。这些事后比较没有用于重新训练或挑选模型。

质心预测仍未优于静止基线；这套近稳定、小幅运动数据中，预测“基本不动”本身就是强基线。不能只依据毫米级误差声称已学会动力学关系。

对未主动驱动部位的动作影响，课程模型的三步误差为 {curr['inactive_body_action_effect_rmse_mm']:.2f} mm，而“动作对其他部分无额外影响”的基线为 {curr['zero_effect_baseline_rmse_mm']:.2f} mm，尚无优势。

## 交叉组合检验

从同一初态分别执行保持、A、B、A+B，测量 `A+B − A − B + 保持`，提取接触和支撑带来的非线性组合效应。共 {composition['paired_sets']} 组、{composition['physical_sequences']} 条物理序列。

网络预测该组合效应的误差为 **{composition['interaction_prediction_rmse_mm']:.2f} mm**；直接假设两动作效果可相加、没有额外组合效应的误差为 **{composition['zero_interaction_additive_baseline_rmse_mm']:.2f} mm**。本轮不能声称注意力学会了有效的组合推理。

推理时将预测感受反馈置零，三步身体点误差为 {curr['sensory_path_interventions']['zero']['body_point_3step_rmse_mm']:.2f} mm，原模型为 {curr['horizons'][-1]['body_point_rmse_mm']:.2f} mm。说明虽然感受反馈已经接入并能改变预测，但未显示稳定的性能收益。这是推理干预诊断，并非重新训练后的因果消融。

## 物理闭环控制

16 个目标场景：仰卧／俯卧、两个新种子、左右手腕及左右脚，各抬高 15 mm。每种方法运行相同 2.56 s，末尾 0.30 s 内要求高度误差不超过 8 mm、目标点速度不超过 0.04 m/s。

| 方法 | 成功 | 末段最大误差的中位数 |
|---|---:|---:|
{crows}

{pair_text}

省力但未达到目标的尝试不算省力成功。所有失败、候选预测和实际用力均保留。本轮未重新验证完整翻身、坐起或走路，不能将局部抬举或用力排序的结果外推为这些能力。

## 判断与下一步

这轮支持“预测动作的用力程度”可以得到一定信号，也实现了所需的课程与感受预测链路。它没有验证出更强的跨肢体关系和非线性组合能力。进一步审计发现，单独预测的躯干／关节速度所对应的质心速度，与冲量积分预测的质心速度之间 RMSE 为 {replay['predicted_generalized_velocity_vs_impulse_com_velocity_rmse_m_s']:.3f} m/s。当前只约束了质心位置与质量加权几何、质心运动与预测外力之间的关系，尚未让整套预测速度严格一致；这会限制长跨度想象。

优先改进应是：约束广义速度与质心速度一致；补充腿部卸载／悬空和支撑切换数据；让配对动作的因果变化成为训练损失，而不只在评估时计算；按各阶段能力达标决定晋级；增加预测不确定性与接触时序处理。当前使用理想仿真接触力、质心和状态观测，尚未替换成真机估计器。

原有冻结实验文件哈希检查通过；新数据分割、命令掩码、训练呈现一致性、模型质量加权质心一致性及冲量摩擦约束检查通过。详细记录见 [checks.json](schema_v2_results/checks.json)。

全部 {replay['trajectories_replayed']} 条闭环动作均从初态重新执行，状态／电机／控制量的最大重放误差为 {replay['maximum_state_motor_error']:.1f}，成功判定全部重新计算一致。详见 [重放与速度一致性审计](schema_v2_results/replay_report.json)。

## 文件与复现

- [预测与控制图](schema_v2_results/study_summary.png)、[实际／想象组合关系图](schema_v2_results/combination_relations.png)。
- [真实物理控制录像](schema_v2_results/control_examples.mp4)：一个省力规划成功例、同一目标的普通规划尝试和一个失败脚部目标；完整成功率以全部 48 个试验为准。
- [预测报告](schema_v2_results/prediction_report.json)、[组合试验](schema_v2_results/composition_report.json)、[全部闭环试验](schema_v2_results/control_report.json)。
- [协议](SCHEMA_CURRICULUM_PROTOCOL.md)、[数据采集](schema_v2_data.py)、[注意力模型](schema_v2_models.py)、[训练](schema_v2_train.py)、[策略控制](schema_v2_control.py)。

在现有 Pi Plus 项目环境内运行；重跑采集／训练会覆盖对应版本文件，保留冻结记录后应另建实验目录：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_data.py --shard 0
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_data.py --shard 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_data.py --shard 2
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_data.py --merge
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_train.py --label curriculum
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_train.py --label mixed
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_train.py --label policy
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_evaluate.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_composition.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_control.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_checks.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python schema_v2_report.py
```
'''
    (ROOT/'SCHEMA_CURRICULUM_STUDY.md').write_text(text)
    print(ROOT/'SCHEMA_CURRICULUM_STUDY.md',flush=True)

if __name__=='__main__':generate()
