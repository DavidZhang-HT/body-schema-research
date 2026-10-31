# Unified body-region goal learning: frozen initial experiment protocol

Created before data collection and training. This is an exploratory simulation study, not a preregistered public clinical/statistical study.

Question: Can one attention-based model and one goal interface move any selected Pi Plus body point/direction under ground contact, and select lower-effort solutions at matched achievement?

The same interface is used for base, head, both wrists and both feet. No body-name-specific control branches or joint-angle target demonstrations. URDF geometry and joint ancestry are allowed priors. Full 24-joint bounded action space. Ideal simulation proprioception and contact measurements; no external base assistance during execution.

Hypotheses (outcomes may be negative):
1. Action-conditioned dynamics improve body-point predictions over fixed-base kinematics on held-out passive starts.
2. One goal-conditioned attention policy/planner reduces target errors for head, hands and feet through the same code path; all body groups are reported.
3. Adding a measured effort objective reduces normalized torque-squared integral among matched successful trials; goal failure cannot count as energy saving.
4. Removing contact information through a separately trained model may reduce predictive/control performance. No causal benefit of attention itself is claimed without a matched non-attention model.

Initial settings: 0.32-second smooth bounded joint-target increments, 2 ms physics. Position goals are +25 mm world height; direction goals rotate a chosen body axis by 15 degrees toward world up. Goals have position/direction masks and a reference frame. Evaluation uses fixed time budgets. Main position success requires final-window height error <= 8 mm and speed <= 0.04 m/s; report the exact settling-window implementation. Direction tolerance 5 degrees. No test threshold will be weakened to obtain a success.

Training starts: supine/prone seeds 200–211. Validation: seeds 212–214. Prediction test: seeds 215–217. Closed-loop test: supine/prone seeds 900,901,902. Test sets must not be used to select checkpoints. Additional existing successful get-up states, if used, are training-only and disclosed. Goal-conditioned hindsight labels come from achieved physical states, not future-state observations at inference.

Controls: zero increment; nominal URDF kinematics with fixed floating base; learned imagination without effort objective; learned imagination with effort objective. A direct goal-policy result is also reported. Learned action selection must not query future MuJoCo rollouts; observations and execution use MuJoCo. If physics validation is added in a separate experiment, label it explicitly and do not pool outcomes.

Effort: integrate sum((actual torque / URDF torque limit)^2) dt; separately measure positive mechanical work, peak torque fraction, limit penetration and speed-limit exceedance. Report effort only at the same execution duration, and paired successful comparisons. These are mechanical/actuator-load proxies, not measured electrical energy.

Retain failed trials, validation curves, data split, source hashes, model checkpoints and full trajectories. Any protocol changes after inspection go into a dated amendments section and remain distinguishable from initial settings. No global optimality, real-world transfer, or arbitrary-pose get-up claim.

## Amendment before closed-loop test seeds were run

Validation on seed 212 showed that the first policy nearly ignored goal changes (action sensitivity RMS 0.00185), and several imagined-control trials approached the point but failed the predeclared settling requirement. Prototype results and source are retained in `goal_results/prototype_v1/`.

The inverse policy now uses explicit goal-to-body cross attention. Forward dynamics and its prediction split remain unchanged. Execution is fixed at 10 blocks (3.2 s): 8 goal-control blocks followed by 2 common joint-hold blocks (0.64 s); all methods use the same holding rule. The original final-window 8 mm / 5 degree / 0.04 m/s success thresholds remain unchanged. This implements arrival-and-hold rather than weakening the settling criterion. No closed-loop test seed 900–902 was examined before these changes.
