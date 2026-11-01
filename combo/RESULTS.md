# Measured outcomes

| Start | Seed | Search/QP success | Replay success | Evals | Blocks | Wall s |
|---|---:|---|---|---:|---:|---:|
| supine | 0 | False | False | 6000 | 18 | 447.5 |
| supine | 1 | False | False | 6000 | 20 | 479.4 |
| supine | 2 | False | False | 6000 | 19 | 428.2 |
| prone | 0 | False | False | 6000 | 17 | 470.5 |
| prone | 1 | False | False | 6000 | 17 | 477.4 |
| prone | 2 | False | False | 6000 | 16 | 463.7 |

supine: 0/3 completed search runs succeeded.

prone: 0/3 completed search runs succeeded.

## supine / 0

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: base_height=0.1013 (requires >0.16); foot_load=0.0000 (requires >75); nonfoot_load=106.8747 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: SUPINE → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → TRANSITION → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → TRANSITION → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.074 / l_shoulder_pitch_joint | trunk:0.57, L_shank:0.29, R_shank:0.14 |
| 1 | SUPINE → SIT | DYNAMIC_UNRESOLVED | True | 0.085 / l_hip_pitch_joint | trunk:0.40, L_thigh:0.21, L_shank:0.01, L_foot:0.04, R_thigh:0.22, R_shank:0.02, R_foot:0.09 |
| 2 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.091 / l_hip_roll_joint | L_thigh:0.41, L_foot:0.12, R_thigh:0.47 |
| 3 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.119 / l_hip_roll_joint | trunk:0.26, L_thigh:0.14, L_foot:0.17, R_thigh:0.43 |
| 4 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.052 / l_hip_pitch_joint | trunk:0.12, L_thigh:0.30, L_ankle:0.06, L_foot:0.05, R_thigh:0.45, L_hand:0.02 |
| 5 | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.074 / r_shoulder_pitch_joint | L_thigh:0.35, L_shank:0.06, L_foot:0.10, R_thigh:0.29, R_shank:0.20 |
| 6 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.078 / l_shoulder_pitch_joint | L_thigh:0.47, L_shank:0.06, R_thigh:0.29, R_hand:0.18 |
| 7 | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.223 / l_hip_pitch_joint | L_thigh:0.26, L_foot:0.27, R_thigh:0.48 |
| 8 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.192 / l_hip_pitch_joint | L_thigh:0.22, L_ankle:0.06, L_foot:0.22, R_thigh:0.50 |
| 9 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.215 / l_hip_pitch_joint | L_thigh:0.22, L_foot:0.28, R_thigh:0.50 |
| 10 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.187 / l_shoulder_roll_joint | L_thigh:0.37, L_ankle:0.09, L_hand:0.18, R_hand:0.36 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.025 / l_calf_joint | L_thigh:0.47, L_ankle:0.02, L_foot:0.09, R_shank:0.24, L_hand:0.03, R_hand:0.15 |
| 12 | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.101 / l_hip_pitch_joint | L_thigh:0.41, L_shank:0.13, L_foot:0.09, R_shank:0.37 |
| 13 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.060 / l_hip_pitch_joint | L_thigh:0.29, L_shank:0.28, R_shank:0.34, R_hand:0.09 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.062 / l_shoulder_pitch_joint | L_thigh:0.49, L_shank:0.02, L_foot:0.08, R_shank:0.29, R_hand:0.12 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.026 / l_hip_roll_joint | L_thigh:0.46, L_shank:0.03, L_foot:0.08, R_shank:0.22, L_hand:0.02, R_hand:0.18 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.036 / l_hip_pitch_joint | L_thigh:0.31, L_shank:0.31, R_shank:0.17, R_hand:0.20 |
| 17 | HAND_SUPPORT → HAND_SUPPORT | QUASI_STATIC | True | 0.037 / l_hip_pitch_joint | L_thigh:0.31, L_shank:0.31, R_shank:0.17, R_hand:0.21 |
| balance | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.056 / l_hip_pitch_joint | trunk:0.45, L_ankle:0.08, R_thigh:0.25, R_shank:0.22 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → SIT (4.00–4.98 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (4.98–6.88 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (6.88–7.08 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (7.08–7.88 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (7.88–8.28 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (8.28–10.28 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (10.28–10.78 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (10.78–10.88 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (10.88–11.38 s) | DYNAMIC_UNRESOLVED | False |
| SIT → TRANSITION (11.38–11.98 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (11.98–12.08 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (12.08–12.18 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (12.18–12.58 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (12.58–12.68 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (12.68–13.78 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (13.78–13.88 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (13.88–14.91 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (14.91–15.01 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (15.01–15.11 s) | DYNAMIC_UNRESOLVED | True |

Terminal sampled tail: DYNAMIC_UNRESOLVED, span 0.38 s, peak KE 2.298168 J.

Transition labels: DYNAMIC_UNRESOLVED=18, QUASI_STATIC=1.
Sampled peak KE: 2.298 J; minimum shell height: -0.35 mm; minimum self distance: -1.65 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## supine / 1

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: base_height=0.1254 (requires >0.16); foot_load=38.6115 (requires >75); nonfoot_load=69.6440 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: SUPINE → SIT → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.166 / r_hip_pitch_joint | trunk:0.65, L_foot:0.19, L_forearm:0.02, R_hand:0.14 |
| 1 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.020 / head_yaw_joint | trunk:0.12, L_foot:0.12, R_thigh:0.21, R_shank:0.04, R_ankle:0.02, R_foot:0.06, L_upper_arm:0.07, L_forearm:0.18, R_upper_arm:0.03, R_forearm:0.14 |
| 2 | SUPINE → SIT | DYNAMIC_UNRESOLVED | True | 0.118 / l_hip_pitch_joint | trunk:0.50, L_thigh:0.22, L_shank:0.10, R_thigh:0.09, R_shank:0.08 |
| 3 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.043 / l_hip_pitch_joint | trunk:0.81, L_ankle:0.09, R_ankle:0.08, R_foot:0.01 |
| 4 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.026 / l_hip_pitch_joint | L_thigh:0.28, L_ankle:0.02, L_foot:0.10, R_thigh:0.28, R_foot:0.12, L_hand:0.11, R_hand:0.11 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.040 / l_hip_pitch_joint | L_ankle:0.17, R_foot:0.20, L_hand:0.32, R_hand:0.30 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.054 / l_hip_pitch_joint | L_ankle:0.18, R_ankle:0.17, R_foot:0.03, L_hand:0.31, R_hand:0.31 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.041 / l_hip_pitch_joint | L_ankle:0.03, L_foot:0.14, R_ankle:0.18, L_hand:0.31, R_hand:0.34 |
| 8 | HAND_SUPPORT → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.136 / l_hip_pitch_joint | R_foot:0.26, L_hand:0.46, R_hand:0.29 |
| 9 | TRANSITION → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.036 / l_hip_pitch_joint | L_foot:0.22, R_foot:0.13, L_hand:0.30, R_hand:0.35 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.136 / r_hip_pitch_joint | L_foot:0.40, L_hand:0.19, R_hand:0.41 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.140 / l_hip_pitch_joint | L_foot:0.39, L_hand:0.20, R_hand:0.41 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.040 / head_pitch_joint | L_foot:0.22, R_foot:0.14, L_hand:0.30, R_hand:0.34 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | QUASI_STATIC | True | 0.042 / l_hip_pitch_joint | L_foot:0.24, R_foot:0.14, L_hand:0.29, R_hand:0.33 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.123 / l_hip_pitch_joint | L_foot:0.36, L_hand:0.24, R_hand:0.40 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.123 / l_hip_pitch_joint | L_foot:0.36, L_hand:0.24, R_hand:0.40 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.127 / l_hip_pitch_joint | L_foot:0.36, L_hand:0.24, R_hand:0.40 |
| 17 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.139 / l_hip_pitch_joint | L_foot:0.47, L_hand:0.18, R_hand:0.35 |
| 18 | HAND_SUPPORT → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.108 / l_hip_pitch_joint | L_foot:0.31, L_hand:0.25, R_hand:0.44 |
| 19 | TRANSITION → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.057 / head_pitch_joint | L_foot:0.24, R_foot:0.09, L_hand:0.28, R_hand:0.39 |
| balance | HAND_SUPPORT → SIT | STATICS_GAP | False | infeasible / None |  |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → SIT (3.48–4.96 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (4.96–6.16 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (6.16–6.76 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (6.76–6.86 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (6.86–7.36 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (7.36–7.46 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (7.46–8.66 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (8.66–8.76 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (8.76–8.88 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (8.88–8.96 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (8.96–9.06 s) | DYNAMIC_UNRESOLVED | False |
| TRANSITION → HAND_SUPPORT (9.06–9.16 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (9.16–12.06 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (12.06–12.16 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (12.16–14.46 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (14.46–14.56 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (14.56–14.66 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (14.66–14.76 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (14.76–14.86 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (14.86–14.96 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (14.96–15.79 s) | DYNAMIC_UNRESOLVED | True |

Terminal sampled tail: STATICS_GAP, span 0.42 s, peak KE 1.655621 J.

Transition labels: DYNAMIC_UNRESOLVED=19, QUASI_STATIC=1, STATICS_GAP=1.
Sampled peak KE: 1.656 J; minimum shell height: -0.65 mm; minimum self distance: -1.11 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## supine / 2

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: foot_load=22.5314 (requires >75); nonfoot_load=85.1376 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: SUPINE → TRANSITION → SUPINE → TRANSITION → SUPINE → SIT → HAND_SUPPORT → SIT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.058 / l_calf_joint | trunk:0.60, L_shank:0.24, R_shank:0.17 |
| 1 | SUPINE → SIT | DYNAMIC_UNRESOLVED | True | 0.100 / l_hip_pitch_joint | trunk:0.50, L_thigh:0.06, L_shank:0.08, R_thigh:0.29, R_shank:0.06 |
| 2 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.070 / l_hip_pitch_joint | trunk:0.50, L_thigh:0.12, L_shank:0.11, R_thigh:0.20, R_shank:0.08 |
| 3 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.041 / l_hip_pitch_joint | trunk:0.11, L_thigh:0.04, L_shank:0.18, R_thigh:0.10, R_shank:0.23, L_hand:0.17, R_hand:0.17 |
| 4 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_ankle:0.17, R_ankle:0.18, L_hand:0.32, R_hand:0.32 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.021 / l_hip_pitch_joint | L_shank:0.21, L_foot:0.04, R_shank:0.17, R_foot:0.06, L_hand:0.25, R_hand:0.27 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.035 / l_hip_pitch_joint | L_shank:0.23, L_ankle:0.07, R_shank:0.22, R_ankle:0.07, L_hand:0.21, R_hand:0.20 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.036 / l_hip_pitch_joint | L_shank:0.23, L_ankle:0.06, R_shank:0.23, R_ankle:0.07, L_hand:0.21, R_hand:0.20 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.050 / l_hip_pitch_joint | L_shank:0.33, R_shank:0.32, L_hand:0.18, R_hand:0.17 |
| 9 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.038 / r_hip_pitch_joint | L_shank:0.20, R_shank:0.30, L_hand:0.27, R_hand:0.24 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.041 / head_pitch_joint | L_shank:0.21, R_shank:0.30, L_hand:0.22, R_hand:0.27 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.086 / l_hip_pitch_joint | L_thigh:0.02, L_shank:0.28, R_shank:0.38, R_hand:0.32 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.077 / l_hip_pitch_joint | L_thigh:0.32, R_shank:0.37, R_hand:0.31 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.151 / r_shoulder_pitch_joint | L_thigh:0.28, R_shank:0.25, R_hand:0.47 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.134 / r_shoulder_pitch_joint | L_thigh:0.28, R_shank:0.32, R_hand:0.40 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.070 / head_pitch_joint | L_thigh:0.10, L_shank:0.25, R_shank:0.30, R_hand:0.35 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.077 / l_hip_roll_joint | L_thigh:0.19, L_shank:0.18, R_shank:0.34, R_hand:0.29 |
| 17 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.079 / head_pitch_joint | L_shank:0.39, L_ankle:0.03, L_foot:0.02, R_thigh:0.17, R_shank:0.10, R_hand:0.30 |
| 18 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.083 / head_pitch_joint | L_thigh:0.16, L_shank:0.21, L_ankle:0.06, L_foot:0.03, R_shank:0.39, R_hand:0.15 |
| balance | HAND_SUPPORT → SIT | STATICS_GAP | False | infeasible / None |  |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → TRANSITION (3.72–4.00 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SUPINE (4.00–4.10 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → TRANSITION (4.10–4.50 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SUPINE (4.50–4.60 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → SIT (4.60–4.70 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (4.70–5.80 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (5.80–6.20 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (6.20–6.30 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (6.30–6.40 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (6.40–8.70 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (8.70–8.90 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (8.90–9.30 s) | DYNAMIC_UNRESOLVED | False |
| TRANSITION → HAND_SUPPORT (9.30–9.40 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (9.40–11.70 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (11.70–11.80 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (11.80–12.90 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (12.90–13.00 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (13.00–14.30 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (14.30–14.40 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (14.40–15.33 s) | STATICS_GAP | False |
| TRANSITION → SIT (15.33–15.51 s) | STATICS_GAP | False |

Terminal sampled tail: STATICS_GAP, span 0.38 s, peak KE 11.263290 J.

Transition labels: DYNAMIC_UNRESOLVED=19, STATICS_GAP=1.
Sampled peak KE: 11.263 J; minimum shell height: -0.57 mm; minimum self distance: -1.57 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 0

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: base_height=0.1040 (requires >0.16); foot_load=0.0000 (requires >75); nonfoot_load=107.3923 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: PRONE → SIT → HAND_SUPPORT → CROUCH → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → PRONE | DYNAMIC_UNRESOLVED | True | 0.089 / l_shoulder_pitch_joint | trunk:0.66, L_foot:0.10, R_shank:0.21, R_hand:0.03 |
| 1 | PRONE → PRONE | DYNAMIC_UNRESOLVED | True | 0.094 / r_hip_pitch_joint | trunk:0.34, L_thigh:0.12, L_foot:0.14, R_thigh:0.39 |
| 2 | PRONE → SIT | DYNAMIC_UNRESOLVED | True | 0.135 / r_hip_pitch_joint | trunk:0.34, L_thigh:0.04, L_foot:0.13, R_thigh:0.48 |
| 3 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.082 / head_pitch_joint | L_thigh:0.33, L_foot:0.14, R_thigh:0.38, R_foot:0.15 |
| 4 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.065 / head_pitch_joint | L_foot:0.18, R_thigh:0.52, R_foot:0.14, L_hand:0.16 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.090 / l_elbow_joint | L_foot:0.11, R_thigh:0.54, R_foot:0.13, L_hand:0.22 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.092 / head_pitch_joint | L_foot:0.10, R_thigh:0.51, R_foot:0.14, L_hand:0.25 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.095 / l_elbow_joint | L_foot:0.09, R_thigh:0.54, R_foot:0.13, L_hand:0.24 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.192 / l_hip_pitch_joint | R_thigh:0.28, R_shank:0.40, L_hand:0.31 |
| 9 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.093 / head_pitch_joint | L_foot:0.10, R_thigh:0.32, R_shank:0.13, L_hand:0.31, R_hand:0.14 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.094 / head_pitch_joint | L_shank:0.08, L_foot:0.09, R_thigh:0.36, R_shank:0.21, L_hand:0.25, R_hand:0.02 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.180 / l_hip_pitch_joint | R_thigh:0.27, R_shank:0.32, L_hand:0.34, R_hand:0.07 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.096 / head_pitch_joint | L_shank:0.24, R_thigh:0.40, R_shank:0.16, L_hand:0.19, R_hand:0.02 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.094 / l_shoulder_pitch_joint | L_shank:0.24, R_thigh:0.46, R_shank:0.13, L_hand:0.17 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.166 / l_hip_pitch_joint | R_thigh:0.29, R_shank:0.40, L_hand:0.31 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.171 / l_hip_pitch_joint | R_thigh:0.28, R_shank:0.41, L_hand:0.31 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.161 / r_hip_pitch_joint | R_thigh:0.20, R_shank:0.46, L_hand:0.34 |
| balance | HAND_SUPPORT → SIT | STATICS_GAP | True | 0.091 / l_hip_pitch_joint | trunk:0.41, L_thigh:0.38, R_thigh:0.08, R_shank:0.12, L_hand:0.01 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → SIT (1.28–2.76 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (2.76–3.96 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → CROUCH (3.96–4.46 s) | DYNAMIC_UNRESOLVED | False |
| CROUCH → SIT (4.46–4.56 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (4.56–4.66 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (4.66–6.16 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (6.16–6.26 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (6.26–6.36 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (6.36–6.46 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (6.46–6.86 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (6.86–6.96 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (6.96–8.06 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (8.06–8.36 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (8.36–9.26 s) | DYNAMIC_UNRESOLVED | False |
| SIT → HAND_SUPPORT (9.26–9.36 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (9.36–9.86 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (9.86–9.96 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (9.96–11.06 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (11.06–11.16 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (11.16–11.59 s) | DYNAMIC_UNRESOLVED | True |

Terminal sampled tail: STATICS_GAP, span 0.46 s, peak KE 1.573452 J.

Transition labels: DYNAMIC_UNRESOLVED=17, STATICS_GAP=1.
Sampled peak KE: 1.573 J; minimum shell height: -0.30 mm; minimum self distance: -2.43 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 1

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: base_height=0.1128 (requires >0.16); foot_load=0.0000 (requires >75); nonfoot_load=107.0852 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: PRONE → ALLFOURS → PRONE → SIT → TRANSITION → SIT → TRANSITION → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → PRONE | DYNAMIC_UNRESOLVED | True | 0.088 / head_pitch_joint | trunk:0.70, L_foot:0.14, R_foot:0.11, L_hand:0.03, R_hand:0.01 |
| 1 | PRONE → PRONE | DYNAMIC_UNRESOLVED | True | 0.093 / head_pitch_joint | trunk:0.57, L_shank:0.23, R_shank:0.21 |
| 2 | PRONE → SIT | DYNAMIC_UNRESOLVED | True | 0.098 / l_hip_pitch_joint | trunk:0.44, L_thigh:0.25, R_thigh:0.19, R_foot:0.10, L_hand:0.03 |
| 3 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.088 / head_pitch_joint | L_thigh:0.60, R_thigh:0.20, R_shank:0.20 |
| 4 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.085 / head_pitch_joint | L_thigh:0.18, L_shank:0.28, R_thigh:0.54 |
| 5 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.086 / head_pitch_joint | L_thigh:0.55, R_thigh:0.32, R_shank:0.14 |
| 6 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.085 / head_pitch_joint | L_thigh:0.43, L_shank:0.09, R_thigh:0.28, R_shank:0.21 |
| 7 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.089 / head_pitch_joint | L_thigh:0.25, L_shank:0.25, R_thigh:0.33, R_shank:0.17 |
| 8 | SIT → SIT | QUASI_STATIC | True | 0.091 / head_pitch_joint | L_thigh:0.29, L_shank:0.23, R_thigh:0.30, R_shank:0.18 |
| 9 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.092 / head_pitch_joint | L_thigh:0.31, L_shank:0.25, R_thigh:0.38, R_shank:0.06 |
| 10 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.096 / head_pitch_joint | L_thigh:0.21, L_shank:0.27, R_thigh:0.32, R_shank:0.21 |
| 11 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.097 / head_pitch_joint | L_thigh:0.42, L_shank:0.09, R_thigh:0.27, R_shank:0.21 |
| 12 | SIT → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.089 / r_hip_pitch_joint | L_thigh:0.45, L_shank:0.16, R_thigh:0.31, R_shank:0.05, R_foot:0.03 |
| 13 | TRANSITION → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.094 / r_hip_pitch_joint | L_thigh:0.58, R_thigh:0.13, R_shank:0.24, L_hand:0.05 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.094 / head_pitch_joint | L_thigh:0.63, R_thigh:0.07, R_shank:0.30 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.092 / head_pitch_joint | L_thigh:0.49, R_thigh:0.34, R_shank:0.01, L_hand:0.16 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.099 / l_shoulder_pitch_joint | L_thigh:0.49, R_thigh:0.25, L_hand:0.26 |
| balance | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.093 / head_pitch_joint | trunk:0.39, L_thigh:0.30, L_foot:0.10, R_thigh:0.15, R_foot:0.06 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → ALLFOURS (1.12–1.40 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → PRONE (1.40–1.50 s) | DYNAMIC_UNRESOLVED | True |
| PRONE → SIT (1.50–2.60 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (2.60–3.20 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (3.20–3.30 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (3.30–8.92 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (8.92–9.00 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (9.00–9.20 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (9.20–9.60 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (9.60–9.70 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (9.70–10.20 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (10.20–10.40 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (10.40–10.80 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (10.80–11.00 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (11.00–11.43 s) | DYNAMIC_UNRESOLVED | True |

Terminal sampled tail: DYNAMIC_UNRESOLVED, span 0.52 s, peak KE 1.120134 J.

Transition labels: DYNAMIC_UNRESOLVED=17, QUASI_STATIC=1.
Sampled peak KE: 1.198 J; minimum shell height: -0.47 mm; minimum self distance: -1.10 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 2

Initial passive settle: True; QP attempts: 0.
Budget exhausted. Final retained state misses handover gates: base_height=0.1288 (requires >0.16); foot_load=28.7528 (requires >75); nonfoot_load=79.3764 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.

Sampled mode sequence: PRONE → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → PRONE → SIT → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.074 / head_pitch_joint | L_foot:0.19, R_foot:0.16, L_upper_arm:0.28, L_forearm:0.04, R_upper_arm:0.17, R_forearm:0.15 |
| 1 | TRANSITION → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.078 / r_hip_pitch_joint | L_foot:0.16, R_foot:0.21, L_upper_arm:0.06, L_forearm:0.27, R_upper_arm:0.17, R_forearm:0.12 |
| 2 | ALLFOURS → PRONE | DYNAMIC_UNRESOLVED | True | 0.091 / l_hip_pitch_joint | trunk:0.02, L_thigh:0.52, R_foot:0.13, L_forearm:0.03, R_forearm:0.30 |
| 3 | PRONE → SIT | DYNAMIC_UNRESOLVED | True | 0.122 / l_hip_pitch_joint | trunk:0.12, L_thigh:0.42, R_thigh:0.35, R_foot:0.11 |
| 4 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.182 / l_hip_pitch_joint | R_thigh:0.52, R_foot:0.20, L_hand:0.26, R_hand:0.03 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.133 / l_hip_pitch_joint | L_thigh:0.48, R_thigh:0.28, R_foot:0.22, R_hand:0.02 |
| 6 | HAND_SUPPORT → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.198 / l_hip_pitch_joint | R_foot:0.32, L_hand:0.39, R_hand:0.29 |
| 7 | TRANSITION → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.087 / r_hip_pitch_joint | L_foot:0.13, R_foot:0.24, L_hand:0.33, R_hand:0.30 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.202 / l_hip_pitch_joint | L_foot:0.42, L_hand:0.28, R_hand:0.30 |
| 9 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.174 / l_hip_pitch_joint | L_foot:0.46, L_hand:0.27, R_hand:0.27 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.120 / l_hip_pitch_joint | L_foot:0.31, L_hand:0.32, R_hand:0.37 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.126 / l_hip_pitch_joint | L_foot:0.33, L_hand:0.32, R_hand:0.35 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.149 / l_hip_pitch_joint | L_foot:0.32, L_hand:0.31, R_hand:0.37 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.198 / l_shoulder_pitch_joint | L_foot:0.38, L_hand:0.26, R_hand:0.36 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.186 / l_shoulder_pitch_joint | L_foot:0.31, L_hand:0.32, R_hand:0.36 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.146 / r_hip_pitch_joint | L_foot:0.28, L_hand:0.33, R_hand:0.39 |
| balance | HAND_SUPPORT → SIT | DYNAMIC_UNRESOLVED | True | 0.109 / l_hip_roll_joint | L_thigh:0.36, L_foot:0.19, R_thigh:0.37, R_hand:0.08 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → ALLFOURS (1.16–1.24 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.24–1.34 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (1.34–2.14 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (2.14–2.54 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → PRONE (2.54–2.64 s) | DYNAMIC_UNRESOLVED | True |
| PRONE → SIT (2.64–3.24 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (3.24–3.84 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (3.84–4.94 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (4.94–5.14 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (5.14–5.36 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (5.36–5.44 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (5.44–7.54 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (7.54–7.64 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (7.64–9.04 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (9.04–9.14 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (9.14–11.07 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (11.07–11.17 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (11.17–12.51 s) | DYNAMIC_UNRESOLVED | True |

Terminal sampled tail: DYNAMIC_UNRESOLVED, span 0.94 s, peak KE 0.074996 J.

Transition labels: DYNAMIC_UNRESOLVED=17.
Sampled peak KE: 1.462 J; minimum shell height: -0.90 mm; minimum self distance: -0.83 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.
