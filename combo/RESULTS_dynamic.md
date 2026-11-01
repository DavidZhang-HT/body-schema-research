# Measured outcomes

| Start | Seed | Search/QP success | Replay success | Evals | Blocks | Wall s |
|---|---:|---|---|---:|---:|---:|
| supine | 0 | True | True | 5607 | 24 | 391.4 |
| supine | 1 | True | True | 4235 | 16 | 406.6 |
| supine | 2 | True | True | 3542 | 14 | 248.9 |
| prone | 0 | False | False | 6000 | 16 | 445.4 |
| prone | 1 | True | True | 4986 | 19 | 631.6 |
| prone | 2 | False | False | 6000 | 18 | 354.4 |

supine: 3/3 completed search runs succeeded.

prone: 1/3 completed search runs succeeded.

## supine / 0

Initial passive settle: True; QP attempts: 5.

Sampled mode sequence: SUPINE → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SUPINE → SIT → HAND_SUPPORT → SIT → TRANSITION → SIT → HAND_SUPPORT → SIT → HAND_SUPPORT → SIT → CROUCH → SIT → TRANSITION → SIT → CROUCH → SIT → CROUCH → SIT → HAND_SUPPORT → ALLFOURS → TRANSITION → CROUCH → TRANSITION → CROUCH → ALLFOURS → HAND_SUPPORT → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → STAND.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.074 / l_shoulder_pitch_joint | trunk:0.57, L_shank:0.29, R_shank:0.14 |
| 1 | SUPINE → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.078 / l_hip_pitch_joint | trunk:0.10, L_thigh:0.24, L_shank:0.05, R_thigh:0.12, R_shank:0.09, L_forearm:0.20, R_forearm:0.20 |
| 2 | HAND_SUPPORT → SIT | STATICS_GAP | True | 0.050 / l_hip_pitch_joint | trunk:0.50, L_foot:0.10, R_thigh:0.32, R_foot:0.08 |
| 3 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.021 / r_hip_pitch_joint | trunk:0.42, L_thigh:0.19, L_foot:0.10, R_thigh:0.16, R_foot:0.11, L_hand:0.01, R_hand:0.01 |
| 4 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.205 / l_hip_pitch_joint | R_thigh:0.46, R_foot:0.27, L_hand:0.27 |
| 5 | HAND_SUPPORT → SIT | STATICS_GAP | True | 0.163 / r_hip_pitch_joint | L_thigh:0.75, R_foot:0.25 |
| 6 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.193 / r_hip_pitch_joint | L_thigh:0.73, R_foot:0.27 |
| 7 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.074 / r_thigh_joint | L_thigh:0.66, L_foot:0.16, R_foot:0.18 |
| 8 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.085 / r_hip_pitch_joint | L_thigh:0.57, L_foot:0.18, R_foot:0.25 |
| 9 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.087 / r_thigh_joint | L_thigh:0.60, L_foot:0.22, R_foot:0.18 |
| 10 | SIT → SIT | STATICS_GAP | True | 0.092 / r_shoulder_pitch_joint | L_thigh:0.55, L_foot:0.19, R_foot:0.27 |
| 11 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.084 / l_hip_pitch_joint | L_thigh:0.49, L_foot:0.21, R_foot:0.31 |
| 12 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.37, L_foot:0.23, R_foot:0.40 |
| 13 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.129 / l_hip_pitch_joint | L_thigh:0.34, L_foot:0.31, R_foot:0.35 |
| 14 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.076 / l_hip_pitch_joint | L_thigh:0.30, L_foot:0.25, R_foot:0.45 |
| 15 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.42, L_foot:0.20, R_foot:0.37 |
| 16 | SIT → SIT | DYNAMIC_UNRESOLVED | True | 0.080 / l_hip_pitch_joint | L_thigh:0.38, L_foot:0.21, R_foot:0.41 |
| 17 | SIT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.285 / l_elbow_joint | R_foot:0.44, L_hand:0.56 |
| 18 | HAND_SUPPORT → ALLFOURS | STATICS_GAP | True | 0.082 / l_hip_pitch_joint | L_foot:0.30, R_foot:0.38, L_hand:0.32 |
| 19 | ALLFOURS → CROUCH | DYNAMIC_UNRESOLVED | True | 0.272 / l_hip_pitch_joint | L_foot:0.65, R_foot:0.35 |
| 20 | CROUCH → CROUCH | STATICS_GAP | True | 0.088 / l_hip_pitch_joint | L_foot:0.38, R_foot:0.41, L_hand:0.22 |
| 21 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.220 / l_hip_pitch_joint | L_foot:0.75, R_foot:0.25 |
| 22 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.211 / l_hip_pitch_joint | L_foot:0.73, R_foot:0.27 |
| 23 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.175 / l_hip_pitch_joint | L_foot:0.73, R_foot:0.27 |
| balance | CROUCH → STAND | DYNAMIC_UNRESOLVED | True | 0.090 / l_hip_roll_joint | L_foot:0.53, R_foot:0.47 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → HAND_SUPPORT (4.00–4.78 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (4.78–4.88 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (4.88–5.18 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SUPINE (5.18–5.28 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → SIT (5.28–5.48 s) | STATICS_GAP | False |
| SIT → HAND_SUPPORT (5.48–5.68 s) | STATICS_GAP | True |
| HAND_SUPPORT → SIT (5.68–5.78 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (5.78–6.08 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (6.08–6.18 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (6.18–6.28 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (6.28–6.58 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (6.58–6.68 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SIT (6.68–7.08 s) | DYNAMIC_UNRESOLVED | True |
| SIT → CROUCH (7.08–11.38 s) | STATICS_GAP | True |
| CROUCH → SIT (11.38–11.68 s) | DYNAMIC_UNRESOLVED | True |
| SIT → TRANSITION (11.68–11.98 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SIT (11.98–12.08 s) | DYNAMIC_UNRESOLVED | True |
| SIT → CROUCH (12.08–13.18 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → SIT (13.18–13.28 s) | DYNAMIC_UNRESOLVED | True |
| SIT → CROUCH (13.28–13.38 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → SIT (13.38–13.48 s) | DYNAMIC_UNRESOLVED | True |
| SIT → HAND_SUPPORT (13.48–14.38 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → ALLFOURS (14.38–14.98 s) | STATICS_GAP | True |
| ALLFOURS → TRANSITION (14.98–15.68 s) | STATICS_GAP | True |
| TRANSITION → CROUCH (15.68–15.78 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (15.78–15.88 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (15.88–15.98 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → ALLFOURS (15.98–16.08 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → HAND_SUPPORT (16.08–16.28 s) | STATICS_GAP | False |
| HAND_SUPPORT → CROUCH (16.28–16.48 s) | STATICS_GAP | True |
| CROUCH → TRANSITION (16.48–16.88 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (16.88–16.98 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (16.98–17.38 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (17.38–17.58 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (17.58–17.98 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (17.98–18.08 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → STAND (18.08–20.61 s) | DYNAMIC_UNRESOLVED | True |

37 distinct graph-edge witnesses replayed and explained; 32 certified destination configurations.

Terminal sampled tail: QUASI_STATIC, span 0.98 s, peak KE 0.000014 J.

Transition labels: DYNAMIC_UNRESOLVED=20, STATICS_GAP=5.
Sampled peak KE: 2.309 J; minimum shell height: -0.92 mm; minimum self distance: -0.66 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## supine / 1

Initial passive settle: True; QP attempts: 53.

Sampled mode sequence: SUPINE → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SUPINE → TRANSITION → SUPINE → SIT → SUPINE → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → ALLFOURS → TRANSITION → ALLFOURS → HAND_SUPPORT → TRANSITION → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → STAND.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.157 / r_hip_pitch_joint | trunk:0.44, L_foot:0.22, L_upper_arm:0.05, L_forearm:0.06, R_upper_arm:0.10, R_forearm:0.12 |
| 1 | HAND_SUPPORT → SUPINE | DYNAMIC_UNRESOLVED | True | 0.118 / l_hip_pitch_joint | trunk:0.31, L_thigh:0.18, L_shank:0.09, R_thigh:0.41 |
| 2 | SUPINE → HAND_SUPPORT | STATICS_GAP | True | 0.061 / l_hip_pitch_joint | L_shank:0.26, R_shank:0.32, L_hand:0.20, R_hand:0.22 |
| 3 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.036 / l_hip_roll_joint | L_shank:0.26, R_shank:0.23, L_hand:0.21, R_hand:0.30 |
| 4 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.033 / l_hip_pitch_joint | L_shank:0.26, R_shank:0.17, R_foot:0.06, L_hand:0.19, R_hand:0.33 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.034 / l_hip_pitch_joint | L_shank:0.26, R_shank:0.17, R_ankle:0.06, L_hand:0.18, R_hand:0.32 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.196 / l_hip_pitch_joint | L_ankle:0.08, L_foot:0.19, L_hand:0.28, R_hand:0.44 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.038 / l_hip_pitch_joint | L_shank:0.19, L_ankle:0.05, R_ankle:0.16, L_hand:0.27, R_hand:0.33 |
| 8 | HAND_SUPPORT → ALLFOURS | STATICS_GAP | True | 0.057 / head_pitch_joint | L_shank:0.57, R_foot:0.15, R_hand:0.28 |
| 9 | ALLFOURS → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.066 / head_pitch_joint | L_shank:0.52, R_foot:0.20, R_hand:0.28 |
| 10 | HAND_SUPPORT → TRANSITION | STATICS_GAP | True | 0.130 / r_hip_pitch_joint | L_shank:0.31, R_shank:0.46, R_ankle:0.20, R_foot:0.03 |
| 11 | TRANSITION → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.229 / l_hip_pitch_joint | L_foot:0.46, R_shank:0.54 |
| 12 | TRANSITION → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.116 / l_hip_pitch_joint | L_foot:0.33, R_shank:0.67 |
| 13 | TRANSITION → TRANSITION | STATICS_GAP | True | 0.171 / l_thigh_joint | L_foot:0.43, R_shank:0.29, R_foot:0.28 |
| 14 | TRANSITION → CROUCH | STATICS_GAP | True | 0.239 / l_calf_joint | L_foot:0.40, R_shank:0.20, R_foot:0.40 |
| 15 | CROUCH → CROUCH | STATICS_GAP | True | 0.240 / l_calf_joint | L_foot:0.60, R_shank:0.04, R_foot:0.36 |
| balance | CROUCH → STAND | DYNAMIC_UNRESOLVED | True | 0.060 / l_calf_joint | L_foot:0.46, R_foot:0.54 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → HAND_SUPPORT (3.48–3.56 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (3.56–3.66 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (3.66–3.76 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SUPINE (3.76–4.16 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → TRANSITION (4.16–4.26 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → SUPINE (4.26–4.36 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → SIT (4.36–4.46 s) | DYNAMIC_UNRESOLVED | True |
| SIT → SUPINE (4.46–4.66 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → TRANSITION (4.66–4.86 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (4.86–5.26 s) | STATICS_GAP | True |
| HAND_SUPPORT → TRANSITION (5.26–5.56 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (5.56–5.66 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (5.66–6.16 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (6.16–6.26 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (6.26–7.36 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (7.36–7.46 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → ALLFOURS (7.46–8.86 s) | STATICS_GAP | True |
| ALLFOURS → TRANSITION (8.86–8.96 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (8.96–9.06 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → HAND_SUPPORT (9.06–9.26 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (9.26–9.66 s) | STATICS_GAP | False |
| TRANSITION → CROUCH (9.66–12.16 s) | STATICS_GAP | False |
| CROUCH → TRANSITION (12.16–12.46 s) | STATICS_GAP | True |
| TRANSITION → CROUCH (12.46–12.48 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (12.48–12.76 s) | STATICS_GAP | False |
| TRANSITION → CROUCH (12.76–12.96 s) | STATICS_GAP | True |
| CROUCH → STAND (12.96–15.19 s) | DYNAMIC_UNRESOLVED | True |

41 distinct graph-edge witnesses replayed and explained; 37 certified destination configurations.

Terminal sampled tail: QUASI_STATIC, span 0.98 s, peak KE 0.000048 J.

Transition labels: DYNAMIC_UNRESOLVED=11, STATICS_GAP=6.
Sampled peak KE: 3.525 J; minimum shell height: -1.97 mm; minimum self distance: -1.63 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## supine / 2

Initial passive settle: True; QP attempts: 6.

Sampled mode sequence: SUPINE → HAND_SUPPORT → SUPINE → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → SIT → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → CROUCH → TRANSITION → CROUCH → HAND_SUPPORT → TRANSITION → CROUCH → HAND_SUPPORT → CROUCH → HAND_SUPPORT → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → STAND.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | SUPINE → SUPINE | DYNAMIC_UNRESOLVED | True | 0.219 / r_hip_pitch_joint | trunk:0.47, L_foot:0.23, L_forearm:0.06, R_forearm:0.23 |
| 1 | SUPINE → TRANSITION | STATICS_GAP | False | infeasible / None |  |
| 2 | TRANSITION → SIT | STATICS_GAP | False | infeasible / None |  |
| 3 | SIT → HAND_SUPPORT | STATICS_GAP | True | 0.116 / r_hip_pitch_joint | L_thigh:0.14, L_ankle:0.05, R_thigh:0.35, R_shank:0.26, R_hand:0.19 |
| 4 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.717 / r_shoulder_pitch_joint | R_foot:0.48, R_hand:0.52 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | STATICS_GAP | True | 0.080 / r_hip_pitch_joint | L_thigh:0.29, R_thigh:0.44, R_hand:0.27 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.078 / r_hip_pitch_joint | L_thigh:0.28, R_thigh:0.47, R_hand:0.24 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.104 / l_calf_joint | L_thigh:0.11, L_foot:0.24, R_foot:0.28, R_hand:0.37 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.113 / l_thigh_joint | L_foot:0.29, R_foot:0.30, R_hand:0.41 |
| 9 | HAND_SUPPORT → TRANSITION | STATICS_GAP | True | 0.181 / l_calf_joint | L_foot:0.28, R_shank:0.55, R_foot:0.18 |
| 10 | TRANSITION → CROUCH | STATICS_GAP | False | infeasible / None |  |
| 11 | CROUCH → HAND_SUPPORT | STATICS_GAP | True | 0.148 / l_hip_pitch_joint | L_foot:0.51, R_foot:0.31, R_hand:0.18 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.191 / l_hip_pitch_joint | L_foot:0.58, R_foot:0.33, R_hand:0.09 |
| 13 | HAND_SUPPORT → CROUCH | DYNAMIC_UNRESOLVED | True | 0.279 / l_hip_pitch_joint | L_foot:0.65, R_foot:0.35 |
| balance | CROUCH → STAND | DYNAMIC_UNRESOLVED | True | 0.103 / l_ankle_roll_joint | L_foot:0.46, R_foot:0.54 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| SUPINE → HAND_SUPPORT (3.72–4.00 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → SUPINE (4.00–4.10 s) | DYNAMIC_UNRESOLVED | True |
| SUPINE → HAND_SUPPORT (4.10–4.50 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (4.50–4.60 s) | STATICS_GAP | False |
| TRANSITION → HAND_SUPPORT (4.60–5.20 s) | STATICS_GAP | False |
| HAND_SUPPORT → SIT (5.20–5.40 s) | STATICS_GAP | False |
| SIT → HAND_SUPPORT (5.40–5.70 s) | STATICS_GAP | True |
| HAND_SUPPORT → TRANSITION (5.70–7.00 s) | STATICS_GAP | False |
| TRANSITION → HAND_SUPPORT (7.00–7.10 s) | STATICS_GAP | False |
| HAND_SUPPORT → TRANSITION (7.10–8.80 s) | STATICS_GAP | True |
| TRANSITION → HAND_SUPPORT (8.80–8.90 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (8.90–9.40 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (9.40–10.00 s) | STATICS_GAP | True |
| CROUCH → TRANSITION (10.00–10.10 s) | STATICS_GAP | False |
| TRANSITION → CROUCH (10.10–10.20 s) | STATICS_GAP | True |
| CROUCH → HAND_SUPPORT (10.20–10.50 s) | STATICS_GAP | True |
| HAND_SUPPORT → TRANSITION (10.50–10.60 s) | STATICS_GAP | False |
| TRANSITION → CROUCH (10.60–10.70 s) | STATICS_GAP | False |
| CROUCH → HAND_SUPPORT (10.70–10.90 s) | STATICS_GAP | True |
| HAND_SUPPORT → CROUCH (10.90–11.20 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → HAND_SUPPORT (11.20–11.30 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → CROUCH (11.30–11.70 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (11.70–12.33 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (12.33–12.43 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (12.43–13.23 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (13.23–13.43 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (13.43–13.53 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (13.53–13.63 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (13.63–13.73 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (13.73–14.33 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → STAND (14.33–15.33 s) | DYNAMIC_UNRESOLVED | True |

38 distinct graph-edge witnesses replayed and explained; 32 certified destination configurations.

Terminal sampled tail: QUASI_STATIC, span 0.98 s, peak KE 0.000200 J.

Transition labels: DYNAMIC_UNRESOLVED=8, STATICS_GAP=7.
Sampled peak KE: 2.963 J; minimum shell height: -0.91 mm; minimum self distance: -0.87 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 0

Initial passive settle: True; QP attempts: 19.
Budget exhausted. Final retained state misses handover gates: none; QP rejected entry.
These are measured algorithmic stopping constraints, not a proof of physical impossibility.
Last QP attempt ended at base=0.080 m, up_z=0.765, nonfoot load=101.5 N after about 1.02 s.

Sampled mode sequence: PRONE → TRANSITION → ALLFOURS → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → TRANSITION → HAND_SUPPORT → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH → TRANSITION → CROUCH.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → PRONE | QUASI_STATIC | True | 0.000 / l_hip_pitch_joint | trunk:0.41, L_shank:0.07, L_foot:0.10, R_shank:0.07, R_foot:0.11, L_forearm:0.07, R_forearm:0.07, head:0.11 |
| 1 | PRONE → TRANSITION | STATICS_GAP | False | infeasible / None |  |
| 2 | TRANSITION → ALLFOURS | STATICS_GAP | True | 0.098 / head_pitch_joint | L_shank:0.28, R_shank:0.23, L_hand:0.24, R_hand:0.26 |
| 3 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.093 / head_pitch_joint | L_foot:0.23, R_foot:0.20, L_hand:0.28, R_hand:0.29 |
| 4 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.090 / l_hip_pitch_joint | L_foot:0.20, R_foot:0.21, L_hand:0.29, R_hand:0.29 |
| 5 | ALLFOURS → ALLFOURS | STATICS_GAP | True | 0.134 / l_hip_pitch_joint | L_foot:0.37, R_foot:0.14, R_hand:0.49 |
| 6 | ALLFOURS → HAND_SUPPORT | STATICS_GAP | True | 0.171 / l_hip_roll_joint | L_foot:0.32, R_foot:0.24, R_hand:0.44 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.097 / l_hip_roll_joint | L_foot:0.35, R_foot:0.23, R_hand:0.42 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.091 / l_hip_roll_joint | L_foot:0.33, R_foot:0.27, R_hand:0.41 |
| 9 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.094 / l_hip_roll_joint | L_foot:0.33, R_foot:0.28, R_hand:0.40 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.075 / head_yaw_joint | L_foot:0.35, R_foot:0.30, R_hand:0.35 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.150 / r_thigh_joint | L_foot:0.41, R_foot:0.42, R_hand:0.17 |
| 12 | HAND_SUPPORT → CROUCH | STATICS_GAP | True | 0.225 / l_hip_pitch_joint | L_foot:0.62, R_foot:0.38 |
| 13 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.309 / l_calf_joint | L_foot:0.68, R_foot:0.32 |
| 14 | CROUCH → CROUCH | STATICS_GAP | False | infeasible / None |  |
| 15 | CROUCH → CROUCH | STATICS_GAP | True | 0.240 / l_hip_pitch_joint | L_foot:0.62, R_foot:0.38 |
| balance | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.129 / l_ankle_pitch_joint | L_foot:0.51, R_foot:0.49 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → TRANSITION (1.28–2.06 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (2.06–2.56 s) | STATICS_GAP | True |
| ALLFOURS → HAND_SUPPORT (2.56–5.16 s) | STATICS_GAP | True |
| HAND_SUPPORT → TRANSITION (5.16–5.66 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (5.66–5.76 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (5.76–8.16 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (8.16–8.26 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → CROUCH (8.26–8.66 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (8.66–8.86 s) | STATICS_GAP | True |
| TRANSITION → CROUCH (8.86–8.96 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (8.96–9.46 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (9.46–9.56 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → TRANSITION (9.56–9.76 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (9.76–9.86 s) | DYNAMIC_UNRESOLVED | True |

51 distinct graph-edge witnesses replayed and explained; 46 certified destination configurations.

Terminal sampled tail: DYNAMIC_UNRESOLVED, span 0.98 s, peak KE 0.053256 J.

Transition labels: QUASI_STATIC=1, STATICS_GAP=7, DYNAMIC_UNRESOLVED=9.
Sampled peak KE: 5.052 J; minimum shell height: -2.25 mm; minimum self distance: -0.18 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 1

Initial passive settle: True; QP attempts: 77.

Sampled mode sequence: PRONE → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → HAND_SUPPORT → ALLFOURS → TRANSITION → HAND_SUPPORT → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → HAND_SUPPORT → TRANSITION → CROUCH → TRANSITION → CROUCH → STAND.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → TRANSITION | DYNAMIC_UNRESOLVED | True | 0.124 / head_yaw_joint | L_foot:0.28, R_shank:0.35, head:0.37 |
| 1 | TRANSITION → ALLFOURS | STATICS_GAP | True | 0.097 / head_pitch_joint | L_foot:0.19, R_shank:0.08, R_foot:0.09, L_forearm:0.38, R_hand:0.26 |
| 2 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.091 / l_hip_pitch_joint | L_shank:0.35, R_shank:0.06, R_foot:0.04, L_hand:0.27, R_hand:0.28 |
| 3 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.107 / l_hip_pitch_joint | R_shank:0.48, L_hand:0.39, R_hand:0.13 |
| 4 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.111 / l_hip_pitch_joint | R_shank:0.48, R_foot:0.01, L_hand:0.40, R_hand:0.11 |
| 5 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.170 / r_hip_pitch_joint | R_foot:0.34, L_hand:0.45, R_hand:0.21 |
| 6 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.307 / r_hip_pitch_joint | R_foot:0.41, L_hand:0.59 |
| 7 | ALLFOURS → ALLFOURS | STATICS_GAP | True | 0.255 / r_calf_joint | R_shank:0.13, R_foot:0.45, L_hand:0.43 |
| 8 | ALLFOURS → TRANSITION | STATICS_GAP | False | infeasible / None |  |
| 9 | TRANSITION → ALLFOURS | STATICS_GAP | True | 0.092 / head_pitch_joint | L_shank:0.26, L_foot:0.17, R_foot:0.35, L_hand:0.22 |
| 10 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.096 / l_hip_roll_joint | L_shank:0.19, L_foot:0.08, R_foot:0.44, L_hand:0.30 |
| 11 | ALLFOURS → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.095 / l_hip_roll_joint | L_shank:0.25, L_foot:0.11, R_foot:0.30, L_hand:0.34 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.089 / l_hip_roll_joint | L_shank:0.50, R_foot:0.20, L_hand:0.30 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.103 / l_hip_roll_joint | L_foot:0.24, R_foot:0.34, L_hand:0.42 |
| 14 | HAND_SUPPORT → TRANSITION | STATICS_GAP | True | 0.089 / r_hip_pitch_joint | L_shank:0.55, L_foot:0.20, R_foot:0.25 |
| 15 | TRANSITION → CROUCH | DYNAMIC_UNRESOLVED | True | 0.160 / l_thigh_joint | L_foot:0.41, R_foot:0.59 |
| 16 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.163 / l_calf_joint | L_foot:0.44, R_foot:0.56 |
| 17 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.159 / l_thigh_joint | L_foot:0.45, R_foot:0.55 |
| 18 | CROUCH → CROUCH | DYNAMIC_UNRESOLVED | True | 0.157 / l_thigh_joint | L_foot:0.42, R_foot:0.58 |
| balance | CROUCH → STAND | DYNAMIC_UNRESOLVED | True | 0.115 / l_hip_pitch_joint | L_foot:0.46, R_foot:0.54 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → ALLFOURS (1.12–1.20 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.20–1.40 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (1.40–1.80 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.80–1.90 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (1.90–2.20 s) | STATICS_GAP | True |
| ALLFOURS → TRANSITION (2.20–2.60 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (2.60–2.70 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → HAND_SUPPORT (2.70–3.00 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → ALLFOURS (3.00–3.20 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (3.20–3.30 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → HAND_SUPPORT (3.30–3.40 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → ALLFOURS (3.40–3.50 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (3.50–3.60 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (3.60–3.70 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (3.70–6.40 s) | STATICS_GAP | True |
| TRANSITION → ALLFOURS (6.40–6.60 s) | STATICS_GAP | True |
| ALLFOURS → TRANSITION (6.60–6.80 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (6.80–6.90 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → HAND_SUPPORT (6.90–8.00 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (8.00–9.70 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (9.70–10.40 s) | STATICS_GAP | True |
| CROUCH → TRANSITION (10.40–12.10 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → CROUCH (12.10–12.30 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → STAND (12.30–14.83 s) | DYNAMIC_UNRESOLVED | True |

38 distinct graph-edge witnesses replayed and explained; 34 certified destination configurations.

Terminal sampled tail: QUASI_STATIC, span 0.98 s, peak KE 0.000066 J.

Transition labels: DYNAMIC_UNRESOLVED=15, STATICS_GAP=5.
Sampled peak KE: 3.892 J; minimum shell height: -0.79 mm; minimum self distance: -0.65 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.

## prone / 2

Initial passive settle: True; QP attempts: 1.
Budget exhausted. Final retained state misses handover gates: up_z=0.6402 (requires >0.9); nonfoot_load=27.0642 (requires <8).
These are measured algorithmic stopping constraints, not a proof of physical impossibility.
Last QP attempt ended at base=0.099 m, up_z=0.307, nonfoot load=75.0 N after about 1.20 s.

Sampled mode sequence: PRONE → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → TRANSITION → ALLFOURS → HAND_SUPPORT → TRANSITION → CROUCH → HAND_SUPPORT → TRANSITION → CROUCH → HAND_SUPPORT → CROUCH → HAND_SUPPORT → SIDE_R → HAND_SUPPORT → ALLFOURS.
Replay full-snapshot maximum error: 0.

| Block | Endpoint modes | Label | End cert | End rho / joint | End LP loads (mg) |
|---|---|---|---|---|---|
| 0 | PRONE → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.072 / l_hip_pitch_joint | L_foot:0.22, R_foot:0.22, L_forearm:0.15, R_forearm:0.12, head:0.28 |
| 1 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.239 / l_elbow_joint | L_shank:0.22, R_shank:0.14, L_hand:0.32, R_hand:0.32 |
| 2 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.097 / r_shoulder_pitch_joint | L_shank:0.12, L_foot:0.11, R_shank:0.20, R_foot:0.09, L_hand:0.25, R_hand:0.22 |
| 3 | ALLFOURS → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.218 / r_shoulder_pitch_joint | L_shank:0.37, R_shank:0.28, R_hand:0.35 |
| 4 | ALLFOURS → HAND_SUPPORT | STATICS_GAP | True | 0.130 / l_hip_roll_joint | L_foot:0.38, R_foot:0.22, R_hand:0.40 |
| 5 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.194 / l_hip_roll_joint | L_foot:0.31, R_foot:0.13, R_hand:0.56 |
| 6 | HAND_SUPPORT → HAND_SUPPORT | STATICS_GAP | True | 0.106 / l_hip_roll_joint | L_foot:0.31, R_foot:0.33, R_hand:0.36 |
| 7 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.116 / l_hip_roll_joint | L_foot:0.30, R_foot:0.34, R_hand:0.35 |
| 8 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.097 / l_hip_roll_joint | L_foot:0.38, R_foot:0.23, R_hand:0.39 |
| 9 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.104 / l_hip_roll_joint | L_foot:0.37, R_foot:0.24, R_hand:0.39 |
| 10 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.091 / l_hip_roll_joint | L_foot:0.38, R_foot:0.21, R_hand:0.41 |
| 11 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.079 / l_hip_pitch_joint | L_foot:0.39, R_foot:0.16, R_hand:0.45 |
| 12 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.095 / l_hip_roll_joint | L_foot:0.39, R_foot:0.25, R_hand:0.36 |
| 13 | HAND_SUPPORT → HAND_SUPPORT | STATICS_GAP | True | 0.094 / l_hip_roll_joint | L_foot:0.37, R_foot:0.26, R_hand:0.38 |
| 14 | HAND_SUPPORT → HAND_SUPPORT | STATICS_GAP | True | 0.164 / l_hip_roll_joint | L_foot:0.46, R_foot:0.20, R_hand:0.33 |
| 15 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.141 / l_hip_roll_joint | L_foot:0.45, R_foot:0.22, R_hand:0.33 |
| 16 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.160 / l_hip_pitch_joint | L_foot:0.47, R_foot:0.21, R_hand:0.31 |
| 17 | HAND_SUPPORT → HAND_SUPPORT | DYNAMIC_UNRESOLVED | True | 0.101 / l_hip_roll_joint | L_foot:0.45, R_foot:0.22, R_hand:0.33 |
| balance | HAND_SUPPORT → ALLFOURS | DYNAMIC_UNRESOLVED | True | 0.081 / l_hip_roll_joint | trunk:0.03, L_foot:0.27, R_thigh:0.38, R_shank:0.03, R_foot:0.07, L_hand:0.11, R_hand:0.10 |

| Mode edge (sampled interval) | Label | Destination certified |
|---|---|---|
| PRONE → ALLFOURS (1.16–1.24 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.24–1.44 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (1.44–1.54 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.54–1.64 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (1.64–1.74 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (1.74–2.04 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (2.04–2.24 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → TRANSITION (2.24–2.54 s) | DYNAMIC_UNRESOLVED | True |
| TRANSITION → ALLFOURS (2.54–2.84 s) | DYNAMIC_UNRESOLVED | True |
| ALLFOURS → HAND_SUPPORT (2.84–3.84 s) | STATICS_GAP | False |
| HAND_SUPPORT → TRANSITION (3.84–3.94 s) | STATICS_GAP | True |
| TRANSITION → CROUCH (3.94–4.04 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → HAND_SUPPORT (4.04–4.14 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → TRANSITION (4.14–5.14 s) | STATICS_GAP | False |
| TRANSITION → CROUCH (5.14–5.24 s) | STATICS_GAP | True |
| CROUCH → HAND_SUPPORT (5.24–5.34 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → CROUCH (5.34–9.14 s) | DYNAMIC_UNRESOLVED | True |
| CROUCH → HAND_SUPPORT (9.14–9.34 s) | STATICS_GAP | True |
| HAND_SUPPORT → SIDE_R (9.34–11.54 s) | STATICS_GAP | True |
| SIDE_R → HAND_SUPPORT (11.54–11.64 s) | DYNAMIC_UNRESOLVED | True |
| HAND_SUPPORT → ALLFOURS (11.64–12.77 s) | DYNAMIC_UNRESOLVED | True |

47 distinct graph-edge witnesses replayed and explained; 41 certified destination configurations.

Terminal sampled tail: DYNAMIC_UNRESOLVED, span 0.94 s, peak KE 0.235377 J.

Transition labels: DYNAMIC_UNRESOLVED=15, STATICS_GAP=4.
Sampled peak KE: 3.965 J; minimum shell height: -0.57 mm; minimum self distance: -0.67 mm.
Full per-sample reasons, torque ratios, polygons and margins are in the explanation JSON.
