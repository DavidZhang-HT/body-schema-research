# F3 attention per phase (run a, verdict NOT_FOUND)

Phase-mean of each head; rows = phase, columns = body-part group / joint group.

## Support attention a_cur (contact-link)
| phase | head | back | pelvis/butt | arms(elbow) | hands | thighs | knees/shanks | feet |
|---|---|---|---|---|---|---|---|---|
| P1_head_up | 0.00 | 0.16 | 0.21 | 0.00 | 0.01 | 0.21 | 0.25 | 0.16 |

## Support attention a_des (desired, contact-agnostic)
| phase | head | back | pelvis/butt | arms(elbow) | hands | thighs | knees/shanks | feet |
|---|---|---|---|---|---|---|---|---|
| P1_head_up | 0.07 | 0.36 | 0.24 | 0.02 | 0.00 | 0.11 | 0.12 | 0.07 |

## Certified load share (statics LP)
| phase | head | back | pelvis/butt | arms(elbow) | hands | thighs | knees/shanks | feet |
|---|---|---|---|---|---|---|---|---|
| P1_head_up | 0.00 | 0.21 | 0.35 | 0.00 | 0.03 | 0.29 | 0.02 | 0.09 |

## Motion attention a_joint
| phase | hip | knee | ankle | shoulder | elbow |
|---|---|---|---|---|---|
| P1_head_up | 0.17 | 0.14 | 0.12 | 0.49 | 0.08 |

## Effort attention alpha (normalised joint marginal)
| phase | hip | knee | ankle | shoulder | elbow |
|---|---|---|---|---|---|
| P1_head_up | 0.34 | 0.09 | 0.17 | 0.31 | 0.08 |

## Snapshots at phase entry / exit (a_cur argmax group, load-share argmax group)
| phase | entry a_cur top | entry load top | exit a_cur top | exit load top |
|---|---|---|---|---|
| P1_head_up | back (0.96) | back (0.81) | knees/shanks (0.43) | pelvis/butt (0.64) |
