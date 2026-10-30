| Scenario | Severity | Monitor | Drifted columns | min p | Macro-F1 | ΔF1 vs control | Verdict |
|---|---|---|---|---|---|---|---|
| control | - | ok | - | 5.6e-01 | 1.000 | -0.000 | ok (no false alarm) |
| blur | 1 (1.0) | ALERT | sharpness, confidence, label | 3.9e-37 | 0.527 | -0.473 | detected, harmful |
| blur | 2 (2.0) | ALERT | sharpness, confidence, label | 2.2e-80 | 0.356 | -0.644 | detected, harmful |
| blur | 3 (4.0) | ALERT | sharpness, confidence, label | 3.4e-88 | 0.141 | -0.859 | detected, harmful |
| brightness | 1 (20) | ok | - | 8.9e-03 | 0.978 | -0.022 | undetected, benign |
| brightness | 2 (40) | ALERT | brightness, confidence, label | 1.2e-08 | 0.867 | -0.133 | detected, harmful |
| brightness | 3 (70) | ALERT | brightness, confidence, label | 3.5e-21 | 0.694 | -0.306 | detected, harmful |
| contrast | 1 (0.8) | ALERT | contrast, sharpness | 1.6e-03 | 0.993 | -0.007 | detected, benign (model robust) |
| contrast | 2 (0.6) | ALERT | contrast, sharpness, confidence, label | 2.1e-10 | 0.842 | -0.158 | detected, harmful |
| contrast | 3 (0.4) | ALERT | contrast, sharpness, confidence, label | 1.9e-29 | 0.381 | -0.619 | detected, harmful |
| noise | 1 (5) | ALERT | sharpness | 1.3e-20 | 0.963 | -0.037 | detected, benign (model robust) |
| noise | 2 (15) | ALERT | contrast, sharpness, confidence, label | 2.2e-80 | 0.679 | -0.321 | detected, harmful |
| noise | 3 (30) | ALERT | contrast, sharpness, confidence, label | 2.2e-80 | 0.367 | -0.633 | detected, harmful |

_135 images per window, reference = other half of the validation split. 'harmful' = macro-F1 drop > 0.05; with this few images small F1 differences are noise._
