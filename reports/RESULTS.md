# Results

_Generated 2026-10-30 15:30 UTC from artifacts/ and reports/ at commit 324ad1a._

## Headline results (NEU-DET classification)

| Model | Val macro-F1 | Test macro-F1 |
|---|---|---|
| HOG+LBP+GLCM + LogReg | 0.970 | 0.956 |
| HOG+LBP+GLCM + LightGBM | 0.989 | 0.985 |
| ResNet fine-tuned | 1.000 | 0.996 |

## Augmentation ablation

| Augmentation | Val macro-F1 (mean ± std) | Per-seed |
|---|---|---|
| none | 1.0000 ± 0.0000 | 1.000 |
| flip | 1.0000 ± 0.0000 | 1.000 |
| geometric | 1.0000 ± 0.0000 | 1.000 |
| photometric | 1.0000 ± 0.0000 | 1.000 |
| full | 1.0000 ± 0.0000 | 1.000 |

## Label noise

- cleanlab flagged **12** candidate samples; manual review not done yet (`reports/label_issues_confirmed.csv`).

## Anomaly detection (PatchCore, MVTec AD)

_Not run yet: PatchCore. Run `make patchcore`._

## PatchCore vs supervised on unseen defect types

_Not run yet: comparison. Run `make compare`._

## Deployment benchmark

| Backend | Size (MB) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) | Val acc | Val macro-F1 | Agreement |
|---|---|---|---|---|---|---|---|---|
| pytorch fp32 | 44.8 | 61.61 | 152.99 | 213.83 | 19.2 | 1.000 | 1.000 | 1.000 |
| onnxruntime fp32 | 44.7 | 34.43 | 36.31 | 49.98 | 31.6 | 1.000 | 1.000 | 1.000 |
| onnxruntime int8 | 11.3 | 20.43 | 21.55 | 26.23 | 52.6 | 0.981 | 0.981 | 0.981 |

_Intel64 Family 6 Model 165 Stepping 2, GenuineIntel · 12 cores available · 1 thread(s) · onnxruntime 1.30.0 · latency = batch 1, throughput = batch 32_
- ONNX Runtime p50 latency 34.43 ms vs PyTorch 61.61 ms (+44% change; 1 thread(s), 12 cores available).
- int8: 44.7 -> 11.3 MB, p50 34.43 -> 20.43 ms, val macro-F1 -0.019 vs fp32.

## Drift monitoring simulation

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

## Claims supported by these numbers

- On the held-out test split, LightGBM on handcrafted features scores 0.985 macro-F1 and the fine-tuned ResNet 0.996 (difference +0.011).
- On CPU, ONNX Runtime p50 latency is 34.4 ms vs 61.6 ms in PyTorch.
