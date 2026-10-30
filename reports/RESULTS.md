# Results

_Generated 2026-10-30 17:01 UTC from artifacts/ and reports/ at commit 217c4ae._

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

| Category | Image AUROC | Pixel AUROC | Precision@thr | Recall@thr | Bank size |
|---|---|---|---|---|---|
| bottle | 1.000 | 0.983 | 0.97 | 1.00 | 14739 |
| metal_nut | 1.000 | 0.986 | 0.97 | 1.00 | 15523 |
| capsule | 0.968 | 0.987 | 0.97 | 0.96 | 15444 |
| **mean** | 0.989 | 0.986 | | | |

## PatchCore vs supervised on unseen defect types

### Unseen defect types (held out of supervised training): PatchCore vs `handcrafted_lgbm`

| Category | Held-out type | n defect / good | PatchCore AUROC | Supervised AUROC | Δ (PC − sup) [95% CI] | PC TPR@5%FPR | Sup TPR@5%FPR |
|---|---|---|---|---|---|---|---|
| bottle | broken_large | 20 / 20 | 1.000 | 0.970 | +0.030 [+0.000, +0.088] | 1.00 | 0.90 |
| bottle | broken_small | 22 / 20 | 1.000 | 0.893 | +0.107 [+0.023, +0.221] | 1.00 | 0.50 |
| bottle | contamination | 21 / 20 | 1.000 | 0.795 | +0.205 [+0.086, +0.355] | 1.00 | 0.57 |
| metal_nut | bent | 25 / 22 | 1.000 | 0.596 | +0.404 [+0.249, +0.571] | 1.00 | 0.12 |
| metal_nut | color | 22 / 22 | 1.000 | 0.562 | +0.438 [+0.273, +0.624] | 1.00 | 0.09 |
| metal_nut | flip | 23 / 22 | 1.000 | 0.455 | +0.545 [+0.368, +0.723] | 1.00 | 0.00 |
| metal_nut | scratch | 23 / 22 | 1.000 | 0.492 | +0.508 [+0.334, +0.672] | 1.00 | 0.26 |
| capsule | crack | 23 / 23 | 0.981 | 0.798 | +0.183 [+0.055, +0.323] | 0.91 | 0.04 |
| capsule | faulty_imprint | 22 / 23 | 0.962 | 0.875 | +0.087 [-0.038, +0.215] | 0.82 | 0.32 |
| capsule | poke | 21 / 23 | 0.971 | 0.824 | +0.147 [+0.010, +0.294] | 0.95 | 0.10 |
| capsule | scratch | 23 / 23 | 0.932 | 0.907 | +0.025 [-0.096, +0.147] | 0.65 | 0.48 |
| capsule | squeeze | 20 / 23 | 1.000 | 0.728 | +0.272 [+0.122, +0.430] | 1.00 | 0.00 |
| **mean over 12 folds** | | | 0.987 | 0.741 | +0.246 | 0.94 | 0.28 |

PatchCore ahead in 12/12 folds (CI excludes 0 in its favour: 9; in the supervised model's favour: 0).

### Seen defect types (held-out images of types the supervised model trained on): PatchCore vs `handcrafted_lgbm`

| Category | Held-out type | n defect / good | PatchCore AUROC | Supervised AUROC | Δ (PC − sup) [95% CI] | PC TPR@5%FPR | Sup TPR@5%FPR |
|---|---|---|---|---|---|---|---|
| bottle | broken_large | 22 / 20 | 1.000 | 0.811 | +0.189 [+0.061, +0.334] | 1.00 | 0.68 |
| bottle | broken_small | 21 / 20 | 1.000 | 0.926 | +0.074 [+0.005, +0.176] | 1.00 | 0.76 |
| bottle | contamination | 21 / 20 | 1.000 | 0.993 | +0.007 [+0.000, +0.031] | 1.00 | 0.95 |
| metal_nut | bent | 35 / 22 | 1.000 | 0.678 | +0.322 [+0.187, +0.466] | 1.00 | 0.43 |
| metal_nut | color | 37 / 22 | 1.000 | 0.732 | +0.268 [+0.141, +0.402] | 1.00 | 0.38 |
| metal_nut | flip | 36 / 22 | 1.000 | 0.532 | +0.468 [+0.323, +0.602] | 1.00 | 0.17 |
| metal_nut | scratch | 36 / 22 | 1.000 | 0.763 | +0.237 [+0.131, +0.369] | 1.00 | 0.64 |
| capsule | crack | 44 / 23 | 0.963 | 0.844 | +0.120 [+0.000, +0.243] | 0.84 | 0.14 |
| capsule | faulty_imprint | 45 / 23 | 0.970 | 0.851 | +0.119 [+0.017, +0.234] | 0.87 | 0.47 |
| capsule | poke | 45 / 23 | 0.965 | 0.819 | +0.146 [+0.031, +0.279] | 0.82 | 0.18 |
| capsule | scratch | 44 / 23 | 0.978 | 0.882 | +0.096 [+0.008, +0.192] | 0.93 | 0.39 |
| capsule | squeeze | 46 / 23 | 0.960 | 0.837 | +0.123 [-0.000, +0.262] | 0.83 | 0.11 |
| **mean over 12 folds** | | | 0.986 | 0.806 | +0.181 | 0.94 | 0.44 |

PatchCore ahead in 12/12 folds (CI excludes 0 in its favour: 9; in the supervised model's favour: 0).

### Unseen defect types (held out of supervised training): PatchCore vs `resnet18`

| Category | Held-out type | n defect / good | PatchCore AUROC | Supervised AUROC | Δ (PC − sup) [95% CI] | PC TPR@5%FPR | Sup TPR@5%FPR |
|---|---|---|---|---|---|---|---|
| bottle | broken_large | 20 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| bottle | broken_small | 22 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| bottle | contamination | 21 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| metal_nut | bent | 25 / 22 | 1.000 | 0.953 | +0.047 [+0.000, +0.118] | 1.00 | 0.88 |
| metal_nut | color | 22 / 22 | 1.000 | 0.942 | +0.058 [+0.002, +0.143] | 1.00 | 0.86 |
| metal_nut | flip | 23 / 22 | 1.000 | 1.000 | +0.000 [-0.000, +0.000] | 1.00 | 1.00 |
| metal_nut | scratch | 23 / 22 | 1.000 | 0.984 | +0.016 [+0.000, +0.053] | 1.00 | 0.96 |
| capsule | crack | 23 / 23 | 0.981 | 0.953 | +0.028 [-0.021, +0.096] | 0.91 | 0.70 |
| capsule | faulty_imprint | 22 / 23 | 0.962 | 0.953 | +0.010 [-0.071, +0.097] | 0.82 | 0.86 |
| capsule | poke | 21 / 23 | 0.971 | 0.988 | -0.017 [-0.079, +0.035] | 0.95 | 0.95 |
| capsule | scratch | 23 / 23 | 0.932 | 0.968 | -0.036 [-0.125, +0.036] | 0.65 | 0.74 |
| capsule | squeeze | 20 / 23 | 1.000 | 0.959 | +0.041 [+0.004, +0.098] | 1.00 | 0.75 |
| **mean over 12 folds** | | | 0.987 | 0.975 | +0.012 | 0.94 | 0.89 |

PatchCore ahead in 6/12 folds (CI excludes 0 in its favour: 3; in the supervised model's favour: 0).

### Seen defect types (held-out images of types the supervised model trained on): PatchCore vs `resnet18`

| Category | Held-out type | n defect / good | PatchCore AUROC | Supervised AUROC | Δ (PC − sup) [95% CI] | PC TPR@5%FPR | Sup TPR@5%FPR |
|---|---|---|---|---|---|---|---|
| bottle | broken_large | 22 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| bottle | broken_small | 21 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| bottle | contamination | 21 / 20 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 1.00 | 1.00 |
| metal_nut | bent | 35 / 22 | 1.000 | 0.979 | +0.021 [+0.000, +0.060] | 1.00 | 0.94 |
| metal_nut | color | 37 / 22 | 1.000 | 0.998 | +0.002 [+0.000, +0.011] | 1.00 | 0.97 |
| metal_nut | flip | 36 / 22 | 1.000 | 0.981 | +0.019 [+0.000, +0.051] | 1.00 | 0.89 |
| metal_nut | scratch | 36 / 22 | 1.000 | 0.975 | +0.025 [+0.000, +0.072] | 1.00 | 0.94 |
| capsule | crack | 44 / 23 | 0.963 | 0.981 | -0.018 [-0.075, +0.024] | 0.84 | 0.89 |
| capsule | faulty_imprint | 45 / 23 | 0.970 | 0.984 | -0.014 [-0.058, +0.034] | 0.87 | 0.96 |
| capsule | poke | 45 / 23 | 0.965 | 0.993 | -0.028 [-0.076, +0.010] | 0.82 | 0.96 |
| capsule | scratch | 44 / 23 | 0.978 | 0.990 | -0.012 [-0.058, +0.023] | 0.93 | 0.93 |
| capsule | squeeze | 46 / 23 | 0.960 | 0.976 | -0.016 [-0.079, +0.035] | 0.83 | 0.78 |
| **mean over 12 folds** | | | 0.986 | 0.988 | -0.002 | 0.94 | 0.94 |

PatchCore ahead in 4/12 folds (CI excludes 0 in its favour: 0; in the supervised model's favour: 0).

_Small evaluation sets: wide CIs are expected. Supervised scores are averaged over training seeds._

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
- PatchCore reaches a mean image AUROC of 0.989 over 3 MVTec AD categories.
