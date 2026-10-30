**Validation set:** 1529 samples, 30 errors; accuracy 0.980, macro-F1 0.980.

| Class | Recall |
|---|---|
| crazing | 0.988 |
| inclusion | 0.984 |
| patches | 0.972 |
| pitted_surface | 0.988 |
| rolled-in_scale | 1.000 |
| scratches | 0.949 |

| True | Predicted | Count |
|---|---|---|
| scratches | inclusion | 13 |
| patches | crazing | 7 |
| crazing | patches | 3 |
| inclusion | scratches | 2 |
| inclusion | pitted_surface | 2 |
| pitted_surface | inclusion | 2 |
| pitted_surface | scratches | 1 |

**Error causes (manual review of 30 errors):**

| Cause | Count | Share |
|---|---|---|
| unusual_lighting | 9 | 30% |
| model_error_other | 6 | 20% |
| background_texture | 5 | 17% |
| small_or_faint_defect | 5 | 17% |
| ambiguous_class_boundary | 2 | 7% |
| low_contrast | 2 | 7% |
| label_noise | 1 | 3% |
