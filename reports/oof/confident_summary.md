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

**Error causes (manual review of 17 errors):**

| Cause | Count | Share |
|---|---|---|
| small_or_faint_defect | 4 | 24% |
| label_noise | 3 | 18% |
| ambiguous_class_boundary | 3 | 18% |
| low_contrast | 2 | 12% |
| unusual_lighting | 2 | 12% |
| model_error_other | 2 | 12% |
| background_texture | 1 | 6% |

_Only 17 errors reviewed: treat shares as indicative, not precise._
