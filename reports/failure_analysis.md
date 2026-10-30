# Failure Analysis

> Template. Replace every TODO with real results. Delete this line when done.
> All analysis uses the **validation** split. The test split is evaluated once, at the end.

## 1. Setup
- Model analysed: TODO (e.g. LightGBM on HOG+LBP+GLCM / ResNet18 fine-tuned)
- Split: stratified 70/15/15, seed 42; val n = TODO
- Reproduce: `make failure-analysis`

## 2. Where the model fails
<!-- paste the generated tables from reports/failure_analysis_generated.md -->
TODO: overall metrics, per-class recall, top confusion pairs.

![errors](figures/errors_TRUE_as_PRED.png)

TODO: 2-3 sentences. Which classes are confused, and does it match what you see in the images?

## 3. Why: manual review of every error
Each error was inspected by hand and assigned one cause (`reports/error_review.csv`).

TODO: paste the cause table. Then for the top 2 causes, one paragraph each with an example image.
Note how many errors you reviewed. If < 30, say the shares are indicative only.

## 4. Label noise
- Method: confident learning (cleanlab) on 5-fold out-of-fold probabilities, train+val only.
- Candidates flagged: TODO (X% of train+val). Confirmed by manual review: TODO.
- Common confusions among flagged samples: TODO.

## 5. Fix and measured effect
| Change | Val macro-F1 before | after | Notes |
|---|---|---|---|
| Drop N confirmed mislabeled samples from train | TODO | TODO | val labels untouched |
| TODO (e.g. targeted augmentation for top cause) | TODO | TODO | |

Caveat to state explicitly: if val itself contains mislabeled samples, the measured gain is understated.
If you also re-labeled val, report both the original-label and corrected-label numbers.

## 6. What I did not fix / limitations
- TODO
