The three evaluation metrics (all in GBP, original scale)

| Metric | Question it answers |
|---|---|
| R2 | How well do the predicted amounts fit? 0 = no better than the average, 1 = perfect. |
| MAE | Typical error in GBP; not dominated by a few huge customers. |
| Capture@20% | If we contact the top 20% ranked by the model, what share of all future money do they hold? Random is about 20%. |

Dropped to keep things simple: RMSE (says the same as R2), Spearman, confidence-interval columns. Kept as helpers, not metrics: the ceiling (best possible Capture@20%), the "without the 10 biggest customers" check, and the bootstrap for close calls.

STEP 6: baselines (fitted on train only)

1. mean: everyone gets the train-average target.
2. past_spend: predicted future spend = k x Monetary (floored at 0), k fitted on train. k only sets the GBP scale; the ranking is just "sort by past spend".

| | Train CV: mean | Train CV: past spend | Test: mean | Test: past spend |
|---|---|---|---|---|
| R2 | 0.00 | 0.75 | 0.00 | 0.71 |
| MAE (GBP) | 944 | 523 | 1,086 | 613 |
| Capture@20% | 0.17 | 0.74 | 0.16 | 0.74 |

Ceiling for Capture@20% (perfect ranking): 0.86 (train), 0.83 (test).
Without the 10 biggest customers, past spend: R2 0.55 (CV) / 0.63 (test), Capture@20% 0.69 / 0.67.

The bar for step 7: beat past spend by more than the noise (about +-0.04 on capture).
