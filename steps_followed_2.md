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

STEP 7 decisions
- Winner is chosen by R2 in GBP (MAE breaks ties), from 5-fold CV on the train snapshot only. Test is not touched until a winner is chosen.
- Hyperparameters are tuned with Optuna (TPE, seed 42, same CV folds for every trial and model).
- Features: R, F, M only for now. Extra features come later, one at a time.
- One candidate at a time: 7a Ridge on log1p, 7b XGBoost on raw GBP, 7c Tweedie, 7d Hurdle.

STEP 7a: Ridge on log1p(target), train CV

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|
| mean | 0.00 | 944 | 0.17 | 1.00 |
| past_spend | 0.75 | 523 | 0.74 | 1.00 |
| LogRidge, plain expm1 (alpha 1,116) | 0.40 | 588 | 0.72 | 0.26 |
| LogRidge + Duan smearing (alpha 4,283) | 0.51 | 606 | 0.72 | 1.10 |

- Does not beat past spend on R2 (0.51 vs 0.75) or MAE. Ranking is a tie (0.72 vs 0.74, within the +-0.04 noise).
- Why: trained for accuracy in log units, but R2 in GBP is decided by the biggest customers. Past spend is linear in GBP, so it gets them right almost for free.
- Plain expm1 gives a typical value, not the average (Jensen): predicts only 26% of the real total. Smearing fixes the total (1.10) but is fragile: the factor S is about 13-17 because of huge log residuals (half the customers are zero), and at small alpha the smeared R2 is hugely negative.
- First run used alpha range up to 1,000 and the best alpha sat on the edge; widened to 1e6.
- CV result is stable over 10 fold seeds (R2 sd <= 0.02).
