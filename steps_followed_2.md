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
- Hyperparameters are tuned with Optuna (TPE, seed 42, 250 trials, same CV folds for every trial and model).
- Features: R, F, M only for now. Extra features come later, one at a time.
- One candidate at a time: 7a Ridge on log1p, 7b XGBoost on raw GBP, 7c Tweedie, 7d Hurdle.

STEP 7a: Ridge on log1p(target), train CV (250 Optuna trials; tuned: alpha only)

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|
| mean | 0.00 | 944 | 0.17 | 1.00 |
| past_spend | 0.75 | 523 | 0.74 | 1.00 |
| LogRidge, plain expm1 (alpha 1,050) | 0.40 | 586 | 0.72 | 0.28 |
| LogRidge + Duan smearing (alpha 4,829) | 0.55 | 575 | 0.72 | 0.99 |

- Does not beat past spend on R2 (0.55 vs 0.75) or MAE (575 vs 523). Ranking is a tie (0.72 vs 0.74, within the +-0.04 noise).
- Why: trained for accuracy in log units, but R2 in GBP is decided by the biggest customers. Past spend is linear in GBP, so it gets them right almost for free.
- Plain expm1 gives a typical value, not the average (Jensen): predicts only 28% of the real total. Smearing fixes the total (0.99) but is fragile: the factor S is about 13-17 because of huge log residuals (half the customers are zero), and at small alpha the smeared R2 is hugely negative.
- First run used alpha range up to 1,000 and the best alpha sat on the edge; widened to 1e6. Going from 30 to 250 trials barely moved the alphas; smearing R2 rose 0.51 -> 0.55.
- CV result is stable over 10 fold seeds (smearing R2 0.558 +- 0.011).

STEP 7b: XGBoost on raw GBP, train CV (250 Optuna trials; tuned: n_estimators, max_depth, learning_rate, subsample, colsample_bytree)

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|
| past_spend | 0.75 | 523 | 0.74 | 1.00 |
| Ridge on log1p + smearing (7a) | 0.55 | 575 | 0.72 | 0.99 |
| XGBoost raw GBP, tuning score | 0.50 | 600 | 0.71 | 1.00 |
| XGBoost raw GBP, honest estimate (10 other fold splits) | 0.48 +- 0.02 | | | |

- Best settings: 480 trees, depth 5, learning rate 0.011, subsample 0.30, colsample 0.57. Three settings ended at the edge of their ranges (flat plateau; widening further is unlikely to change the story). subsample range was widened from 0.5-1.0 to 0.3-1.0 after it hit the edge.
- More tuning did not help for real: 30 -> 250 trials raised the tuning score 0.45 -> 0.50 but the honest estimate only 0.475 -> 0.478. The extra trials mostly bought a better-looking number (the more combinations tried, the more likely one looks good on these folds by luck).
- Does not beat past spend, and is below the 7a Ridge (0.48 vs 0.56 on the honest estimate). Totals are well calibrated (1.00). Ranking is a tie (0.71 vs 0.74).
- Why it loses: trees cannot predict higher than the biggest values they saw. The largest XGBoost prediction is about 54,000 while real spend reaches 110,668; for the customer with 318,751 past spend (actual 92,477) it predicts 32,078 and the past-spend rule 120,142. About 66% of XGBoost's squared error comes from the 10 biggest customers (55% for past spend).
- Feature importance: Frequency 0.54, Monetary 0.41, Recency 0.05.

Standing by R2 (train CV): past_spend 0.75 > Ridge+smearing 0.55 > XGBoost 0.50 (honest 0.48) > Ridge plain 0.40 > mean 0.00.

STEP 7c: XGBoost with the Tweedie objective, train CV (250 Optuna trials; tuned: the 5 XGBoost settings + tweedie_variance_power)

Tweedie = the same boosted trees with a loss built for "many exact zeros plus a long right tail". Predictions are never negative, and errors are weighted relative to customer size.

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual | R2 without top 10 |
|---|---|---|---|---|---|
| past_spend | 0.75 | 523 | 0.74 | 1.00 | 0.55 |
| Ridge on log1p + smearing (7a) | 0.55 | 575 | 0.72 | 0.99 | 0.50 |
| XGBoost raw GBP (7b) | 0.50 | 600 | 0.71 | 1.00 | 0.32 |
| XGBoost Tweedie (7c) | 0.51 (honest 0.49 +- 0.01) | 534 | 0.74 | 0.92 | 0.54 |

- Best settings: 395 very shallow trees (depth 2, the minimum allowed), learning rate 0.014, variance power 1.57. A simple model; even depth 1 might do as well.
- Best model on MAE (534) and ties past spend on ranking (0.744 vs 0.739). On par with past spend once the 10 biggest customers are removed (R2 0.54 vs 0.55), far above 7b (0.32). First candidate that is competitive with the spreadsheet rule for the typical customer.
- But R2 stays near 0.5: still a tree model, it cannot predict the giants (largest prediction about 39,000 vs 110,668 actual; 76% of its squared error comes from the 10 biggest customers).
- Under-predicts the total by 8% (0.92).
- By our rule (R2 picks the winner) past spend still leads; among models Ridge+smearing (0.55) is ahead of Tweedie (0.51).

Standing by R2 (train CV): past_spend 0.75 > Ridge+smearing 0.55 > Tweedie 0.51 (honest 0.49) > XGBoost 0.50 (honest 0.48) > Ridge plain 0.40 > mean 0.00.
Next: 7d Hurdle. Ideas for the giants problem, to discuss later: predict the ratio target / Monetary; add features that separate big customers.

WHY TREE MODELS (7b XGBoost, 7c Tweedie) STRUGGLE WITH THE "GIANTS"

Giants = the few customers with very large past spend. They decide R2 in GBP, which is why the trees stay near R2 0.5 while the past-spend rule gets 0.75.

- A tree answers with the average of a bucket: customers are sorted into buckets by questions (Monetary above X? Frequency above Y?) and each bucket answers with the average future spend of the training customers in it.
- So it cannot go above what it has seen (an average cannot exceed the largest value in it, and boosting adds only small bounded corrections). A customer with 400,000 past spend gets the same answer as one with 150,000 if both fall in the top bucket.
- And the top bucket is tiny and mixed: among the 10 biggest past spenders, actual future spend ranges from 4,005 to 110,668, so the tree answers with roughly their average, flat.
- A linear rule (0.37 x Monetary) is a ruler: bigger input, proportionally bigger output, no cap. Customer 18102: past spend 318,751, actual 92,477, past-spend rule 120,142, XGBoost 32,065, Tweedie 31,740.
- Made worse here by Optuna choosing very cautious trees (slow, small samples, shallow), and by having only a few dozen giants with 20% of them held out in each CV fold.
- Measured: largest prediction XGBoost raw about 54,000, Tweedie about 39,000, vs 110,668 actual. Share of squared error from the 10 biggest customers: past spend 55%, XGBoost 66%, Ridge+smearing 72%, Tweedie 76%.
- Wording correction: "cannot go above what it has seen" is true in the strict sense, but the practical problem is flat averages of a few mixed customers. It can be reduced, not removed.
- Ideas to reduce it (not tried): predict the ratio target / Monetary; add features that separate the giants; linear model at the leaves or blend trees with the linear rule.

STEP 7d: Hurdle model (two stages), train CV (250 Optuna trials per variant; 10 settings tuned jointly)

Stage 1 classifier: will they buy at all? -> P(buy). Stage 2 regressor: how much, if they buy? (trained only on the 2,039 buyers). Prediction = P(buy) x expected spend. Two variants of stage 2: raw GBP, and log1p with Duan smearing.

| | R2 (tuning) | R2 (honest) | MAE (GBP) | Capture@20% | total predicted / total actual | R2 without top 10 |
|---|---|---|---|---|---|---|
| past_spend | 0.75 | 0.74 | 523 | 0.74 | 1.00 | 0.55 |
| Ridge on log1p + smearing (7a) | 0.55 | 0.56 | 575 | 0.72 | 0.99 | 0.50 |
| XGBoost raw GBP (7b) | 0.50 | 0.48 | 600 | 0.71 | 1.00 | 0.32 |
| XGBoost Tweedie (7c) | 0.51 | 0.49 | 534 | 0.74 | 0.92 | 0.54 |
| Hurdle, stage 2 raw (7d) | 0.53 | 0.50 | 589 | 0.73 | 1.00 | 0.41 |
| Hurdle, stage 2 log + smear (7d) | 0.57 | 0.52 | 534 | 0.75 | 0.97 | 0.49 |

- Hurdle (log + smear) is the best candidate on tuning R2, MAE (tied with Tweedie) and Capture@20%. On the honest R2, Ridge + smearing is still ahead (0.56 vs 0.52), but the gap is about one standard deviation of the fold-split noise.
- Stage 1 is not the bottleneck: AUC 0.76-0.78, predicted share of buyers matches the real share (0.481). A PERFECT "who buys" model would lift R2 only from 0.567 to 0.589. The weak point is the amount among the biggest buyers.
- Smearing is stable in stage 2 (factor 1.32 vs 13.4 in 7a) because buyers have no zeros. That is why log + smear beats raw in the Hurdle.
- Splitting the problem helped: both Hurdle variants beat their single-model counterparts on the tuning score.
- Tuning score is optimistic (0.567 vs 0.522 honest) because 10 settings were tuned at once.
- Same giants problem as 7b/7c: largest prediction about 50,000-54,000 vs 110,668 actual; about 70% of the error from the 10 biggest customers.

STEP 7 overall: no candidate beats the past-spend rule on R2 (0.75). Standing by honest R2: past_spend 0.74 > Ridge+smearing 0.56 > Hurdle log+smear 0.52 > Hurdle raw 0.50 > Tweedie 0.49 > XGBoost 0.48 > Ridge plain 0.43 > mean 0.00. By MAE and ranking the best models (Tweedie, Hurdle) are level with past spend.
Next (to discuss): attack the giants (predict the ratio target / Monetary; add features that separate big customers), then evaluate the chosen model on test once.

STEP 7e: the ratio idea (predict a multiplier, not pounds), train CV (250 Optuna trials per variant)

scale = max(Monetary, 50); multiplier = future spend / scale (what the trees learn); prediction = predicted multiplier x scale. The scale comes from the customer, so giants get big predictions automatically. A constant multiplier of 0.37 IS the past-spend baseline, so the model can only improve on it by correcting it. Customers weighted by scale ** weight_power (Optuna picks 0 to 2). Two variants: squared error and Tweedie.

| | R2 (tuning) | R2 (honest) | MAE (GBP) | Capture@20% | total predicted / total actual | R2 without top 10 |
|---|---|---|---|---|---|---|
| past_spend | 0.75 | 0.74 | 523 | 0.739 | 1.00 | 0.55 |
| Hurdle log+smear (7d), best tree-only | 0.56 | 0.52 | 536 | 0.746 | 0.97 | 0.49 |
| Ratio XGBoost, squared error | 0.758 | 0.731 | 518 | 0.742 | 1.01 | 0.56 |
| Ratio XGBoost, Tweedie | 0.760 | 0.736 | 508 | 0.741 | 0.99 | 0.57 |

- The giants problem is fixed: R2 from about 0.5 to about 0.76; largest prediction 101,000-117,000 (was about 50,000) vs 110,668 real; share of error from the 10 biggest customers 54-55% (was 66-77%).
- But it only MATCHES past spend: honest R2 0.73-0.74 vs 0.744. Paired bootstrap vs past spend: MAE gain +15 GBP (95% interval 6 to 28, small but real, about 3%); R2 +0.013 (interval -0.013 to 0.044, no clear gain); Capture@20% +0.002 (a tie).
- Why so little: Optuna chose learning rates at the bottom of the range, so the model barely moves from the constant multiplier. What it learned is a recency adjustment: multiplier 0.41 for the most recent quartile down to 0.31 for the oldest; Frequency barely matters (0.36-0.38). The real multipliers vary far more (0.12 to 2.4 for the 5th-95th percentile) and R, F, M cannot predict who will be on the high or low end.
- Conclusion: the models are no longer the limit; the information is. We need features that say something new: order sizes (AOV), purchase interval, trend in spending, return rate, tenure.

Standing by honest R2: Ratio Tweedie 0.74 = Ratio squared 0.73 = past_spend 0.74 > Ridge+smearing 0.56 > Hurdle 0.52 > Tweedie 0.49 > XGBoost 0.48.
Next: step 8, add features one at a time to the ratio model, each with a measured effect.
