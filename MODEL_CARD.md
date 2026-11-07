# Model card: customer spend forecast (CLV_stepwise)

## 1. What the model does

For every customer who bought in the last 12 months, it predicts **how much money they will keep (net of returns) in the next 6 months**, in GBP. The prediction is meant for ranking and sizing customers: who to prioritise, and roughly how much is at stake.

It is **not** a model of who will respond to a campaign, and it says nothing about margin. See section 8.

Short version of the result: the model is "about 0.37 x past spend", nudged by roughly +/-12% for recency and purchase rhythm. On new data it is about 3% better than that plain rule in MAE, and equal in ranking and in R2 (section 5).

## 2. Data

- UCI Online Retail II (a UK online gift retailer), invoice lines from 2009-12-01 to 2011-12-09. 1,067,371 rows, 53,628 invoices, 5,942 customers.
- Cleaning (step 2): rows without a Customer ID (23%), non-positive prices, and internal stock codes (M, ADJUST, BANK CHARGES, TEST001 and similar, 1,428 rows) are removed. Postage (POST, C2, DOT) is kept, because customers pay it. Cancelled invoices ("C...") become **returns**, kept as negative rows dated at the cancellation. Row counts add up: 804,739 purchase rows (5,854 customers) and 18,136 return rows (2,465 customers).
- "Money kept" = purchases minus returns, summed over a date window.

## 3. Snapshot design (no leakage by construction)

| | Features from | Target (spend in) | Customers | Share with zero target |
|---|---|---|---|---|
| Train | 2009-12-01 to 2010-11-30 | 2010-12-01 to 2011-05-31 | 4,242 | 52% |
| Test | 2010-06-01 to 2011-05-31 | 2011-06-01 to 2011-11-30 | 4,326 | 43% |

- Out-of-time: the test target period is entirely after everything the model saw. The two targets do not overlap.
- Windows are half-open: features use rows strictly before the cutoff date; the cutoff day belongs to the target.
- Target is floored at 0 (28 customers have negative net spend); the unfloored value is kept as `target_net`.
- Only customers who bought during the feature window are scored. New customers cannot be predicted from history.
- Heavy right tail: the 10 biggest customers hold 18% to 21% of all target money.

## 4. Model

**Headline model C** (`outputs/model_c.joblib`, spec in `outputs/model_spec.json`):

- prediction = multiplier x scale, with scale = max(past net spend, 50 GBP)
- the multiplier comes from boosted trees (XGBoost, Tweedie loss) trained on `future spend / scale`, with customers weighted by scale^0.85
- 50 trees, depth 2, learning rate 0.0177, subsample 0.58, column sample 0.71, Tweedie variance power 1.88 (Optuna, 250 trials, objective = R2 in 5-fold CV on train only)
- columns: Recency, Frequency, Monetary (net spend), IPI (mean days between purchase days; blank if one purchase day), OverdueRatio (Recency / max(IPI, 1); blank if IPI blank). Blanks are handled natively by XGBoost, so no fill values are fitted on train and nothing can leak into test.

Why this shape: ordinary tree models predict bucket averages, so they cannot produce the very large numbers the few giant customers need (largest prediction about 50,000 vs 110,000 real). Predicting a multiplier and multiplying by the customer's own past spend removes that limit (step 7e).

How it was chosen (5-fold CV on train; R2 / MAE; later numbers are from fresh fold seeds where noted):

| Model | R2 | MAE (GBP) | Note |
|---|---|---|---|
| mean baseline | about 0 | far worse | step 6 |
| past spend (0.37 x Monetary) | 0.75 | 523 | step 6 |
| log-Ridge + smearing (7a) | 0.55 | 575 | |
| XGBoost on raw GBP (7b) | 0.50 (honest 0.48) | | cannot predict giants |
| XGBoost Tweedie (7c) | 0.51 (honest 0.49) | 534 | |
| Hurdle, log + smearing (7d) | 0.57 (honest 0.52) | 534 | |
| Ratio XGBoost, Tweedie (7e), R,F,M | 0.76 (honest 0.74) | 508 | wins on R2 |
| **C: ratio Tweedie, R,F,M + IPI + Overdue** | 0.752 (fresh seeds 20 to 29, sd 0.008) | 504 | headline |

"Honest" = same settings scored on 10 other fold splits; tuning scores are optimistic.

Features tried and kept or rejected (step 8, rule below): AOV rejected (MAE significantly worse); the interval + overdue ratio pair was kept provisionally; spend trend, return rate, tenure and product count were rejected. **Keep rule E**: a feature is kept only if a bootstrap over customers shows R2 or MAE significantly better, neither significantly worse, and a practical size (MAE gain at least 5 GBP or R2 gain at least 0.005). On fresh splits the pair's gain was +4.0 GBP MAE, interval [-2.4, +9.7]: it no longer passed. Model A (same model, R, F, M only) is statistically indistinguishable from C; **C is the headline model by the user's choice**, A is the simpler equivalent.

## 5. Results on the test snapshot (scored once; models fitted on train only)

4,326 customers, 43% zero target. Capture@20% ceiling (perfect ranking) = 0.832. 95% intervals are bootstraps over customers.

| Model | R2 | MAE (GBP) | Capture@20% | R2 / MAE without the 10 biggest customers |
|---|---|---|---|---|
| mean | -0.002 | 1,086 | 0.163 | -0.000 / 887 |
| past spend | 0.707 [0.575, 0.776] | 613 [535, 703] | 0.739 [0.695, 0.782] | 0.631 / 511 |
| A: R,F,M | 0.707 [0.600, 0.767] | 596 [521, 685] | 0.739 [0.697, 0.782] | 0.671 / 492 |
| **C: headline** | **0.716** [0.594, 0.782] | **594** [521, 683] | **0.740** [0.695, 0.783] | 0.673 / 492 |

- C vs past spend: MAE better by 18.9 GBP [+10.0, +29.1] (about 3%, real); R2 +0.010 [+0.001, +0.025] (barely); Capture identical (-0.000).
- C vs A: no difference we can trust (R2 +0.008 [-0.013, +0.021]; MAE +1.8 GBP [-3.3, +7.5]).
- Train to test: R2 0.75 to 0.72, Capture 0.745 to 0.740: a modest drop, no collapse. MAE rises (504 to 594) because test customers spend more.
- Totals are under-predicted: predicted money = 69% of actual. Largest actual 183,530 vs largest prediction 95,530; 83% of C's squared error comes from the 10 biggest customers.

**Targeting view** (test): the top 20% by model C spend 3,543 GBP each in the next 6 months, 3.7x the average of 958 GBP (lift +270%, interval +248% to +290%), and hold 74% of all future money. Picking the 20% by past spend gives 3,539 GBP: **the model does not beat sorting by past spend for targeting.** Recency alone is clearly worse (2,858 GBP).

## 6. Segments (step 10)

A 2x2 grid of predicted value x probability of buying again, with cut-offs chosen by the user and fixed on train:

- high value = predicted spend of about 721 GBP or more (80th percentile of train out-of-fold predictions)
- likely to return = logistic probability (R, F, M) above 0.5. The classifier has AUC 0.778 (5-fold CV on train; recency alone 0.693; XGBoost no better) and is well calibrated on train.

| Segment | Test customers | Share of money | Mean future spend | Bought again |
|---|---|---|---|---|
| Protect | 779 (18%) | 71.5% | 3,802 GBP | 93% |
| Win back | 25 (0.6%) | 1.0% | 1,637 GBP | 64% |
| Grow | 835 (19%) | 12.0% | 594 GBP | 76% |
| Low priority | 2,687 (62%) | 15.6% | 240 GBP | 41% |

The segments separate customers in the expected order on both train and test. The top box is almost the same as a hand rule (top 20% past spend and bought in the last 90 days: 769 of 804 customers overlap). The Win back box is nearly empty because valuable customers are almost always the ones likely to return.

## 7. What drives the predictions (SHAP, step 11)

SHAP on the multiplier: the interval (33%), recency (31%) and overdue ratio (22%) give 86% of the total; Frequency and Monetary only 14% inside the trees (Monetary already acts as the scale). The prediction ranks customers almost exactly like past spend (rank correlation 0.993). SHAP shows what the model uses, not what helps: the interval is the largest nudge, yet its accuracy gain could not be proven.

## 8. Known limitations

1. **Not causal.** It ranks who will spend, not who is persuadable. No campaign data exists; incremental revenue (uplift) is unknown. Break-even arithmetic in `business_decisions.md` rests on assumed costs.
2. **Revenue, not margin.** No product cost or discount data.
3. **The model is only a little better than past spend.** About 3% in MAE; no gain in ranking.
4. **The giants dominate.** 10 customers hold about a fifth of the money and cause most of the error; the model under-predicts them. Treat them as named accounts.
5. **Two snapshots only.** Findings rest on one train period and one test period; a third period would show whether they are stable.
6. **Probabilities shift over time.** 57% of test customers bought again vs 48% in train, so "likely to return > 50%" is stricter on test than intended (Grow bought again 76%, Low priority 41%, vs 61% and 30% on train). Ranking holds; the levels moved.
7. **Seasonality.** One retailer, two years, a Christmas-heavy business; the test target (Jun to Nov) and train target (Dec to May) are not alike.
8. **Test-set use.** The test snapshot was scored once for the model (step 9) and afterwards also used to describe the segments and the targeting view, and the "better Win back list" (high past spend and no purchase in 90 days: 156 test customers, 1,619 GBP each) was found by looking at test. These later uses did not change the model, but the Win back list is a lead, not a confirmed result.
9. **Tuning scores are optimistic.** Optuna scores run about 0.02 to 0.03 R2 above the same settings on fresh fold splits. All headline CV numbers use fresh seeds.

## 9. Audit: choices made to avoid known defects

These come from the defects found in the earlier one-shot version of this project (`ML_proj2_CIP_CLV`).

| Risk | What we did |
|---|---|
| Leakage from the future | Features strictly before the cutoff; target window after it; two non-overlapping snapshots. |
| Return rate counted line items, not invoices | Where used, counted by invoice and within the feature window only. |
| Model chosen on the test set | Selection by 5-fold CV on train only; test scored once for the headline model. |
| Fill values computed on the wrong batch | No fill values: blanks go straight to XGBoost. |
| "A/B simulation" without treatment, cost or incremental revenue | Reframed as a targeting simulation; stated plainly that it measures ranking, not ROI. |
| Segmentation by K-Means on features chosen to raise a score | Value x return-probability grid built from model outputs; cut-offs chosen before looking at results. |
| Features added to hit a target | Features added one at a time and kept only under rule E, with a bootstrap over customers. |
| Over-reading fold-split significance | Fold splits reuse the same customers, so their t-values overstate certainty; verdicts come from a customer bootstrap. |

## 10. Files and reproduction

- Model and numbers: `outputs/model_c.joblib`, `outputs/logistic_return.joblib`, `outputs/model_spec.json`, `outputs/metrics.json` (written by `src/step15_save.py`; it reproduces the test and targeting numbers above).
- Code: `src/step1_load_explore.py` to `src/step12_targeting.py`, one script per step; notebooks in `notebooks/`.
- Notes: `steps_followed_1.md` to `steps_followed_4.md`, `baseline_metrics_1.md`, `model_tried_summary_2.md`, `feature_adding_3.md`, `model_interpretation.md`, `business_decisions.md`.
- Environment: conda env `ml_clv`, run with `PYTHONNOUSERSITE=1`.

```
PYTHONNOUSERSITE=1 python src/step9_test.py        # test evaluation
PYTHONNOUSERSITE=1 python src/step15_save.py       # saves model + metrics.json
```
