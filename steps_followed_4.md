# Steps followed, part 4: segmentation (step 10)

Plan (from the old project's src/segmentation.py): a 2x2 grid of predicted value (model C) x probability of buying again, giving four segments with four different actions: Protect (high value, likely to return), Win back (high value, unlikely), Grow (low value, likely), Low priority (low value, unlikely). The old project dropped K-Means after finding it just split on "any spend in the last 30 days?", a rule a marketer can write by hand.

Step 10 sub-steps: (a) classifier for "will buy again", (b) cut-offs and the 2x2 segments, (c) do the segments mean anything (actual future spend per segment), (d) compare with a hand-written RFM rule, (e) test once.

DECISION (user): option 2, business-sized cut-offs. "High value" = top 20% by predicted value (same cut Capture@20% is judged on); "likely to return" = probability above 50%. Segments will be unbalanced, but each cut-off means something.

## STEP 10a: classifier P(spends > 0 in the next 6 months)   [src/step10_repurchase.py, notebooks/10a_repurchase.ipynb]
Target: 1 if floored 6-month spend > 0 (train: 48.1% yes; test 57%). Scored by AUC, 5-fold CV on train, mean over 10 fold seeds. Test untouched.

| Model | AUC |
|---|---|
| recency only | 0.693 |
| logistic regression, R,F,M (log-scaled) | 0.778 |
| XGBoost, R,F,M (fixed small trees) | 0.775 |
| XGBoost, R,F,M + IPI + Overdue | 0.776 |

- Recency alone has real signal; Frequency and Monetary lift AUC by 0.085.
- Plain logistic regression ties XGBoost, so the relationship is simple and trees add nothing.
- The IPI + Overdue pair adds +0.0015 AUC (bootstrap -0.003 to +0.006): nothing, consistent with step 8.
- Calibration is good: in 10 equal bins predicted and actual rates agree within a few points. Brier 0.190 vs 0.250 for always guessing the base rate.
- Proposed classifier: logistic regression on R,F,M (simplest, best, calibrated). To confirm with the user.

Calibration (plain words): when the model says "70%", about 70 of 100 such customers should really come back. Our top bin: predicted 91.5%, actual 93.4%; lowest bin: predicted 16%, actual 16%. It matters because the Protect/Grow cut-off is "probability above 50%", which only means "more likely than not" if the numbers are honest. AUC = ranking, calibration = the numbers themselves. Example in model_interpretation.md.

## STEP 10b: the 2x2 segments (train, out-of-fold)   [src/step10_segments.py, notebooks/10b_segments.ipynb]
High value = top 20% of model C's predicted spend; likely to return = logistic probability > 0.5 (classifier from 10a). Note: files from 10a were renamed to step10_repurchase_segmentation.py and 10a_repurchase_segmentation.ipynb; imports updated.

| Segment | Customers | Share of customers | Mean actual future spend (GBP) | Bought again | Share of future money |
|---|---|---|---|---|---|
| Protect | 818 | 19.3% | 2,738 | 85.7% | 73.9% |
| Win back | 31 | 0.7% | 448 | 25.8% | 0.5% |
| Grow | 1,038 | 24.5% | 407 | 61.3% | 14.0% |
| Low priority | 2,355 | 55.5% | 150 | 29.5% | 11.6% |

- The segments are real: spend and return rate differ a lot between them, in the order we would expect.
- Win back is nearly empty (31 customers). Value and probability move together (correlation 0.29): only 3.7% of high-value customers are unlikely to return. The box for "valuable but drifting away" hardly exists at these cut-offs.
- Against a hand-written rule (top 20% past spend x bought in last 90 days): the top box is almost the same (815 of 849 customers overlap; its Protect holds 72.5% of the money vs 73.9%). The model's gain is in the lower boxes: Grow vs Low priority separate 61% vs 30% buying again, the rule's 48% vs 26%.
- Honest reading: the model's top box adds little over a rule a marketer could write; the probability axis is what sharpens the rest.

Summary of how the segments were made and decided (details in model_interpretation.md):
- Honest (out-of-fold) predictions per train customer: value from model C, probability of buying again from the logistic regression.
- Two yes/no answers per customer: High value? = predicted value in the top 20%; Likely to return? = probability > 50% (both cut-offs chosen by the user, not tuned).
- The pair picks the block: yes/yes Protect, yes/no Win back, no/yes Grow, no/no Low priority.
- Blocks are then checked against what really happened (actual spend, share who bought again, share of money). Classifier choice and the 90-day hand rule were my choices; results were looked at once, train only.

Exact value cut-off (train): high value = predicted 6-month spend >= about 721 GBP (80th percentile of model C's out-of-fold predictions; median 233, 95th percentile 2,106).

## STEP 10c: single test-set check of the segments   [src/step10_segments_test.py, notebooks/10c_segments_test.ipynb]
Fixed on train: model C and logistic fitted on all train; high value = predicted spend >= 721 GBP; likely = probability > 0.5. Applied once to test (4,326 customers, 56.9% bought again).

| Segment | Customers | Share | Mean future spend (GBP) | Bought again | Share of future money |
|---|---|---|---|---|---|
| Protect | 779 | 18.0% | 3,802 | 92.7% | 71.5% |
| Win back | 25 | 0.6% | 1,637 | 64.0% | 1.0% |
| Grow | 835 | 19.3% | 594 | 75.7% | 12.0% |
| Low priority | 2,687 | 62.1% | 240 | 40.6% | 15.6% |

- The structure holds: Protect keeps ~72% of the money (train 74%); order Protect > Grow > Low priority is preserved on spend and return rate.
- Probabilities are too pessimistic on test: test customers buy again more (57% vs 48%), so Grow bought again 76% (train 61%) and Low priority 41% (train 30%). Ranking works, levels shifted (base-rate shift).
- Win back is tiny (25) and not a drifting group: they spent 1,637 on average and 64% bought again, more than Grow. The classifier called them unlikely and was wrong for them. The hand rule's win-back (156 customers; high past spend, no purchase in 90 days) behaves alike (1,619; 69%).
- Sensitivity (high value = top 20% of test predictions): essentially the same. Top box vs hand rule: 769 of 804 customers the same.

## STEP 11: SHAP, what drives model C?   [src/step11_shap.py, notebooks/11_shap.ipynb]
Model C = multiplier x scale (scale = max(past spend, 50)). The trees learn only the multiplier (Tweedie, log link), so SHAP explains log(multiplier): +0.1 = multiplier about 10% higher than the average customer. Fitted and explained on train; SHAP values add up to the model output (checked in the code).

| Feature | Mean abs SHAP | Share | Direction |
|---|---|---|---|
| IPI (purchase interval) | 0.046 | 33% | shorter interval, higher multiplier |
| Recency | 0.043 | 31% | more recent, higher multiplier |
| OverdueRatio | 0.030 | 22% | more overdue, lower multiplier |
| Monetary | 0.011 | 8% | weak |
| Frequency | 0.010 | 7% | weak |

- Past spend does almost all the work: prediction ranks customers like Monetary (rank correlation 0.993). The trees nudge the multiplier only within 0.31 to 0.40 (average 0.355, about +/-12%).
- IPI, Recency and OverdueRatio = 86% of the SHAP total.
- Old goal "top 5 features cover >= 70%" is meaningless with exactly 5 features (100% by construction); top 3 cover 86%.
- Three customers (waterfalls in the notebook): biggest spender: past spend 245,028, multiplier 0.390, prediction 95,456, actual 110,668. Typical: past spend 629, multiplier 0.370, prediction 233, actual 0. Lapsed (longest recency): past spend 130, multiplier 0.349, prediction 45, actual 969.
- Caution: SHAP shows what the model uses, not what helps. IPI is the biggest nudge, yet step 8 and the fresh-seed check could not prove it improves accuracy.
