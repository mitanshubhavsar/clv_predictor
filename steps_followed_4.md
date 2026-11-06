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
