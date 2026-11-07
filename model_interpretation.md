# Model interpretation notes

## Step 10: segmentation, in plain words

The idea: instead of one number per customer, put each customer in one of four boxes that map onto four actions.

|  | Likely to buy again (probability > 50%) | Unlikely (probability <= 50%) |
|---|---|---|
| High value (top 20% by predicted spend) | Protect: keep them happy, no discount needed | Win back: valuable but drifting away, run a win-back campaign |
| Not high value | Grow: upsell / cross-sell | Low priority: do not spend effort |

Cut-offs chosen (option 2): top 20% by predicted value, because the whole project is judged on the top 20% (Capture@20%); probability above 50% because that is where "more likely than not" starts. The boxes will be different sizes, and that is fine: "high value" is small by design.

Two numbers per customer:
- Value = model C's prediction (GBP in the next 6 months). It is mostly past spend with a mild adjustment, so the value axis is mostly "who spent a lot".
- Probability of buying again = a separate small classifier. This is where the new information is.

### The classifier (10a): what the words mean
- Target: 1 if the customer spends anything (net, above 0) in the next 6 months, else 0. About 48% of train customers are 1.
- AUC: pick one customer who did buy again and one who did not; AUC is the chance the model gives the buyer the higher score. 0.5 = coin flip, 1.0 = perfect. Ours: 0.78.
- Calibration: when the model says "70%", do about 70% of those customers really come back? Needed because the 50% cut-off only makes sense if the numbers are honest. Ours are within a few points in all 10 bins.
- Brier score: the average squared gap between the predicted probability and what happened (0 or 1); lower is better. Ours 0.190; guessing 48% for everyone gives 0.250.
- What drives it: Recency alone gives AUC 0.69; adding Frequency and Monetary gives 0.78. Customers who bought recently, often and for a lot are likely to buy again, which is what you would expect. The interval and overdue-ratio columns add nothing.
- A simple logistic regression does as well as XGBoost, so we keep the simple one.

### Calibration: a simple practical example

Calibration asks: when the model says "70%", do about 70 out of 100 of those customers really come back?

Weather example. Two forecasters each say "80% chance of rain" on 100 different days.
- Forecaster A: it rained on 79 of those days. Their 80% means 80%. Well calibrated.
- Forecaster B: it rained on only 40. Their "80%" really means 40%. Overconfident, badly calibrated.
Both can rank days equally well (same AUC, "which day is rainier"), but only A's percentages can be taken at face value.

Our customers. Take the 425 customers the model scored in the top bin (average predicted 0.915):
- Predicted: 91.5% will buy again. Actual: 93.4% did. Close, so the number can be trusted.
- A "bin" = a group: sort all customers by predicted probability and cut into 10 equal groups of about 424.
- Lowest bin (the 10% least likely to return): predicted 15.8%, actual 16.0%.
- Middle bin (5th of 10): predicted 48.8%, actual 49.1%.
- 8th bin: predicted 68.4%, actual 63.2% (the biggest gap in the table, 5 points).

Why it matters here: our "likely to return" cut-off is "probability above 50%". If the model were overconfident (saying 55% when the truth is 30%), the Protect and Grow boxes would be full of customers who mostly do not come back.

AUC = does the model order customers correctly (ranking). Calibration = is the percentage itself true (the numbers). The 2x2 needs both.

### What the segments showed (train, out-of-fold)
- Protect: 19% of customers, 74% of the future money, 86% buy again. Win back: only 31 customers (0.7%). Grow: 25%, 61% buy again. Low priority: 56%, 30% buy again, 12% of the money.
- Why Win back is almost empty: customers who spent a lot recently are almost always the ones likely to come back (value and probability are correlated). Very few are "valuable but drifting". A real win-back list would need a looser rule, e.g. high past spend and a long gap since the last purchase.
- A simple hand rule (top 20% past spend and bought in the last 90 days) finds almost the same Protect group (815 of 849 customers the same). The model adds sharper separation among everyone else.

### How the segmentation was built and decided (step 10b)

How it was built:
1. Honest predictions for every train customer: each prediction comes from a model that never saw that customer (5-fold out-of-fold, fold seed 0). Two predictions per customer: value = model C's predicted GBP spend for the next 6 months; probability = the logistic regression's chance of buying again.
2. Two lines (the user's option 2): high value = predicted value in the top 20% (value >= its 80th percentile); likely to return = probability > 0.5.
3. Each customer's two answers pick one of four blocks.
4. Check against what really happened: per block, actual future spend, share who bought again, share of total future money. This is the only test of whether the blocks mean anything.

How it was decided:
- Cut-offs (top 20%, 50%): chosen by the user; not tuned, no other values tried.
- Classifier: proposed by me (logistic regression: simplest, best AUC 0.778 vs 0.775, well calibrated); accepted by the user moving on.
- Value: model C (the user's headline model).
- Hand-written comparison rule (top 20% past spend and bought in the last 90 days): invented by me as a yardstick; the 90 days is arbitrary.
- Results were looked at once and reported as they came out (the nearly empty Win back box is a finding, not something fixed by moving a cut-off). Train only; test not used yet.

### Which customer goes in which block

Each customer gets two yes/no answers: High value? (predicted spend in the top 20%) and Likely to return? (probability > 50%).

| High value? | Likely to return? | Block |
|---|---|---|
| Yes | Yes | Protect |
| Yes | No | Win back |
| No | Yes | Grow |
| No | No | Low priority |

In the code this is the NAMES dictionary in src/step10_segments.py, mapping (high, likely) to the block name.

Made-up examples, to show the logic only:

| Customer | Predicted spend | Rank | Probability | High value? | Likely? | Block |
|---|---|---|---|---|---|---|
| A | high | top 5% | 0.92 | Yes | Yes | Protect |
| B | high | top 15% | 0.30 | Yes | No | Win back |
| C | low | middle | 0.70 | No | Yes | Grow |
| D | very low | bottom | 0.15 | No | No | Low priority |

Meaning in practice: Protect = keep happy, no discounts needed. Win back = valuable but unlikely to return, run a win-back campaign. Grow = modest spend but loyal, upsell / cross-sell. Low priority = do not spend effort.
The two questions are independent: value comes from model C, return probability from the classifier; neither model knows about the blocks.

Exact cut-off (train, out-of-fold, model C): "high value" means a predicted 6-month spend of about 721 GBP or more (the 80th percentile). For scale: the median prediction is 233 GBP, the 95th percentile 2,106 GBP; the 80th percentile of past spend (Monetary) is 2,029 GBP, which is why the model's predictions (about 0.37 x past spend, adjusted) cut in at a much smaller number.

### Segments on the test set (step 10c)
- The four blocks keep their shape on new data: Protect is 18% of customers and holds 71.5% of the money (train 74%); the order Protect > Grow > Low priority holds for spend and for return rate.
- Probabilities ran too low on test: 57% of test customers bought again versus 48% in train, so "50%" is a stricter line than intended; Grow actually bought again 76% (train 61%), Low priority 41% (train 30%). The model ranks customers correctly, but the percentages are not exact when buying habits shift over time.
- Win back (25 customers) is not really "drifting": they spent 1,637 GBP on average and 64% bought again, more than Grow. High past spend with no purchase in the last 90 days (the hand rule, 156 customers) gives a similar group.
