# Adding features: model choice and protocol

Status before this step: step 7 is finished (candidates 7a to 7e, see model_tried_summary_2.md). Only Recency, Frequency, Monetary (R, F, M) are used so far. The test snapshot has NOT been touched.

## 1. Which model we carry forward: Ratio XGBoost with the Tweedie loss (step 7e)

| Why this one | |
|---|---|
| Best MAE of any candidate | 508 GBP, against 523 for past spend |
| Fixes the giants problem | largest prediction 101,445 against 110,668 real (7b to 7d: about 50,000) |
| Honest R2 level with past spend | 0.736 +- 0.023, against 0.744 |
| Handles the zeros | Tweedie suits "many exact zeros plus a heavy tail"; totals calibrated (0.99) |
| Contains past spend as a special case | a constant multiplier of 0.37 IS the baseline, so it can only be improved by INFORMATION, which is what we add now |

Why not the others: 7a to 7d all lose on the few giant customers (trees answer with averages of what they saw). The squared-error ratio variant is almost as good (honest R2 0.731) and is kept only as a final cross-check.

Status: a working choice made on train cross-validation only. Not a final winner yet; test is untouched.

## 2. How features are added: one at a time, with a fair test each time

The trap is noise. Re-tuning 250 trials after every feature gives tuning scores that wobble by about 0.02, which is larger than the effect one feature will probably have. So features are judged with FIXED settings on the SAME fold splits.

1. Screen each feature alone. Take R, F, M plus ONE new feature. Keep the 7e settings fixed (no re-tuning). Score it on the same 10 fold splits as the baseline model. Report the PAIRED gain (new minus baseline) for each split.
2. Keep rule, fixed in advance (before looking at any result): keep a feature if R2 improves in at least 9 of the 10 fold splits. MAE is the tiebreaker if R2 is unchanged.
3. Combine the keepers, best first, re-checking after each addition (forward selection).
4. Re-tune ONCE at the end with the final feature set (250 trials) and confirm the honest estimate on other fold splits.
5. Only then touch test, once, for the final model. Past spend and the other models are reported on test for reference.

Rules for building features (lessons from the old project's audit):
- Every feature is computed ONLY from rows before the cutoff, by ONE function used for both the train and the test snapshot.
- Any fill value (for example a median) comes from train and is reused unchanged on test.
- Return rate is computed inside the feature window only, and by invoices, not line items (the old version leaked target-window behaviour and understated the rate about 3x).
- No feature is added just to improve a score target (the old project added LastMonthSpend partly to raise a silhouette score).

## 3. The candidate features

| Feature | Definition | Why it might help |
|---|---|---|
| AOV | spend / number of orders | tells big-basket buyers from many-small-order buyers |
| Purchase interval (+ overdue ratio) | average days between orders; Recency / interval | a customer overdue relative to their own rhythm is more likely lapsed |
| Spend trend | recent months vs earlier months | rising or falling customers |
| Return rate | cancelled invoices / all invoices, feature window only | high returners keep less money |
| Tenure, product diversity | days since first purchase; number of distinct products | loyal and broad buyers |

Order of testing: AOV, purchase interval, spend trend, return rate, tenure and diversity.

## 4. Honest expectation

The gains may be small, or zero for some features (Frequency added almost nothing beyond Monetary in 7e). If a feature does not help, that is a valid result and it is dropped. We end with either the simplest model that is measurably better than past spend, or the finding that these features cannot clearly beat it.

## 5. Why more information is the lever (from 7e)

- The ratio model barely moves away from the constant multiplier 0.37: its multipliers span 0.26 to 0.45 while the real ones span 0.12 to 2.4 (5th to 95th percentile).
- What it learned is only a recency adjustment (0.41 for the most recent quartile down to 0.31 for the oldest). Frequency barely matters (0.36 to 0.38).
- Against past spend: MAE gain +15 GBP (95% interval 6 to 28, small but real), R2 +0.013 (interval -0.013 to 0.044, no clear gain), Capture@20% +0.002 (a tie).
- So the models are no longer the limit; the information is.

---

## 6. Feature 1 tested: AOV (average order value)  [notebook: notebooks/8a_aov.ipynb, code: src/step8_features.py]

AOV = gross purchase spend / number of purchase invoices. Test: ratio Tweedie model (7e settings FIXED), R, F, M vs R, F, M + AOV, the SAME 10 fold splits (paired).

| Paired difference (new minus base) | mean | sd across splits | splits improved | paired t |
|---|---|---|---|---|
| R2 | -0.029 | 0.011 | 0 / 10 | -8.0 |
| MAE gain in GBP (base minus new) | -3.6 | 2.6 | 1 / 10 | -4.3 |
| Capture@20% | +0.0001 | 0.0017 | 5 / 10 | 0.1 |

Result: AOV makes the model WORSE, consistently. Do not add.

Why:
- It carries some signal (rank correlation with the real multiplier among buyers: AOV -0.20; for comparison Recency +0.20, Frequency -0.46, Monetary -0.45) but it overlaps with Monetary (rank correlation 0.68; Monetary is about AOV x Frequency), so the model mostly has this information already.
- Trees use the extra column to split further and fit noise.

Was the fixed-settings screen unfair to AOV? No. Re-tuning with AOV included (100 trials): tuning score 0.788 (higher than the baseline's 0.760) but only 0.689 +- 0.085 on the other fold splits (range 0.505 to 0.775), against 0.736 +- 0.025 for the baseline. Optuna picked a flexible model (depth 5) that fits the tuning folds by luck. A tuning score can be off by about 0.1, so only the paired screen on fresh fold splits is trusted.

## 7. The keep rule: what the numbers say (decision still open)

AOV is a clear "no" under any rule, so it cannot tell us which rule is best. How the candidate rules behave (sd of the paired R2 difference across splits = 0.011):

| Rule | A USELESS feature passes | True R2 gain needed to pass half the time | ... to pass 90% of the time |
|---|---|---|---|
| R2 better in >= 9 of 10 splits | 1.1% | 0.011 | 0.018 |
| R2 better in >= 8 of 10 splits | 5.5% | 0.007 | 0.014 |
| R2 better in >= 7 of 10 splits | 17.2% | 0.004 | 0.010 |

A rule is a trade-off between letting a useless feature through and missing a helpful one. Caveat: the 10 fold splits reuse the same customers, so they are not 10 independent tests; the real false-positive rate is higher than the table.

Alternative rule that uses the SIZE of the gain, not just how often it is positive: keep if the mean paired gain is positive with paired t >= 2 on R2, and MAE is not clearly worse. AOV's t is -8 (clear reject). It would accept a small but consistent gain (for example +0.005 R2 with low noise) that the 9-of-10 rule might reject.

Decision: wait for a borderline feature. Next: purchase interval and overdue ratio (8b), then compare the rules on a second example.
