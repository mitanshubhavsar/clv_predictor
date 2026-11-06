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

---

## 8. Features 2 and 3 tested: purchase interval and overdue ratio  [notebook: notebooks/8b_interval.ipynb]

- IPI = mean days between consecutive purchase DAYS. OverdueRatio = Recency / IPI (above 1 = later than the customer's own rhythm).
- 35% of customers (one purchase day) have no interval: left blank, XGBoost handles missing values itself, so there is no fill value to leak from train to test (Frequency == 1 already flags them).
- Signal check: IPI alone says nothing about the real multiplier among buyers (rank correlation +0.004); OverdueRatio +0.10. But OverdueRatio does separate who comes back: share who buy again 65-70% (early or on time), 59% (1-2x overdue), 52% (2-4x), 34% (more than 4x).

Fold-split screen (7e settings fixed, same 10 splits, new minus base):

| Variant | R2 change | R2 better in | MAE gain (GBP) | MAE better in | Capture@20% change |
|---|---|---|---|---|---|
| + IPI | -0.011 | 2/10 | -0.01 | 4/10 | +0.005 |
| + OverdueRatio | -0.010 | 0/10 | +0.9 | 7/10 | +0.001 |
| + IPI + OverdueRatio | -0.013 | 0/10 | +4.3 | 10/10 | +0.005 (10/10) |

The metrics disagree for the pair: MAE and ranking better, R2 worse.

Why R2 drops: the giants. The squared error of the 10 biggest customers is higher with the new features in all 10 splits; for the other 4,232 customers it is lower in most splits. Without the top 10, the R2 change is only -0.004 (better in 4/10 splits), about zero.

Bootstrap OVER CUSTOMERS (the 10 fold splits reuse the same customers, so their spread only measures fold-assignment noise; this measures which-customers noise):

| Feature set | R2 change (95% interval) | MAE gain GBP (95% interval) | Capture@20% change (95% interval) |
|---|---|---|---|
| + AOV | -0.026 (-0.063 to 0.000) | -6.1 (-12.6 to -0.1) worse | +0.001 (-0.003 to 0.004) |
| + IPI | -0.003 (-0.020 to 0.014) | +1.4 (-3.1 to 5.4) | +0.005 (-0.001 to 0.011) |
| + OverdueRatio | -0.007 (-0.020 to 0.004) | +1.5 (-2.5 to 5.7) | +0.002 (-0.003 to 0.006) |
| + IPI + OverdueRatio | -0.005 (-0.020 to 0.008) | +5.7 (+1.1 to +10.2) real | +0.005 (-0.002 to 0.012) |

The capture "t = 10" from the fold splits is NOT significant over customers.

## 9. The keep rule, compared on real examples (decision to take)

| Rule | AOV | IPI | OverdueRatio | IPI + OverdueRatio |
|---|---|---|---|---|
| A: R2 better in >= 9/10 splits | reject | reject | reject | reject |
| B: R2 better in >= 8/10 splits | reject | reject | reject | reject |
| C: R2 mean > 0, paired t >= 2, MAE not clearly worse | reject | reject | reject | reject |
| D: R2 or MAE significantly better over customers, none significantly worse | reject | reject | reject | KEEP |
| E: D plus a practical size (MAE gain >= 5 GBP or R2 >= +0.005) | reject | reject | reject | KEEP |

- A to C look only at R2, and R2 here is decided by about ten customers and by fold assignment. They are blind to a small, real gain in typical-customer error (MAE) and they overstate how certain 10 reused splits are.
- D and E judge each metric by a bootstrap over customers, require that nothing gets significantly worse, and (E) a minimum practical size. They are the more honest of the five, and they still reject AOV (its MAE is significantly worse).
- Caveat (multiple comparisons): four variants were tried and the pair's MAE interval only just excludes 0 (+1.1 GBP at the low end). Treat the pair as a PROVISIONAL KEEP, to be re-checked after all features are in (final re-tune and fresh fold splits).
- Honest size of the gain: about 1% of MAE for two extra columns (35% of one of them blank).

Next: 8c, spend trend and return rate.

## 10. DECISION: rule E adopted as the keep rule

Decision (user): adopt rule E for every remaining feature.

Rule E, in words: a feature (or feature group) is KEPT only if, on the 5-fold CV predictions of the train set with the 7e settings fixed:
1. R2 or MAE is significantly BETTER by a paired bootstrap over customers (95% interval entirely above 0), and
2. neither R2 nor MAE is significantly WORSE (95% interval entirely below 0), and
3. the gain has a practical size: mean MAE gain >= 5 GBP or mean R2 gain >= +0.005.

Why E and not A to C or D:
- A to C judge only R2, which is decided by about ten customers and by fold assignment; they cannot see a small real gain for typical customers, and fold-split t-values overstate certainty because the splits reuse the same customers.
- D has the right logic but accepts gains of any size; E adds a practical-size bar so a statistically real but trivial gain is not paid for with an extra column.
- E still rejects AOV (MAE significantly worse), so it is not simply "lenient".

Consequences:
- IPI + OverdueRatio: KEEP, but PROVISIONAL (four variants tried, MAE interval only just above 0; multiple-comparisons caveat). Re-check at the end on fresh fold splits after the final re-tune.
- AOV: rejected. IPI alone and OverdueRatio alone: not kept on their own (the pair is kept as one group).
- Test set stays untouched until the final model is chosen.
- Going forward each new feature is screened with: fold-split table (for information) + customer bootstrap (for the verdict), via `rule_verdicts` in src/step8_features.py (rule E is the last entry).

Next: 8c, spend trend (recent vs earlier months) and return rate (feature window only, by invoices).

## 11. Features 4 and 5 tested: spend trend and return rate  [notebook: notebooks/8c_trend_returns.ipynb]

Definitions (feature window only, before the cutoff):
- RecentShare = gross purchase spend in the last 90 days / total gross purchase spend (0 to 1; blank only if total spend <= 0).
- ReturnRate = return invoices / purchase invoices (by invoices, window only; fixes the old project's all-history version).

Tested on top of the kept model (R, F, M + IPI + OverdueRatio), 7e settings fixed, rule E:

| Added | R2 change (95% over customers) | MAE gain, GBP (95%) | Capture change | Fold splits R2 better | Rule E |
|---|---|---|---|---|---|
| RecentShare | +0.010 (-0.015 to +0.030) | +1.5 (-5.6 to +9.9) | +0.001 | 10/10 (t = 6.2) | reject |
| ReturnRate | +0.003 (-0.006 to +0.010) | +0.9 (-1.7 to +4.3) | -0.000 | 5/10 | reject |
| both | +0.014 (-0.007 to +0.035) | +1.2 (-4.8 to +8.2) | +0.001 | 10/10 | reject |

Lessons:
- RecentShare is the case that shows why we dropped rules A to C: the 10 fold splits say "better in 10/10, t = 6", so A, B and C would all keep it. But the splits reuse the same customers, so that only shows the fold assignment is not the cause. Across customers the interval for R2 spans -0.015 to +0.030 (the giants decide it) and for MAE it spans zero. We cannot tell it from noise, so it is rejected.
- ReturnRate: no signal (about 80% of customers have 0 returns in the window). On plain R, F, M it was even significantly worse in MAE (-4.0 GBP).
- Kept set unchanged: R, F, M + IPI + OverdueRatio (provisional).

Next: 8d, tenure / product diversity (the last candidates), then re-tune once and confirm on fresh splits.
