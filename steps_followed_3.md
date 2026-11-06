STEP 8: adding features (plan, agreed before any feature is built)

Model carried forward: Ratio XGBoost with the Tweedie loss (step 7e). Reasons: best MAE (508 vs 523 for past spend), fixes the giants problem, honest R2 0.736 (level with past spend 0.744), handles the zeros, and contains the past-spend rule as a special case, so it can only be improved by information. Working choice on train CV only; test is untouched.

Protocol (full detail in feature_adding_3.md):

| Step | What |
|---|---|
| 1. Screen | R, F, M plus ONE new feature, 7e settings fixed (no re-tuning), same 10 fold splits as the baseline; look at the paired gain per split |
| 2. Keep rule (fixed in advance) | keep a feature if R2 improves in at least 9 of the 10 fold splits; MAE is the tiebreaker |
| 3. Combine | add the keepers best first, re-checking after each |
| 4. Re-tune once | 250 Optuna trials with the final feature set; confirm on other fold splits |
| 5. Test once | only for the final model |

Why fixed settings: re-tuning after every feature wobbles the score by about 0.02, more than a single feature's effect.

Candidate features, in the order they will be tested:

| Feature | Definition |
|---|---|
| AOV | spend / number of orders |
| Purchase interval (+ overdue ratio) | average days between orders; Recency / interval |
| Spend trend | recent months vs earlier months |
| Return rate | cancelled invoices / all invoices, feature window only, by invoices |
| Tenure, product diversity | days since first purchase; distinct products |

Rules: features use only rows before the cutoff, one function for train and test; fill values come from train; no feature is added just to raise a score target.

Expectation: gains may be small or zero for some features; dropping a feature that does not help is a valid result.

STEP 8a: AOV tested (details in feature_adding_3.md, notebook notebooks/8a_aov.ipynb)

Ratio Tweedie model, 7e settings fixed, R,F,M vs R,F,M + AOV, same 10 fold splits:

| Paired difference (new minus base) | mean | splits improved | paired t |
|---|---|---|---|
| R2 | -0.029 | 0 / 10 | -8.0 |
| MAE gain (GBP) | -3.6 | 1 / 10 | -4.3 |
| Capture@20% | +0.0001 | 5 / 10 | 0.1 |

- AOV makes the model worse in every split. Do not add. It carries some signal (-0.20 with the multiplier) but overlaps with Monetary (rank correlation 0.68).
- Re-tuning with AOV does not rescue it: tuning score 0.788 but 0.689 +- 0.085 on other splits (baseline 0.736). A tuning score can be off by about 0.1.
- Keep rule: you found 9-of-10 strict. Trade-off, with the observed noise (sd 0.011): 9/10 lets 1.1% of useless features through and needs a true gain of about 0.011 R2 to pass half the time; 8/10: 5.5% and 0.007; 7/10: 17.2% and 0.004. Alternative: keep if mean paired gain > 0 with paired t >= 2 on R2 and MAE not clearly worse. AOV is rejected by all of them, so the rule stays OPEN until a borderline feature (next: purchase interval and overdue ratio, 8b).

STEP 8b: purchase interval (IPI) and overdue ratio tested (details in feature_adding_3.md, notebook notebooks/8b_interval.ipynb)

IPI = mean days between purchase days (35% blank, XGBoost handles missing values, no fill value). OverdueRatio = Recency / IPI. Signal: IPI alone says nothing about the multiplier (+0.00); OverdueRatio separates who comes back (65-70% buy again when on time, 34% when more than 4x overdue) but barely shifts the multiplier among buyers (+0.10).

Effect versus R, F, M (bootstrap over customers, 7e settings fixed):

| Feature set | R2 change | MAE gain (GBP) | Capture@20% change |
|---|---|---|---|
| + AOV | -0.026 | -6.1 (worse) | +0.001 |
| + IPI | -0.003 | +1.4 | +0.005 |
| + OverdueRatio | -0.007 | +1.5 | +0.002 |
| + IPI + OverdueRatio | -0.005 | +5.7 (interval +1.1 to +10.2: real) | +0.005 (not significant) |

- The pair is the borderline case: MAE better (about 1%), ranking and R2 not significantly different, but R2 worse in 10/10 fold splits. The R2 dip is the giants (top-10 error up in every split; without the top 10 the R2 change is -0.004).
- The fold-split t-values overstated the evidence (they reuse the same customers); the bootstrap over customers is the honest one.

Keep rule, compared on the four feature sets: A (R2 better in >= 9/10 splits), B (>= 8/10) and C (R2 mean > 0, t >= 2, MAE not worse) reject all four. D (R2 or MAE significantly better over customers, none significantly worse) and E (D plus a practical size: MAE gain >= 5 GBP or R2 >= +0.005) keep only IPI + OverdueRatio, and still reject AOV. A to C look only at R2, which is decided by about ten customers; D and E are more honest.
Caveat: four variants were tried and the pair's MAE interval only just excludes 0, so it is a PROVISIONAL KEEP to re-check after all features are in.
Next: 8c, spend trend and return rate.

DECISION (step 8): rule E adopted as the keep rule for all remaining features.
Keep a feature (or group) only if, by bootstrap over customers on the CV predictions: (1) R2 or MAE is significantly better, (2) neither is significantly worse, and (3) the gain is practical: MAE gain >= 5 GBP or R2 gain >= +0.005.
Why: A to C look only at R2 (decided by ~10 customers, and reused fold splits overstate certainty); D ignores size; E adds a size bar and still rejects AOV.
Status: AOV rejected; IPI + OverdueRatio kept as a provisional group (re-check at the end on fresh fold splits after the final re-tune). Test set still untouched.
Next: 8c, spend trend and return rate.

STEP 8c: spend trend (RecentShare = share of spend in the last 90 days) and ReturnRate (return invoices / purchase invoices, feature window only).
Result vs the kept model (rule E): RecentShare R2 +0.010 (-0.015 to +0.030), MAE +1.5 GBP (-5.6 to +9.9): reject. ReturnRate R2 +0.003, MAE +0.9 (-1.7 to +4.3): reject. Both together: reject.
Lesson: RecentShare was better in 10/10 fold splits (t = 6) and rules A to C would have kept it; over customers the gain is indistinguishable from zero. Fold splits reuse the same customers, so they overstate certainty.
Kept set unchanged: R, F, M + IPI + OverdueRatio (provisional). Notebook: 8c_trend_returns.ipynb.
Next: 8d, tenure and product diversity.

STEP 8d: Tenure (days since first purchase in window, capped at 365) and NProducts (distinct stock codes).
Rule E vs the kept model: Tenure R2 +0.017 (-0.004 to +0.044), MAE +2.2 (-3.0 to +9.0): reject (nearest miss). NProducts R2 +0.001, MAE -0.2: reject. Both: reject.
STEP 8 SUMMARY: 7 candidates tried; only IPI + OverdueRatio kept (provisional). Past spend already carries most of the information; extra columns move MAE by about 1% at best.
Next: re-tune once on the kept columns (250 trials), check on fresh fold splits, then score the final model once on test.

STEP 8 FINAL: re-tuned the ratio Tweedie model on the kept columns (250 trials), compared on 10 fresh fold seeds (10 to 19).
R2 / MAE / Capture: A (7e settings, R,F,M) 0.7503 / 510.8 / 0.742; B (7e settings, kept columns) 0.7417 / 506.4 / 0.748; C (re-tuned, kept columns) 0.7505 / 504.5 / 0.746.
C vs A over customers: R2 -0.004 [-0.021, +0.012], MAE gain +4.0 GBP [-2.4, +9.7]: below the 5 GBP bar and not significant. The provisional keep of IPI + OverdueRatio is NOT confirmed (multiple-comparisons caveat).
Recommendation: final model = A, the 7e ratio Tweedie on R, F, M only. Test set untouched.

DECISION: headline model = C (user choice); A also reported.
STEP 9, TEST (cutoff 2011-06-01, 4,326 customers, ceiling Capture 0.832), fit on train only:
mean: R2 -0.002, MAE 1,086, Capture 0.163 | past spend: 0.707, 613, 0.739 | A (R,F,M): 0.707, 596, 0.739 | C (headline): 0.716 [0.594, 0.782], 594 [521, 683], 0.740 [0.695, 0.783].
C vs past spend: MAE +18.9 GBP [+10.0, +29.1] (about 3%, real), R2 +0.010 [+0.001, +0.025], Capture -0.000 (same). C vs A: no trustworthy difference.
Without the top 10 customers: R2 0.673, MAE 492. Totals under-predicted (69% of actual); 83% of squared error is the 10 biggest customers (largest actual 183,530, largest prediction 95,530).
Train to test: modest drop, no collapse. Notebook: 9_test.ipynb.
