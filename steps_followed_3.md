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
