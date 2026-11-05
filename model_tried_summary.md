# Models tried: summary

How every candidate is judged (step 7 decisions):
- The winner is chosen by R2 in GBP (MAE breaks ties), from 5-fold cross-validation on the train snapshot only.
- Hyperparameters are tuned with Optuna (TPE sampler, seed 42, 30 trials, the same CV folds for every trial and every model).
- Features: Recency, Frequency, Monetary only. Extra features come later, one at a time.
- The test snapshot is not touched until a winner is chosen.
- The bar to beat is the past-spend baseline: R2 0.75, MAE 523, Capture@20% 0.74 (train CV).

Candidates: 7a Ridge on log1p (done), 7b XGBoost on raw GBP, 7c Tweedie, 7d Hurdle.

---

## 7a. Ridge regression on log1p(target)

Code: `src/step7a_ridge_log.py`, `src/tuning.py`. Notebook: `notebooks/7a_ridge_log.ipynb`.

### What the model is
log1p of R, F, M -> StandardScaler -> Ridge -> predicts log1p(future spend) -> convert back to GBP -> score in GBP.

### What was tuned: one hyperparameter, alpha
Ridge has one knob, alpha: how strongly the model is held back from fitting the training data too closely. Small alpha = fits freely. Large alpha = simpler, more cautious predictions.

Optuna tried 30 alphas between 0.001 and 1,000,000 (log scale). For each one it ran the 5-fold CV and scored R2 in GBP. The alpha with the highest R2 won.

Fixed, not tuned: the features (R, F, M), taking the log of the features, standardizing them, and predicting log1p(future spend).

### The two variants
The model predicts in log units, so the prediction has to be converted back to GBP. The variants differ only in how.

| Variant | How it converts back | Idea |
|---|---|---|
| Plain | Just undo the log (expm1) | Simple, but the result is a typical value, not the average, so it comes out too low (Jensen's inequality) |
| Smearing (Duan) | Undo the log, then multiply by a correction factor = mean of exp(training residuals) | Tries to fix the "too low" problem |

Each variant got its own Optuna search, so each has its own best alpha.

### Results (train, 5-fold CV)

| | Best alpha | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|---|
| Mean baseline | none | 0.00 | 944 | 0.17 | 1.00 |
| Past-spend baseline | none | 0.75 | 523 | 0.74 | 1.00 |
| Ridge, plain expm1 | 1,116 | 0.40 | 588 | 0.72 | 0.26 |
| Ridge, smearing | 4,283 | 0.51 | 606 | 0.72 | 1.10 |

The last column is a diagnostic, not one of the three metrics.
Without the 10 biggest customers: past spend R2 0.55 / Capture 0.69; plain 0.33 / 0.67; smearing 0.37 / 0.67.

### In plain words
- Both variants lose to the simple past-spend rule on amounts: R2 0.40 and 0.51 vs 0.75, and MAE 588 and 606 vs 523.
- On picking the right people it is a tie: 0.72 vs 0.74 is inside the noise of about +-0.04.
- Smearing beats plain on R2 (0.51 vs 0.40). Plain predicts only about 26% of the real total spend; smearing brings the total to about 110%. But smearing is touchy: it only works with a large alpha, and with a small alpha it gives wildly wrong numbers.
- Large alphas won (1,116 and 4,283): the model works best when held back hard. Undoing a log makes the biggest customers' predictions explode if the model is allowed to fit freely.
- Why it loses on amounts: it is trained to be accurate in log units, but R2 in GBP is decided by the biggest customers. Past spend is linear in GBP, so it gets them right almost for free.

### Why smearing is fragile here
The smearing factor is the mean of exp(residual). About half the customers have target 0 (log1p(0) = 0), and a few huge customers are predicted far too low, so log-scale residuals reach about +8 and exp(8) is about 3,000. A handful of such values dominate the factor (S about 13-17). At alpha = 1 the plain model's in-sample R2 is already -1.9, and the smeared one is -1284.

### Checks done
- First run used an alpha range up to 1,000 and the best alpha was 989, on the edge. Widened to 1,000,000; the best alphas are now inside the range.
- Re-ran the CV with 10 different random fold assignments: R2 0.744 +- 0.006 (past spend), 0.419 +- 0.012 (plain), 0.533 +- 0.019 (smearing). Capture spread about 0.001. The gap to past spend is far bigger than fold noise.
- Optuna picked alpha on the same CV that is reported, which is slightly optimistic; with one parameter the effect is negligible.
- Open oddity (explanation not verified): at alpha 4,283 the smeared model's in-sample R2 (0.2) is lower than its CV R2 (0.5). Guess: the smearing factor depends on a handful of extreme residuals.

### Verdict
Does not beat past spend. Not the winner. Next: 7b XGBoost on raw GBP (non-linear, works directly in GBP, so it can learn the shape a log-linear model cannot).
