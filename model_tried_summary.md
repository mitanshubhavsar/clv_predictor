# Models tried: summary

How every candidate is judged (step 7 decisions):
- The winner is chosen by R2 in GBP (MAE breaks ties), from 5-fold cross-validation on the train snapshot only.
- Hyperparameters are tuned with Optuna (TPE sampler, seed 42, 250 trials, the same CV folds for every trial and every model).
- Features: Recency, Frequency, Monetary only. Extra features come later, one at a time.
- The test snapshot is not touched until a winner is chosen.
- The bar to beat is the past-spend baseline: R2 0.75, MAE 523, Capture@20% 0.74 (train CV).

Candidates: 7a Ridge on log1p (done), 7b XGBoost on raw GBP (done), 7c Tweedie (done), 7d Hurdle.

---

## 7a. Ridge regression on log1p(target)

Code: `src/step7a_ridge_log.py`, `src/tuning.py`. Notebook: `notebooks/7a_ridge_log.ipynb`.

### What the model is
log1p of R, F, M -> StandardScaler -> Ridge -> predicts log1p(future spend) [log(1 + x) plain log can't handle zero: log(0) is undefined] -> convert back to GBP -> score in GBP(Great British Pound).

### What was tuned: one hyperparameter, alpha
Ridge has one knob, alpha: how strongly the model is held back from fitting the training data too closely. Small alpha = fits freely. Large alpha = simpler, more cautious predictions.

Optuna tried 250 alphas between 0.001 and 1,000,000 (log scale). For each one it ran the 5-fold CV and scored R2 in GBP. The alpha with the highest R2 won.

Fixed, not tuned: the features (R, F, M), taking the log of the features, standardizing them, and predicting log1p(future spend).

### The two variants
The model predicts in log units, so the prediction has to be converted back to GBP. The variants differ only in how.

| Variant | How it converts back | Idea |
|---|---|---|
| Plain | Just undo the log (expm1) [e^x − 1] | Simple, but the result is a typical value, not the average, so it comes out too low (Jensen's inequality) |
| Smearing (Duan) | Undo the log, then multiply by a correction factor = mean of exp(training residuals) | Tries to fix the "too low" problem |

Each variant got its own Optuna search, so each has its own best alpha.

### Results (train, 5-fold CV)

| | Best alpha | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|---|
| Mean baseline | none | 0.00 | 944 | 0.17 | 1.00 |
| Past-spend baseline | none | 0.75 | 523 | 0.74 | 1.00 |
| Ridge, plain expm1 | 1,050 | 0.40 | 586 | 0.72 | 0.28 |
| Ridge, smearing | 4,829 | 0.55 | 575 | 0.72 | 0.99 |

The last column is a diagnostic, not one of the three metrics.
Without the 10 biggest customers: past spend R2 0.55 / Capture 0.69; plain 0.34 / 0.67; smearing 0.50 / 0.67.

### In plain words
- Both variants lose to the simple past-spend rule on amounts: R2 0.40 and 0.55 vs 0.75, and MAE 586 and 575 vs 523.
- On picking the right people it is a tie: 0.72 vs 0.74 is inside the noise of about +-0.04.
- Smearing beats plain on R2 (0.55 vs 0.40). Plain predicts only about 28% of the real total spend; smearing brings the total to about 99%. But smearing is touchy: it only works with a large alpha, and with a small alpha it gives wildly wrong numbers.
- Large alphas won (1,050 and 4,829): the model works best when held back hard. Undoing a log makes the biggest customers' predictions explode if the model is allowed to fit freely.
- Why it loses on amounts: it is trained to be accurate in log units, but R2 in GBP is decided by the biggest customers. Past spend is linear in GBP, so it gets them right almost for free.

### Why smearing is fragile here
The smearing factor is the mean of exp(residual). About half the customers have target 0 (log1p(0) = 0), and a few huge customers are predicted far too low, so log-scale residuals reach about +8 and exp(8) is about 3,000. A handful of such values dominate the factor (S about 13-17). At alpha = 1 the plain model's in-sample R2 is already -1.9, and the smeared one is -1284.

### Checks done
- First run used an alpha range up to 1,000 and the best alpha was 989, on the edge. Widened to 1,000,000; the best alphas are now inside the range.
- Re-ran the CV with 10 different random fold assignments: R2 0.744 +- 0.006 (past spend), 0.426 +- 0.014 (plain), 0.558 +- 0.011 (smearing). Capture spread about 0.001. The gap to past spend is far bigger than fold noise.
- Optuna picked alpha on the same CV that is reported, which is slightly optimistic; with one parameter the effect is small.
- Trials were raised from 30 to 250: the best alphas barely moved (1,116 -> 1,050 and 4,283 -> 4,829) and the smearing variant's R2 rose from 0.51 to 0.55.
- Open oddity (explanation not verified): at alpha 4,829 the smeared model's in-sample R2 (0.4) is lower than its CV R2 (0.55). Guess: the smearing factor depends on a handful of extreme residuals.

### Verdict
Does not beat past spend. Not the winner. Next: 7b XGBoost on raw GBP (non-linear, works directly in GBP, so it can learn the shape a log-linear model cannot).

---

## 7b. XGBoost on raw GBP

Code: `src/step7b_xgboost.py`. Notebook: `notebooks/7b_xgboost.ipynb`.

### What the model is
Gradient-boosted trees: many small decision trees added one after another, each correcting the previous ones' mistakes. Predicts in GBP directly (squared-error objective), so there is no log and nothing to convert back. Trees do not care about feature scale, so R, F, M go in as they are.

### What was tuned: five hyperparameters (Optuna, 250 trials)

| Setting | Meaning | Range | Best (250 trials) | Best (first run, 30 trials) |
|---|---|---|---|---|
| n_estimators | number of trees | 50 to 500 | 480 (at the top edge) | 88 |
| max_depth | how deep each tree can go | 2 to 6 | 5 | 3 |
| learning_rate | size of each tree's correction | 0.01 to 0.3 (log) | 0.011 (at the bottom edge) | 0.042 |
| subsample | share of customers each tree sees | 0.3 to 1.0 (first run: 0.5 to 1.0) | 0.30 (at the bottom edge) | 0.51 (was at the edge) |
| colsample_bytree | share of features each tree sees | 0.5 to 1.0 | 0.57 | 0.63 |

With 250 trials Optuna picked many small, slow, deliberately noisy trees. Three settings sit at the edge of their ranges, which suggests a flat plateau where "more regularization" is slightly better. Widening further is unlikely to change the story.

### Results (train, 5-fold CV)

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual |
|---|---|---|---|---|
| mean | 0.00 | 944 | 0.17 | 1.00 |
| past_spend | 0.75 | 523 | 0.74 | 1.00 |
| Ridge on log1p + smearing (7a) | 0.55 | 575 | 0.72 | 0.99 |
| XGBoost raw GBP (7b), tuning score | 0.50 | 600 | 0.71 | 1.00 |

Without the 10 biggest customers: past spend R2 0.55 / Capture 0.69; 7a smearing 0.50 / 0.67; XGBoost 0.32 / 0.65.

### The tuning score is optimistic (important)
The same XGBoost settings re-scored on 10 other fold splits: R2 0.478 +- 0.022 (range 0.436 to 0.517), vs 0.503 on the split Optuna saw.
- 30 trials: tuning score 0.453, other splits 0.475.
- 250 trials: tuning score 0.503, other splits 0.478.
More tuning raised the tuning score by 0.05 but the honest estimate by only 0.003. The extra trials mostly bought a better-looking number, not a better model. The more combinations tried, the more likely one looks good on these particular folds by luck.

### In plain words
- XGBoost does not beat past spend, and is below the 7a Ridge (0.50 vs 0.55 on the tuning score; 0.48 vs 0.56 on the honest estimate). MAE is worse than 7a (600 vs 575).
- Totals are well calibrated (1.00), unlike the plain log model.
- Ranking is a tie again: 0.71 vs 0.74 is inside the +-0.04 noise.

### Why a more flexible model loses on R2 in GBP
Trees predict averages of the training customers in a group, so they cannot predict higher than the biggest values they have seen. Past spend (0.37 x Monetary) keeps scaling for any size. Checked on the data (out-of-fold, 250-trial settings):

| Customer | Past spend (Monetary) | Actual | Past-spend rule | XGBoost |
|---|---|---|---|---|
| 18102 | 318,751 | 92,477 | 120,142 | 32,078 |
| 14646 | 245,028 | 110,668 | 89,681 | 30,266 |

- The largest prediction XGBoost ever makes is about 54,000; real spend reaches 110,668.
- About 66% of XGBoost's total squared error comes from just the 10 biggest customers (55% for past spend).
- MAE is also worse (600 vs 523), so it is not only the giants, but they decide R2.

### Other checks
- Feature importance: Frequency 0.54, Monetary 0.41, Recency 0.05. Recency barely matters to the trees although it correlates with the target at -0.38 (Frequency and Monetary already carry most of it).

### Verdict
Does not beat past spend. Not the winner. Ideas to fix the giants problem (not tried, for discussion): (a) predict the ratio target / Monetary so the model corrects the linear rule instead of replacing it; (b) add features that tell big customers apart (later steps).

Standing by R2 (train CV): past_spend 0.75 > Ridge+smearing 0.55 > XGBoost 0.50 (honest estimate 0.48) > Ridge plain 0.40 > mean 0.00.
Next: 7c Tweedie.

---

## 7c. XGBoost with the Tweedie objective

Code: `src/step7c_tweedie.py`. Notebook: `notebooks/7c_tweedie.ipynb`.

### What is different from 7b
Same boosted trees, different loss function (what the model tries to minimise). 7b minimised squared error in GBP. Tweedie assumes the target looks like "many exact zeros plus a long right tail", which is what our 6-month spend looks like.
- Predictions are never negative (the trees work on a log scale internally and the model predicts exp(score)).
- Errors are weighted relative to the size of the customer instead of every GBP counting the same, so the giants dominate the loss less.

### What was tuned: six hyperparameters (Optuna, 250 trials)

| Setting | Meaning | Range | Best |
|---|---|---|---|
| n_estimators | number of trees | 50 to 500 | 395 |
| max_depth | how deep each tree can go | 2 to 6 | 2 (at the bottom edge) |
| learning_rate | size of each tree's correction | 0.01 to 0.3 (log) | 0.014 |
| subsample | share of customers each tree sees | 0.3 to 1.0 | 0.996 (at the top edge) |
| colsample_bytree | share of features each tree sees | 0.5 to 1.0 | 0.76 |
| tweedie_variance_power | how zeros and spread are treated: near 1 = count-like, zero-friendly; near 2 = heavy-tailed positive amounts | 1.01 to 1.99 | 1.57 |

The best model is simple: 395 very shallow trees. max_depth 2 is the minimum allowed, so even depth 1 might do as well; the model does not want complexity.

### Results (train, 5-fold CV)

| | R2 | MAE (GBP) | Capture@20% | total predicted / total actual | R2 without top 10 |
|---|---|---|---|---|---|
| mean | 0.00 | 944 | 0.17 | 1.00 | -0.01 |
| past_spend | 0.75 | 523 | 0.74 | 1.00 | 0.55 |
| Ridge on log1p + smearing (7a) | 0.55 | 575 | 0.72 | 0.99 | 0.50 |
| XGBoost raw GBP (7b) | 0.50 | 600 | 0.71 | 1.00 | 0.32 |
| XGBoost Tweedie (7c) | 0.51 | 534 | 0.74 | 0.92 | 0.54 |

Honest estimate (same settings on 10 other fold splits): R2 0.487 +- 0.013 (range 0.471 to 0.520), vs 0.508 on the split Optuna saw. A smaller gap than 7b had (0.503 vs 0.478), probably because Tweedie settled on a simple model.

### Why the three metrics disagree
- Tweedie is the best model on MAE (534; 7b 600; 7a 575) and ties past spend on ranking (0.744 vs 0.739), but its R2 (0.51) is not the highest.
- MAE is the typical miss; R2 is dominated by the giants. Tweedie is still a tree model and cannot predict the giants: its largest prediction is about 39,000 while real spend reaches 110,668; 76% of its squared error comes from the 10 biggest customers (past spend 55%, 7b 66%, 7a 72%).
- For the other ~4,200 customers it is the best model of all. Without the 10 biggest customers its R2 (0.54) is on par with past spend (0.55) and far above 7b (0.32).
- It under-predicts the total by 8% (0.92); 7b is on target and 7a smearing within 1%.
- Feature importance: Monetary 0.52, Frequency 0.42, Recency 0.06.

### In plain words
The Tweedie loss fixed a lot of 7b's problems (MAE 600 -> 534, R2 without the giants 0.32 -> 0.54) just by describing the target's shape better. It is the first candidate that is competitive with the spreadsheet rule for the typical customer. Its weakness is the one all tree models share here: the few giants.

### Verdict
By our rule (R2 picks the winner) past spend still leads, and among the models Ridge + smearing (0.55) is ahead of Tweedie (0.51). Tweedie wins on MAE and ranking.

Standing by R2 (train CV): past_spend 0.75 > Ridge+smearing 0.55 > Tweedie 0.51 (honest 0.49) > XGBoost 0.50 (honest 0.48) > Ridge plain 0.40 > mean 0.00.
Next: 7d Hurdle.
