# Baselines and evaluation metrics: questions and answers

All numbers are in GBP, on the original scale. Code: `src/metrics.py`, `src/step6_baselines.py`. Notebook: `notebooks/6_baselines.ipynb`.

## 1. The result table

| | Train CV: mean | Train CV: past spend | Test: mean | Test: past spend |
|---|---|---|---|---|
| R2 | 0.00 | 0.75 | 0.00 | 0.71 |
| MAE (GBP) | 944 | 523 | 1,086 | 613 |
| Capture@20% | 0.17 | 0.74 | 0.16 | 0.74 |

Ceiling for Capture@20% (perfect ranking): 0.86 on train, 0.83 on test.
Without the 10 biggest customers, past spend: R2 0.55 (CV) / 0.63 (test), Capture@20% 0.69 / 0.67.

## 2. Doubt: what do "Train CV: mean" and "Train CV: past spend" mean?

Two baselines (simple rules, no machine learning):

| Baseline | Rule |
|---|---|
| mean | Guess the same number for everyone: the average future spend in the training data (GBP 714). |
| past spend | Guess future spend = 0.37 x the customer's Monetary (spend in the last 12 months). Example: GBP 1,000 past spend predicts GBP 370. The 0.37 is learned from training data and only sets the GBP scale. The ranking is simply "sort by past spend". |

Two ways of scoring them:

| Scoring | How |
|---|---|
| Train CV (cross-validation) | Split the 4,242 train customers into 5 groups. For each group: learn the rule from the other 4 groups, predict the held-out group. Pool all 4,242 predictions and compare with what really happened. Every customer is predicted by a rule that never saw them. |
| Test | Learn the rule from all of train, predict the 4,326 test customers (a later period). The more realistic check. |

So "Train CV: past spend" = the past-spend rule, scored on train customers it had not learned from.

## 3. Doubt: how are the metrics computed? (5-customer example)

| Customer | Actual future spend | Mean guess | Past-spend guess |
|---|---|---|---|
| A | 0 | 400 | 50 |
| B | 0 | 400 | 300 |
| C | 200 | 400 | 150 |
| D | 500 | 400 | 450 |
| E | 1,300 | 400 | 1,100 |

**MAE** = average size of the miss, ignoring the sign. Lower is better.
- Mean guess: misses 400, 400, 200, 100, 900 -> average 400.
- Past-spend guess: misses 50, 300, 50, 50, 200 -> average 130.
- In the real table: 523 vs 944 means past spend is off by about GBP 520 for a typical customer, against about GBP 940 for the average guess.

**R2** = how much better than always guessing the average. R2 = 1 - (your squared misses) / (squared misses of the average guess).
- The mean guess scores 0 by definition.
- Past spend in the example: 1 - 137,500 / 1,180,000 = about 0.88.
- 1 = perfect, 0 = no better than the average, can go negative.
- Squaring makes big misses count much more, so a few huge customers can swing R2.

**Capture@20%** = did we pick the right people?
1. Rank customers by the guess, highest first.
2. Take the top 20% (1 person in the example).
3. Add up what they really spent, divide by everyone's total real spend.
- Past spend ranks E first: 1,300 / 2,000 = 0.65.
- The mean guess gives everyone the same number, so there is no ranking and ties are broken at random. A random pick captures about 20% on average. The 0.17 and 0.16 in the table are just luck on one random draw.
- Real table: 0.74 means the top 20% by past spend hold 74% of all future money.

## 4. Doubt: where is "the best possible is 0.86 train / 0.83 test"?

It is a reference line, not a table column.
- Computed in `src/metrics.py`, function `oracle_capture` (line 45).
- Printed by `src/step6_baselines.py` (lines 83 and 92) as `ceiling (perfect ranking) Capture@20%`, and in the output cells of the notebook, just under each results table.

How it is computed: the same Capture@20% calculation, but customers are ranked by what they REALLY spent instead of by a prediction. Take the 20% who actually spent the most, add up their money, divide by the total. It is not 1.0 because the biggest 20% do not hold all the money (the rest is spread over everyone else). Train and test differ because they are different periods with different customers.

## 5. Doubt: "the model captured 0.74 and in real it is 0.83 / 0.86?"

Two corrections:
1. 0.74 is NOT from a model. It is the past-spend baseline ("sort by what they spent before"). No machine-learning model has been trained yet. Step 7 is where models start, and they have to beat 0.74.
2. 0.83 / 0.86 is not the "real" value. It is the ceiling: what a perfect crystal ball would give. Nobody can know it in advance, so no model can reach it. "Best possible" is the right word.

| Ranking | Train | Test | What it means |
|---|---|---|---|
| Random | ~0.20 | ~0.20 | Picking customers by chance |
| Past-spend baseline | 0.74 | 0.74 | A simple rule, no ML |
| Ceiling | 0.86 | 0.83 | Perfect foresight; impossible in practice |

The past-spend rule gets about 82-85% of the way from random to perfect. A model can improve on it by at most about 0.09-0.12 of Capture@20%.
The 0.74 is the same on train and test because it is a property of the rule; the ceiling differs because the periods differ.

## 6. Why only three metrics

| Metric | Question |
|---|---|
| R2 | How well do the predicted amounts fit? |
| MAE | What is the typical error in GBP? (not dominated by a few huge customers) |
| Capture@20% | If we contact the top 20%, what share of future money do we reach? |

Dropped: RMSE (says the same as R2), Spearman, confidence-interval columns.
Kept as helpers, not metrics: the ceiling (section 4), the "without the 10 biggest customers" check, and a bootstrap function for close calls (Capture has noise of about +-0.04, so a model must beat 0.74 by more than that to count).
Known blind spot of the three: they do not show whether TOTAL predicted spend matches TOTAL actual spend (calibration). The old project's model predicted only 67% of actual total spend. Check separately if totals are ever used for budgeting.

## 7. What the numbers mean

- The mean baseline is the floor: R2 is 0, the typical miss is large, picking people is no better than chance.
- Past spend is already strong: it explains about three quarters of the variation, is off by about GBP 520-610 per customer, and finds 74% of the money in the top 20%.
- Amounts are harder on test (MAE 613 vs 523) because test customers spend more and more unevenly. Ranking quality is the same (0.74).
- R2 leans on a few people: without the 10 biggest customers it falls from 0.75 to 0.55 (CV) and 0.71 to 0.63 (test). Capture and MAE hold up much better.
- The same pattern appeared in the old project (past-spend capture 74.2%), which is a useful cross-check that this rebuild behaves.

## 8. Still open

Which metric should pick the winner in step 7?
- Capture@20% if the aim is who to contact.
- R2 with MAE as tiebreaker if the aim is forecasting amounts.
They can disagree. The old project picked by RMSE and its winner then lost to the spreadsheet sort on ranking, so this should be decided before step 7.

## 9. Doubt: how does the Capture@20% code work? (src/metrics.py, function `revenue_capture`)

```python
y, score = np.asarray(y, float), np.asarray(score, float)
n_top = int(round(frac * len(y)))
tiebreak = np.random.default_rng(seed).random(len(y))
order = np.lexsort((tiebreak, -score))  # primary key: -score, secondary: random
return float(y[order[:n_top]].sum() / y.sum())
```

Example (same 5 customers, real output):

| Customer | A | B | C | D | E |
|---|---|---|---|---|---|
| y (actual future spend) | 0 | 0 | 200 | 500 | 1,300 |
| score (past-spend guess) | 50 | 300 | 150 | 450 | 1,100 |

| Line | What it does | Example |
|---|---|---|
| 1 | Make sure y and score are plain numeric arrays. | no visible change |
| 2 | How many customers is "top 20%"? | 0.2 x 5 = 1. With 4,326 test customers it is 865. |
| 3 | Draw a random number for every customer. It only decides who goes first when two scores are equal. The seed makes it repeatable. | [0.637, 0.27, 0.041, 0.017, 0.813] |
| 4 | Sort customers, best score first. | order = [4, 3, 1, 2, 0] = E, D, B, C, A |
| 5 | Add up the real money of the top group, divide by the total. | E: 1,300 / 2,000 = 0.65 |

Two things that are easy to misread in line 4:
- The minus sign: np.lexsort sorts smallest to largest, so -score turns "largest score first" into the order we want.
- The LAST key in lexsort is the main one. So -score is the main key, and the random tiebreak is used only when two scores are equal.

Why the random tiebreak exists: the mean baseline gives everyone the same guess, so every score is tied. Without a tiebreak the sort keeps the original row order, which is arbitrary and could by accident put big customers first or last. With 5 customers and five different seeds the capture swings between 0.00 and 0.25. Averaged over every possible pick it is exactly 0.20 (0 + 0 + 200 + 500 + 1,300) / 5 / 2,000. With 4,000+ customers the swing is small but one draw is still not exactly 0.20. That is where the mean baseline's 0.17 (train) and 0.16 (test) come from: chance, not a property of the baseline.

## 10. Doubt: how is the ceiling (0.86 train, 0.83 test) calculated?

Same Capture@20% calculation, but customers are ranked by what they REALLY spent instead of by a prediction (`oracle_capture(y)` calls `revenue_capture(y, y)`). No ranking can do better.

Recipe: take every customer's real future spend, sort biggest to smallest, keep the top 20%, add up their money, divide by everyone's total money.

5-customer example: the top 20% is 1 customer, E. E holds 1,300 of 2,000, so the ceiling is 0.65.

Real data:

| | Train | Test |
|---|---|---|
| Customers | 4,242 | 4,326 |
| Top 20% | 848 customers | 865 customers |
| Their real money | 2,605,761 | 3,449,340 |
| Everyone's money | 3,028,459 | 4,144,356 |
| Ratio = ceiling | 0.860 | 0.832 |

Why not 1.0: money is spread beyond the top 20%. The other 80% of customers still hold 14% of all money on train (422,698) and 17% on test (695,016). About half the customers spent nothing (52% on train, 43% on test), so the leftover money sits with the remaining customers who spent a little.

Why train and test differ: on test, spend is spread across more customers. 57% of test customers have any spend vs 48% on train, and the spend needed to be in the top 20% is 1,027 on test vs 765 on train. A flatter spread means the top 20% hold a smaller share, so the ceiling is lower.
