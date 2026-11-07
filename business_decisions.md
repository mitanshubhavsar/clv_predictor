# Business decisions: what the analysis supports

## Who to target (step 12)
Contact the 20% of customers with the highest predicted spend. On new data (test, cutoff 2011-06-01) they spend about 3,543 GBP each in the next 6 months, 3.7x the average of 958 GBP, and together hold 74% of all future money (random 20%: 20%). The top 10% alone hold 61%.

How much does the model add? Nothing over sorting customers by past spend: the same 20% choice by past spend gives 3,539 GBP, 74% of the money. Use the simple rule if explanation and maintenance matter most. Ranking by recency alone is clearly worse (2,858 GBP, 60%).

What this does NOT show: that a campaign works. We measured who will spend, not who will respond. Those customers spend anyway; the money a campaign adds (uplift) is unknown until a real test with a holdout group is run.

Break-even arithmetic (assumptions, not results): a contact costing 20 GBP pays for itself if it lifts the targeted customer's spend by about 0.6% (random customers: 2.1%). Targeting the big spenders makes each contact cheap to justify, but the real uplift must be measured.

## Segments (step 10)
- Protect (about 18% of customers, 72% of future money, 93% buy again): keep happy; avoid discounts, they buy anyway.
- Grow (about 19%, 12% of the money, 76% buy again): upsell / cross-sell.
- Low priority (about 62%, 16% of the money, 41% buy again): do not spend effort.
- Win back: our model-based box is tiny (25 customers). A better list: high past spend and no purchase in the last 90 days (156 customers on test, spent 1,619 GBP on average, 69% bought again).

## What drives predictions (step 11)
Past spend, then recency and the interval between purchases. The model is "about 0.37 x past spend", nudged by +/-12% for recency and rhythm. There is no hidden insight beyond: big, recent, regular customers keep spending.

## Cautions
- Giants dominate: 10 customers hold about 20% of the money, and the model under-predicts them (largest actual 183,530 vs largest prediction 95,530). Treat them as a named-account group, not as a model output.
- Probabilities shift between periods (57% of test customers bought again vs 48% in train): re-check calibration before relying on a 50% line.

---

# How each conclusion was reached (evidence and strength)

Legend: DATA = measured on held-out data. RULE-OF-THUMB = my recommendation from the segment's label; the data does not test it.

## 1. "Contact the top 20% by predicted spend"  (DATA)
- Method: models fitted on the train snapshot (features from 2009-12 to 2010-11, target = net spend 2010-12 to 2011-05). Applied to the test snapshot (features 2010-06 to 2011-05, target 2011-06 to 2011-11). For each rule we took the top 20% of customers and measured their REAL spend in the target window. Compared with the average customer (= a random 20%).
- Result: model C top 20% = 3,543 GBP each vs 958 average (+270%, 95% interval +248% to +290%, bootstrap over customers); holds 74% of all future money. Train out-of-fold agrees (+272%).
- Where: step 12 (src/step12_targeting.py, notebooks/12_targeting.ipynb), built on the test evaluation of step 9.
- Strength: strong for "these customers spend a lot". It says nothing about whether contacting them changes anything.

## 2. "The model adds nothing over past spend"  (DATA)
- Method: same top-20% comparison, using past spend (Monetary) as the ranking instead of the model.
- Result: 3,539 vs 3,543 GBP (+269% vs +270%). The model's ranking is almost the same as past spend (rank correlation 0.993, step 11); the test comparison in step 9 found the same: Capture@20% 0.740 vs 0.739.
- Strength: strong. This is consistent across steps 6, 9, 11 and 12.

## 3. "A campaign pays for itself at 0.6% uplift"  (ARITHMETIC ON ASSUMPTIONS)
- Method: cost per contact divided by the average spend of the targeted group (20 / 3,543 = 0.6%; random group 20 / 958 = 2.1%). The 20 GBP cost is an example I chose, not a known figure.
- Strength: weak. No uplift was measured; only a real test with a holdout group can say.

## 4. Segments and their actions
- The four blocks and their sizes, share of money and return rates (DATA): step 10b (train out-of-fold) and 10c (test). Protect 18% of customers / 72% of money / 93% buy again; Grow 19% / 12% / 76%; Low priority 62% / 16% / 41%. The ordering Protect > Grow > Low priority held from train to test.
- The actions ("keep happy, avoid discounts", "upsell", "do not spend effort") are RULE-OF-THUMB from the labels. The data shows these groups differ in spend and return rate; it does not show that discounts are wasted on Protect or that upselling works on Grow.
- "Better Win back list" (high past spend, no purchase in 90 days: 156 test customers, 1,619 GBP each, 69% bought again): found by looking at the TEST table after the model-based box came out tiny, so it was not chosen in advance on train. Treat it as a lead to check on another period, not as a confirmed result. The 90 days is an arbitrary choice.

## 5. "Past spend, then recency and rhythm drive the predictions"  (DATA)
- Method: SHAP on model C's multiplier (step 11), plus rank correlation with past spend.
- Result: multiplier stays within 0.31 to 0.40; the interval (33%), recency (31%) and overdue ratio (22%) account for 86% of the SHAP total.
- Limit: SHAP shows what the model uses, not what helps. Step 8 could not prove the interval and overdue ratio improve accuracy.

## 6. Cautions  (DATA)
- Giants: the 10 biggest customers hold 18% to 21% of the target money (step 5); model C's largest test prediction is 95,530 vs largest actual 183,530 and 83% of its squared error comes from those 10 customers (step 9). "Named-account group" is my recommendation.
- Probability shift: 57% of test customers bought again vs 48% in train (step 10c); Grow bought again 76% vs 61% on train.

## What we have not done that would strengthen these
- Repeat the evaluation on a third period (for example a cutoff in 2011-09 would need a shorter target window), to see whether the ordering and the 3.7x hold again.
- A real campaign test with a holdout group, to measure uplift.
- Check the Win back list on another period before using it.
