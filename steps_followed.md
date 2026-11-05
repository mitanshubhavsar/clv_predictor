STEP 1

What the raw data looks like

Size: 1,067,371 rows, 8 columns. Each row is one product line on an invoice. It is not one row per customer or per order.
Time span: 2009-12-01 to 2011-12-09, about 24 months. That matters later for the time split.
Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.
Customers: 5,942 unique customers across 53,628 invoices.



Problems that need cleaning in step 2
Issue	Count	Why it matters
No Customer ID	243,007 rows (23%)	We can't track a customer we can't identify, so these likely get dropped.
Cancelled invoices (start with "C")	19,494 rows	These are returns, not purchases.
Quantity ≤ 0	22,950 rows	Mostly returns, plus possibly damaged goods or adjustments.
Price ≤ 0	6,207 rows	Free items or bad data.
Extreme values	Quantity ±80,995, Price up to 38,970	One huge order, or a data error, could distort a customer's total.


The Price range also includes −53,594, which is almost certainly an accounting adjustment rather than a sale.

STEP 2
src/step2_clean.py writes data/purchases.csv and data/returns.csv. The row accounting adds up exactly, because the script asserts it.

Where the 1,067,371 rows went

Rows
No Customer ID (dropped)	243,007
Cancelled "C" invoices (moved to returns)	18,744
Quantity ≤ 0 (dropped)	0
Price ≤ 0 (dropped)	71
Purchases kept	805,549


Quantity ≤ 0 dropped nothing. Every negative quantity was already on a "C" invoice, so rule 2 caught them. The 22,950 negatives from step 1 were really cancellations.
Customers: 5,878 have purchases, and 2,572 of them also have returns. That's enough for a return-rate feature.
Orphan returns: 129 return rows belong to customers with no purchases in this data. Those returns will be ignored when we build features.

2 Major decisions:

1. Non-product "stock codes" are still in purchases. These are fees, not products, so they inflate a customer's spend:

Code	Rows	Total £	Meaning
M	709	152,341	Manual entry
POST	1,838	126,563	Postage
C2	253	12,796	Carriage (shipping)
DOT	16	11,906	Dotcom postage
ADJUST, BANK CHARGES, TEST001, etc.	small	small	Adjustments and test rows
Is postage revenue? For "how much will they spend", arguably yes. But M, ADJUST and TEST rows are not customer behaviour.

2. Extreme orders. The top lines are:

£168k for 80,995 paper-craft items by customer 16446.
£77k for 74,215 jars by customer 12346.
£39k for a single line (60 picnic baskets at £649.50).
These single orders could dominate a customer's 6-month spend, and so dominate the model. The 74,215-jar order looks like a likely order-then-cancel pair, but I haven't checked that. I'd wait to handle outliers until we see the target distribution in step 5. They're easier to judge there.


Returns in spend: net them off, or keep the old project's gross approach?
Non-product codes: drop the internal ones, or leave everything in as before?

1. Returns: net them off 

because:  If a customer orders £168k and cancels all of it, they were worth £0. Gross spend would call them the most valuable customer in the dataset. I checked the top 5 orders, and 4 of them were fully cancelled. handful of huge spenders swing the results. Many of those "huge spenders" are cancelled orders. Netting removes the problem at its source, with no arbitrary clipping.


2. Non-product codes: drop the internal ones
Why drop M, ADJUST, ADJUST2, BANK CHARGES, TEST001, TEST002, D, SP1002 and PADS:

They aren't customer behaviour. A manual adjustment or a test row says nothing about how much a customer likes to buy. M rows total £152k and include a single £10,954 line. That's the kind of thing that distorts one customer's total.
They're few. It's under 1,000 rows out of 805k, so dropping them costs almost nothing.

Why keep POST, C2 and DOT (postage and carriage):

Customers really pay it. It's real money from the customer, and it scales with how much they order.
It's a judgment call. If you want product revenue only, drop these too. That's a cleaner definition, but you lose revenue that actually came in.

What changed
Internal codes dropped. M, ADJUST, ADJUST2, BANK CHARGES, TEST001, TEST002, D, SP1002 and PADS are removed, 1,428 rows in total. Postage and carriage (POST, C2, DOT) are kept, and the notebook shows both lists.
New ledger.csv. It stacks purchases and returns, with returns as negative rows dated at the cancellation. Summing it over any date window gives money kept in that window. Features that count things (Frequency, AOV) will use purchases. Money amounts (Monetary and the target) will use ledger.
Row accounting still adds up. Purchases are now 804,739 rows from 5,854 customers. Returns are 18,136 rows from 2,465 customers.

What netting did
Overall effect: gross £17.59M, returns −£0.74M, net £16.84M. That's only about 4% less spend overall.
Per customer, it can be huge. Customer 16446 goes from £168,472 gross to £3 net, and gross spend would have made them the 8th biggest customer. By net spend they drop out of the top 8.
28 customers have negative net spend overall.


Deciding on Temporal Splitting

My data runs from 2009-12-01 to 2011-12-09, about 24 months. Every choice below is about how to spend those 24 months, because each customer needs three things:

A feature window: the history we describe them with.
A target window: the future we predict. It must come strictly after the feature window.
A test period: a later period the model has never seen, to judge it honestly.


The four options

| Option | How it works | Good | Bad |
|---|---|---|---|
| **A. 12-month features, 6-month target, two snapshots** (old project's choice) | Train: features Dec 2009–Nov 2010, target Dec 2010–May 2011. Test: features Jun 2010–May 2011, target Jun–Dec 2011. The two targets don't overlap. | Features cover a full year, so they see one whole seasonal cycle. Test is truly in the future. Uses all 24 months. | Only one training snapshot (a few thousand customers). Test target (Jun–Dec) includes the Christmas run-up, but the training target (Dec–May) doesn't line up the same way, so the two periods aren't alike. |
| **B. Shorter features (e.g. 6 months), 6-month target, many snapshots** | Slide the window forward several times and pool the snapshots. | Much more training data, since the same customer appears several times. | Six months of history misses the yearly cycle of a Christmas-heavy retailer. The same customers repeat across snapshots, so data is less independent than it looks, which makes validation trickier. |
| **C. 12-month features, 12-month target** | A longer horizon, closer to a true "lifetime" value. | A more meaningful CLV. | Uses all 24 months, so nothing is left for an out-of-time test. The only honest check would be a random split. |
| **D. Random customer split on one snapshot** | Features and target from the same two periods; shuffle customers into train and test. | Simplest, and gives a large test set. | Only answers "does the model work for customers like the ones it trained on, in the same period?" It doesn't test next season. The old project started here and dropped it. |

Two rules whichever option is picked

1. Who counts as a customer: only people who bought during the feature window. Customers who first appear in the target window can't be predicted from history, so they're excluded.
2. Zero is a real answer: a customer who bought in the feature window and then bought nothing in the target window has a target of 0. They must stay in. If we dropped them, the model would learn that everyone keeps buying.

Example: the same idea with 5 years of transactions (hypothetical data, 2019-01-01 to 2023-12-31, 60 months)

With 24 months we could only afford 2 snapshots (option A). With 60 months the same rule (12-month features, 6-month target) gives room for many snapshots, plus a separate validation set. Cutoffs are placed every 6 months so the target windows never overlap.

| Role | Cutoff | Feature window (12 months) | Target window (6 months) |
|---|---|---|---|
| Train 1 | 2020-01-01 | 2019-01-01 .. 2019-12-31 | 2020-01-01 .. 2020-06-30 |
| Train 2 | 2020-07-01 | 2019-07-01 .. 2020-06-30 | 2020-07-01 .. 2020-12-31 |
| Train 3 | 2021-01-01 | 2020-01-01 .. 2020-12-31 | 2021-01-01 .. 2021-06-30 |
| Train 4 | 2021-07-01 | 2020-07-01 .. 2021-06-30 | 2021-07-01 .. 2021-12-31 |
| Train 5 | 2022-01-01 | 2021-01-01 .. 2021-12-31 | 2022-01-01 .. 2022-06-30 |
| Train 6 | 2022-07-01 | 2021-07-01 .. 2022-06-30 | 2022-07-01 .. 2022-12-31 |
| Validation (tune models, pick the winner) | 2023-01-01 | 2022-01-01 .. 2022-12-31 | 2023-01-01 .. 2023-06-30 |
| Test (touched once, at the very end) | 2023-07-01 | 2022-07-01 .. 2023-06-30 | 2023-07-01 .. 2023-12-31 |

What this example shows

1. The first possible cutoff is 12 months after the data starts (2020-01-01), because every snapshot needs a full year of history. The first year is only ever used as features.
2. Feature windows overlap between snapshots (Train 1 and Train 2 share 2019-07 .. 2019-12). That is fine: features only look backwards. Target windows must NOT overlap, and here they don't.
3. Order is always train < validation < test in time. The model never learns from anything dated after the period it is judged on.
4. When we predict the test snapshot (cutoff 2023-07-01), every label used for training or tuning (the validation target ends 2023-06-30) was already known on that date. This is the rule that keeps the evaluation honest.
5. The training targets cover three Christmas seasons (2020, 2021, 2022), so the model sees more than one seasonal cycle. That fixes the biggest weakness of the 2-year version (train and test periods behaving differently). 2019 is used only as features, never as a target.
6. Catch: the same customer now appears in several snapshots. If we cross-validate on pooled snapshots, rows of one customer must stay together in the same fold (group by Customer ID), or the validation scores look better than they are.

Compare with our real data (24 months)

| | Real data (24 months) | Example (60 months) |
|---|---|---|
| Training snapshots | 1 (cutoff 2010-12-01) | 6 |
| Separate validation snapshot | No (CV on train only) | Yes (cutoff 2023-01-01) |
| Test snapshot | 1 (cutoff 2011-06-01) | 1 (cutoff 2023-07-01) |
| Christmas seasons seen in training targets | 1 | 3 |
| Unused data at the end | 8 days | none |

Train vs validation vs test

| Dataset | Role |
|---|---|
| Train | The model learns its parameters from it (weights, trees). |
| Validation | Used to make choices about the model: which model, which settings, which features. The model does not learn from it directly. |
| Test | The final exam. Looked at once, after all choices are made. |

Why validation is needed: training data always flatters the model, because it has already seen it. To compare models and settings fairly, score them on data they have not seen (validation) and pick the winner from that.

Why test is also needed: every time we pick something because it scored well on validation, a little of that data leaks into our choices (try 200 settings, keep the best, and the best score is partly luck). Test was never used for any choice, so its score is the honest estimate. If we tweak the model after seeing the test score, test becomes a second validation set and the honest estimate is gone.

Analogy: train = textbook and practice problems, validation = mock exam (we adjust how we study), test = the real exam.

In our real project (24 months): no separate validation snapshot. Validation happens inside train using 5-fold cross-validation (train on 4 folds, score on the 5th, rotate, average). Test is the later-in-time snapshot (cutoff 2011-06-01). The 5-year example has room for a real validation snapshot.

Caution: cross-validation on train uses random customer folds, so it only tests "customers like the training ones, same period", not "next season". A gap between CV score and test score is mostly the time shift (train 50.7% zero-spenders vs test 42.5%), not necessarily a bug.

# Step 4: RFM features

**Goal:** turn each snapshot's feature rows into one row per customer with three numbers.

| Feature | Meaning | Definition used |
|---|---|---|
| **R**ecency | how recently they bought | whole days from their last *purchase* to the **cutoff** (small = recent) |
| **F**requency | how often they buy | number of distinct purchase invoices in the feature window |
| **M**onetary | how much they spend | **net** money kept in the feature window (purchases minus returns) |

Notes:
- Recency is measured from the cutoff, not from today, so train and test are comparable.
- Returns do **not** count as purchases for Recency/Frequency, but they **do** reduce Monetary (same netting decision as the target).

The logic is in `src/step4_rfm.py`.

Target Analysis

What the target looks like
Zeros are the biggest feature.

51% of train customers and 42% of test customers buy nothing in the next 6 months.
Whether they buy again is very predictable from RFM. A one-time buyer returns about 24% of the time in train, and a customer with 11+ orders about 92% of the time.
Recency tells the same story: 70% of the most recent quarter buy again, versus 25% of the least recent.
"Will they buy?" and "how much will they spend?" are therefore two different questions. A two-stage model that predicts both is a natural fit, and it's what the old project's "Hurdle" model does.
Negatives are tiny. There are 51 in train and 28 in test. Almost all are refunds of goods bought earlier, with little or no new buying. Flooring them at 0 moves the mean target by about 0.2%.

The money is concentrated.

The raw target is very skewed (skewness 19–23), and log1p makes it roughly symmetric.
The top 10% of buying customers hold 56–58% of the target money, and the top 20% hold about 70%.
Removing only the 10 biggest customers cuts the mean target from £712 to £588 in train, and from £958 to £757 in test. A few people can swing a money-scale score.