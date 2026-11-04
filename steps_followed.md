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

