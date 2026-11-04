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