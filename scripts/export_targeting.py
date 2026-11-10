"""Step 17h: the targeting-rules table for the 'Simulate targeting' stage of the site.

For each way of ranking test customers, take the top 20% and measure the money they REALLY spent
in the next 6 months. Computed from outputs/test_predictions.csv; writes site/public/data/targeting_rules.json.

Run:  python scripts/export_targeting.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import revenue_capture  # noqa: E402

tp = pd.read_csv(ROOT / "outputs/test_predictions.csv", index_col=0)
y = tp["actual_spend"].to_numpy(float)
n_top = int(round(0.2 * len(y)))
avg = float(y.mean())

rules = [
    ("Model C", "predicted 6-month spend", tp["predicted_spend"].to_numpy(float)),
    ("Past spend", "past net spend, no model", tp["Monetary"].to_numpy(float)),
    ("Model C x probability of returning", "predicted spend multiplied by the chance of buying again",
     (tp["predicted_spend"] * tp["probability_buy_again"]).to_numpy(float)),
    ("Recency only", "most recent buyers first", -tp["Recency"].to_numpy(float)),
    ("Perfect ranking (ceiling)", "the real future spend, which nobody knows in advance", y),
]
rows = []
for name, how, score in rules:
    tie = np.random.default_rng(0).random(len(y))  # same tie-breaking as metrics.revenue_capture
    top = np.lexsort((tie, -score))[:n_top]
    rows.append({"rule": name, "ranks_by": how, "mean_future_spend": float(y[top].mean()),
                 "lift_vs_average": float(y[top].mean() / avg - 1), "share_of_money": float(y[top].sum() / y.sum())})
    assert abs(rows[-1]["share_of_money"] - revenue_capture(y, score)) < 1e-9
rows.append({"rule": "Random 20% (floor)", "ranks_by": "no ranking: the average customer", "mean_future_spend": avg,
             "lift_vs_average": 0.0, "share_of_money": 0.2})
out = {"n_test": int(len(y)), "n_top": n_top, "average_customer": avg, "rows": rows}
(ROOT / "site/public/data/targeting_rules.json").write_text(json.dumps(out, indent=1))
print(pd.DataFrame(rows).round(3).to_string(index=False))
