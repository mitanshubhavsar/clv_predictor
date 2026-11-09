"""Step 17f: the 'no machine learning' section of the site.

Applies the two textbook CLV formulas (ARPU x margin / churn, and AOV x frequency x lifespan)
to our own data, scores them on the same test snapshot as model C, and writes
site/public/data/heuristics.json. Nothing here is fitted except the one number
that the text says is fitted (on train).

Run:  python scripts/export_heuristics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import evaluate  # noqa: E402
from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot  # noqa: E402
from step4_rfm import build_rfm  # noqa: E402

ledger = load_ledger()
tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
Xtr, Xte = build_rfm(tr), build_rfm(te)
ytr, yte = tr.target.to_numpy(float), te.target.to_numpy(float)
tp = pd.read_csv(ROOT / "outputs/test_predictions.csv", index_col=0)
assert len(tp) == len(yte)
pred_c = tp["predicted_spend"].to_numpy()


def period_stats(y):
    bought = y > 0
    return {"customers": int(len(y)), "arpu_active": float(y[bought].mean()), "churn": float(1 - bought.mean()),
            "mean_all": float(y.mean())}


s_tr, s_te = period_stats(ytr), period_stats(yte)
f1_tr = s_tr["arpu_active"] / s_tr["churn"]  # margin = 1: we only have revenue
f1_te = s_te["arpu_active"] / s_te["churn"]

# Formula 2, per customer, for a 6-month horizon.
freq, mon = Xte["Frequency"].to_numpy(float), Xte["Monetary"].to_numpy(float)
aov = np.where(freq > 0, mon / np.maximum(freq, 1), 0.0)
orders_per_6m = freq / 2.0  # Frequency counts the last 12 months
naive = np.clip(aov * orders_per_6m * 1.0, 0, None)  # lifespan = the whole 6 months, margin = 1
k = float(ytr.sum() / np.clip(Xtr["Monetary"].to_numpy(float), 0, None).sum())  # fitted on train
fitted = np.clip(mon, 0, None) * k
average = np.full(len(yte), ytr.mean())


def row(name, how, pred, fitted_on_train):
    e = evaluate(yte, pred)
    return {"name": name, "how": how, "fitted_on_train": fitted_on_train, "R2": e["R2"], "MAE": e["MAE"],
            "Capture@20%": e["Capture@20%"], "total_pred_over_actual": float(pred.sum() / yte.sum())}


table = [
    row("One average for everyone", "Every customer gets the same number: the average 6-month spend on the training period.", average, "one number"),
    row("AOV x orders x 6 months, assume everyone stays", "Order value x orders per 6 months x lifespan of 1 period (the whole 6 months). Nothing fitted.", naive, "nothing"),
    row("Same, with the lifespan fitted on train", f"The same formula with the lifespan set to {2 * k:.2f} of the period, fitted on the training period. This equals 0.37 x past spend.", fitted, "one number"),
    row("Model C", "The boosted-tree multiplier times past spend.", pred_c, "a model"),
]
for r in table:
    print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items() if k_ in ("name", "R2", "MAE", "Capture@20%", "total_pred_over_actual")})

# Diagnostic only (uses test outcomes, so it is NOT a result): why does the unfitted 0.5 rule beat the fitted 0.37 rule on R2?
k_test = float(yte.sum() / np.clip(mon, 0, None).sum())
rescaled_c = pred_c * yte.sum() / pred_c.sum()
diag = {"test_best_k": k_test, "model_C_rescaled_to_actual_total": evaluate(yte, rescaled_c),
        "past_spend_ratio_train": k, "past_spend_ratio_test": k_test}

out = {
    "diagnostic": diag,
    "textbook": {"arpu": 50, "margin": 0.8, "churn": 0.05, "clv": 50 * 0.8 / 0.05},
    "ours": {"train": {**s_tr, "clv_formula1": f1_tr}, "test": {**s_te, "clv_formula1": f1_te},
             "lifetimes_periods_train": 1 / s_tr["churn"]},
    "fitted_k": k, "implied_lifespan_fraction": 2 * k,
    "table": table,
}
(ROOT / "site/public/data/heuristics.json").write_text(json.dumps(out, indent=1))
print("train", s_tr, f1_tr, "\ntest", s_te, f1_te, "\nk", k)
