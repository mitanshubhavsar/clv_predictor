"""Step 11: what drives model C? (SHAP on the tree part of the ratio model)

Model C predicts   prediction = multiplier(R,F,M,IPI,Overdue) x scale,  scale = max(Monetary, 50).
The trees only learn the MULTIPLIER (Tweedie loss, log link), so SHAP explains the
log of the multiplier: a SHAP value of +0.1 means "multiplier x 1.105 vs the average
customer". Monetary acts twice: as the scale (outside the trees, proportional) and
as a column inside the trees.

Fitted on train (all customers), explained on train. Test not needed for this step.

Run:  python src/step11_shap.py
"""
import numpy as np
import pandas as pd
import shap

from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step7e_ratio import RatioXGB
from step8_features import build_features
from step8_final import BEST_8, KEPT


def fit_and_explain():
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_features(train, KEPT), train.target
    model = RatioXGB(objective="tweedie", **BEST_8).fit(X, y)
    booster = model.model_
    sv = shap.TreeExplainer(booster)(X)  # values in log(multiplier) units
    margin = booster.predict(X, output_margin=True)
    assert np.allclose(sv.values.sum(1) + sv.base_values, margin, atol=1e-3), "SHAP does not add up"
    return X, y, model, sv


def main() -> None:
    X, y, model, sv = fit_and_explain()
    mult = np.exp(sv.values.sum(1) + sv.base_values)
    print(f"customers {len(X)}; average multiplier {mult.mean():.3f}; spread (5th to 95th pct) {np.percentile(mult, 5):.3f} to {np.percentile(mult, 95):.3f}")
    imp = pd.Series(np.abs(sv.values).mean(0), index=X.columns).sort_values(ascending=False)
    t = pd.DataFrame({"mean |SHAP|": imp, "share": imp / imp.sum()})
    t["cum share"] = t["share"].cumsum()
    t["direction (corr with SHAP)"] = [np.corrcoef(X[c].fillna(X[c].median()), sv.values[:, list(X.columns).index(c)])[0, 1] for c in t.index]
    print("\nGlobal importance on the multiplier:")
    print(t.round(3).to_string())

    # value of the multiplier by bands of each main feature
    print("\nAverage multiplier by quintile of each feature:")
    out = {}
    for c in ("Recency", "Frequency", "Monetary"):
        q = pd.qcut(X[c].rank(method="first"), 5, labels=False)
        out[c] = pd.Series(mult).groupby(q.to_numpy()).mean()
    print(pd.DataFrame(out).round(3).to_string())
    print("\nreal multiplier (future / scale) by the same quintiles:")
    scale = X["Monetary"].clip(lower=50).to_numpy(); real = y.to_numpy() / scale
    out = {c: pd.Series(real).groupby(pd.qcut(X[c].rank(method="first"), 5, labels=False).to_numpy()).mean() for c in ("Recency", "Frequency", "Monetary")}
    print(pd.DataFrame(out).round(3).to_string())

    # pounds view: how much of the final prediction comes from past spend (the scale) vs the multiplier
    pred = model.predict(X)
    print(f"\nrank correlation, prediction vs past spend (Monetary): {pd.Series(pred).corr(X['Monetary'].reset_index(drop=True), method='spearman'):.3f}")
    print(f"rank correlation, prediction vs multiplier alone:      {pd.Series(pred).corr(pd.Series(mult), method='spearman'):.3f}")


if __name__ == "__main__":
    main()
