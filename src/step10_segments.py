"""Step 10b: the 2x2 segments (train, out-of-fold predictions, test NOT touched).

value axis : model C prediction (GBP), "high" = top 20%
return axis: logistic P(spends again), "likely" = probability > 0.5
Segments: Protect (high, likely), Win back (high, unlikely), Grow (low, likely), Low priority (low, unlikely).

Do they mean anything? Per segment: size, share of future money, mean actual
spend, share who really bought again. Compared with a hand-written rule that
needs no model: high = top 20% by past spend (Monetary), likely = bought in the last 90 days.

Run:  python src/step10_segments.py
"""
import numpy as np
import pandas as pd

from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step7b_xgboost import oof_with_seed
from step8_features import build_features
from step8_final import BEST_8, KEPT
from step7e_ratio import RatioXGB
from step10_repurchase_segmentation import LogisticRFM, oof

NAMES = {(True, True): "Protect", (True, False): "Win back", (False, True): "Grow", (False, False): "Low priority"}


def segment(high: np.ndarray, likely: np.ndarray) -> pd.Series:
    return pd.Series([NAMES[(bool(h), bool(l))] for h, l in zip(high, likely)])


def summarize(seg: pd.Series, y: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"seg": seg.to_numpy(), "y": y.to_numpy()})
    g = d.groupby("seg").agg(customers=("y", "size"), mean_future_spend=("y", "mean"), bought_again=("y", lambda v: (v > 0).mean()))
    g["share of customers"] = g["customers"] / len(d)
    g["share of future money"] = d.groupby("seg")["y"].sum() / d["y"].sum()
    return g.reindex(["Protect", "Win back", "Grow", "Low priority"]).round(3)


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    y = train.target
    X = build_features(train, KEPT)
    value = oof_with_seed(lambda: RatioXGB(objective="tweedie", **BEST_8), X, y, 0)
    prob = oof(LogisticRFM, X, (y > 0).astype(int), 0)

    high = value >= np.quantile(value, 0.8)
    seg = segment(high, prob > 0.5)
    print("MODEL SEGMENTS (top 20% predicted value x probability > 0.5)")
    print(summarize(seg, y).to_string())

    rule_high = X["Monetary"].to_numpy() >= np.quantile(X["Monetary"], 0.8)
    rule_likely = (X["Recency"] <= 90).to_numpy()
    rule = segment(rule_high, rule_likely)
    print("\nHAND-WRITTEN RULE (top 20% past spend x bought in last 90 days)")
    print(summarize(rule, y).to_string())

    print(f"\nhigh-value customers: model {high.sum()}, rule {rule_high.sum()}; overlap {(high & rule_high).sum()}")
    print(f"agreement of segment labels, model vs rule: {(seg.to_numpy() == rule.to_numpy()).mean():.1%}")
    print(f"share of high-value (model) who are unlikely to return: {(high & (prob <= 0.5)).sum() / high.sum():.1%}")
    print(f"corr(value, probability): {np.corrcoef(value, prob)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
