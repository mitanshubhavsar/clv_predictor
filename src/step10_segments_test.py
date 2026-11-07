"""Step 10c: the ONE test-set check of the segments (cutoff 2011-06-01).

Everything is fixed on TRAIN, then applied to test:
  model C and the logistic classifier are fitted on all of train;
  "high value" cut-off = 80th percentile of train out-of-fold predictions (~721 GBP);
  "likely to return" = probability > 0.5;
  hand rule: Monetary >= train 80th percentile, Recency <= 90 days.
Sensitivity (stated in advance, not a second selection): high value = top 20% of
the TEST predictions instead of the train cut-off.

Run:  python src/step10_segments_test.py
"""
import numpy as np

from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot
from step7b_xgboost import oof_with_seed
from step7e_ratio import RatioXGB
from step8_features import build_features
from step8_final import BEST_8, KEPT
from step10_repurchase_segmentation import LogisticRFM
from step10_segments import segment, summarize


def main() -> None:
    ledger = load_ledger()
    tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
    Xtr, Xte = build_features(tr, KEPT), build_features(te, KEPT)
    ytr, yte = tr.target, te.target

    cutoff = np.quantile(oof_with_seed(lambda: RatioXGB(objective="tweedie", **BEST_8), Xtr, ytr, 0), 0.8)
    value = RatioXGB(objective="tweedie", **BEST_8).fit(Xtr, ytr).predict(Xte)
    prob = LogisticRFM().fit(Xtr, (ytr > 0).astype(int)).predict_proba(Xte)
    print(f"test customers {len(yte)}; bought again {np.mean(yte > 0):.1%}; train cut-off for high value: {cutoff:,.0f} GBP")

    high = value >= cutoff
    seg = segment(high, prob > 0.5)
    print(f"\nMODEL SEGMENTS, train cut-offs (high value = {high.mean():.1%} of test customers)")
    print(summarize(seg, yte).to_string())

    rule = segment(Xte["Monetary"].to_numpy() >= np.quantile(Xtr["Monetary"], 0.8), (Xte["Recency"] <= 90).to_numpy())
    print("\nHAND RULE (train cut-off for past spend, bought in last 90 days)")
    print(summarize(rule, yte).to_string())

    seg2 = segment(value >= np.quantile(value, 0.8), prob > 0.5)
    print("\nSENSITIVITY: high value = top 20% of the test predictions")
    print(summarize(seg2, yte).to_string())

    hv = high
    print(f"\nhigh-value overlap model vs rule: {(hv & (Xte['Monetary'].to_numpy() >= np.quantile(Xtr['Monetary'], 0.8))).sum()} of {hv.sum()}")
    print(f"share of high-value unlikely to return: {(hv & (prob <= 0.5)).sum() / hv.sum():.1%}")


if __name__ == "__main__":
    main()
