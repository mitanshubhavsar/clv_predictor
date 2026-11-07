"""Step 12: targeting simulation (what the old project called the "A/B simulation").

Honest scope: this measures RANKING value, not campaign ROI. We pick 20% of customers
by some rule and look at the REAL money they spend in the next 6 months, compared with
a random 20%. There is no treatment, no cost and no causal effect here: those customers
would mostly spend that money anyway. A final break-even table turns it into arithmetic
on explicit assumptions (cost per contact, uplift), nothing more.

Models are fitted on train; the test snapshot is the evaluation (model C already scored
on it in step 9). Train out-of-fold results are shown as a second view.

Run:  python src/step12_targeting.py
"""
import numpy as np
import pandas as pd

from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot
from step7b_xgboost import oof_with_seed
from step7e_ratio import RatioXGB
from step8_features import build_features
from step8_final import BEST_8, KEPT
from step10_repurchase_segmentation import LogisticRFM

FRAC = 0.2


def top_mask(score: np.ndarray, frac: float = FRAC, seed: int = 0) -> np.ndarray:
    k = int(round(frac * len(score)))
    tie = np.random.default_rng(seed).random(len(score))
    idx = np.lexsort((tie, -score))[:k]
    m = np.zeros(len(score), bool); m[idx] = True
    return m


def compare(y: np.ndarray, scores: dict, n_boot: int = 1000) -> pd.DataFrame:
    overall = y.mean()
    rows, masks = {}, {k: top_mask(s) for k, s in scores.items()}
    masks["oracle (perfect ranking)"] = top_mask(y)
    rng = np.random.default_rng(0)
    boots = {k: [] for k in masks}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for k, s in {**scores, "oracle (perfect ranking)": y}.items():
            boots[k].append(y[i][top_mask(s[i], seed=1)].mean() / y[i].mean() - 1)
    for k, m in masks.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        rows[k] = {"mean future spend, targeted 20% (GBP)": y[m].mean(), "random 20% (GBP)": overall,
                   "lift vs random": y[m].mean() / overall - 1, "lift 95% interval": f"[{lo:.2f}, {hi:.2f}]",
                   "share of all money reached": y[m].sum() / y.sum()}
    return pd.DataFrame(rows).T


def decile_table(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-score, kind="stable")
    d = pd.DataFrame({"y": y[order], "decile": np.repeat(np.arange(1, 11), int(np.ceil(len(y) / 10)))[:len(y)]})
    g = d.groupby("decile")["y"].agg(["mean", "sum"])
    g["lift vs average"] = g["mean"] / y.mean(); g["share of money"] = g["sum"] / y.sum()
    return g.drop(columns="sum").round(3)


def main() -> None:
    ledger = load_ledger()
    tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
    Xtr, Xte = build_features(tr, KEPT), build_features(te, KEPT)
    mk = lambda: RatioXGB(objective="tweedie", **BEST_8)

    # TEST
    yt = te.target.to_numpy()
    value = mk().fit(Xtr, tr.target).predict(Xte)
    prob = LogisticRFM().fit(Xtr, (tr.target > 0).astype(int)).predict_proba(Xte)
    scores = {"model C": value, "past spend (Monetary)": Xte["Monetary"].to_numpy(),
              "model C x P(return)": value * prob, "recency only": -Xte["Recency"].to_numpy()}
    pd.set_option("display.width", 220)
    print(f"TEST ({len(yt)} customers, average future spend {yt.mean():,.0f} GBP)")
    t = compare(yt, scores); print(t.round(3).to_string())
    print("\nDecile table, model C (test):"); print(decile_table(yt, value).to_string())

    # TRAIN out-of-fold
    ya = tr.target.to_numpy()
    v = oof_with_seed(mk, Xtr, tr.target, 0)
    print(f"\nTRAIN out-of-fold ({len(ya)} customers, average {ya.mean():,.0f} GBP)")
    print(compare(ya, {"model C": v, "past spend (Monetary)": Xtr["Monetary"].to_numpy()}).round(3).to_string())

    # break-even arithmetic (assumptions, not results)
    m = top_mask(value); n_t = m.sum(); avg_t = yt[m].mean()
    print(f"\nBreak-even for a campaign on the targeted 20% (average future spend {avg_t:,.0f} GBP per customer):")
    print("a contact costing C pays for itself if the campaign raises that customer's spend by at least C / spend")
    for cost in (5, 20, 50):
        print(f"  cost {cost:>3} GBP per customer -> needs uplift >= {cost / avg_t:.1%} of their spend  (random 20%: {cost / yt.mean():.1%})")


if __name__ == "__main__":
    main()
