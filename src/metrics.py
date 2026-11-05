"""The three metrics used to compare every model and baseline. All on the
ORIGINAL money scale (GBP): a model trained on log1p must be converted back
before scoring.

  R2           how well do the predicted amounts fit? (0 = no better than
               predicting the average, 1 = perfect)
  MAE          typical error in GBP (not dominated by a few huge customers)
  Capture@20%  if we contact the 20% of customers ranked highest, what share
               of all actual future money do they hold? (random ~ 20%)

Helpers (not metrics): oracle_capture (the ceiling), drop_top (robustness
check without the biggest customers), bootstrap_ci (noise, for close calls).
"""
import numpy as np


def r2(y, pred) -> float:
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def mae(y, pred) -> float:
    return float(np.mean(np.abs(y - pred)))


def revenue_capture(y, score, frac: float = 0.2, seed: int = 0) -> float:
    """Share of all actual money held by the top `frac` customers by `score`.

    Ties (e.g. a constant prediction) are broken at random with a fixed seed,
    so a constant baseline scores about `frac`, as pure chance should.
    """
    y, score = np.asarray(y, float), np.asarray(score, float)
    n_top = int(round(frac * len(y)))
    tiebreak = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((tiebreak, -score))  # primary key: -score, secondary: random
    return float(y[order[:n_top]].sum() / y.sum())


def evaluate(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"R2": r2(y, pred), "MAE": mae(y, pred), "Capture@20%": revenue_capture(y, pred)}


def oracle_capture(y, frac: float = 0.2) -> float:
    """Best possible Capture@frac: rank by the true value."""
    return revenue_capture(y, y, frac)


def drop_top(y, pred, n: int = 10):
    """Remove the n customers with the largest ACTUAL value (is the score
    driven by a handful of people?)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    keep = np.argsort(-y)[n:]
    return y[keep], pred[keep]


def bootstrap_ci(y, pred, metric: str, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for one metric, resampling customers."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(evaluate(y[idx], pred[idx])[metric])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
