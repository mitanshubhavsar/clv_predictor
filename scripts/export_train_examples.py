"""Step 17g: a few real training rows (features X and target y) for the 'Try models' stage of the site.

Picks six training customers by rule, not by hand: two who spent nothing, then buyers at the median,
75th and 95th percentile of future spend, and the biggest customer. Writes
site/public/data/train_examples.json.

Run:  python scripts/export_train_examples.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot  # noqa: E402
from step8_features import build_features  # noqa: E402
from step8_final import KEPT  # noqa: E402

tr = make_snapshot(load_ledger(), TRAIN_CUTOFF)
X = build_features(tr, KEPT)[["Recency", "Frequency", "Monetary", "IPI", "OverdueRatio"]]
y = tr.target.astype(float)
rng = np.random.default_rng(7)

zero_ids = list(rng.choice(y.index[y == 0], size=2, replace=False))
buyers = y[y > 0].sort_values()
pick = lambda q: buyers.index[int(round(q * (len(buyers) - 1)))]  # noqa: E731
ids = zero_ids + [pick(0.5), pick(0.75), pick(0.95), y.idxmax()]

rows = []
for cid in ids:
    x = X.loc[cid]
    scale = max(float(x["Monetary"]), 50.0)
    rows.append({"customer_id": str(cid), **{k: (None if pd.isna(v) else round(float(v), 2)) for k, v in x.items()},
                 "target": round(float(y.loc[cid]), 2), "label_ratio": round(float(y.loc[cid]) / scale, 3)})
out = {"cutoff": str(TRAIN_CUTOFF)[:10], "n_train": int(len(y)), "rows": rows}
(ROOT / "site/public/data/train_examples.json").write_text(json.dumps(out, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
