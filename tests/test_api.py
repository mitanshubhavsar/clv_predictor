"""Step 16e: the API must give exactly the offline model's predictions, and future rows must not leak in.

Run:  PYTHONNOUSERSITE=1 python -m pytest tests/test_api.py -q
"""
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from serving.app import app  # noqa: E402
from step3_split import TEST_CUTOFF, load_ledger, make_snapshot  # noqa: E402
from step8_features import build_features  # noqa: E402
from step8_final import KEPT  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def offline():
    te = make_snapshot(load_ledger(), TEST_CUTOFF)
    X = build_features(te, KEPT)
    d = joblib.load(ROOT / "outputs" / "model_c.joblib")
    return X, d["model"].predict(X[d["columns"]])


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Customer spend scorer" in r.text and "/score" in r.text


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["columns"] == ["Recency", "Frequency", "Monetary", "IPI", "OverdueRatio"]
    assert "not causal" in " ".join(info["limitations"]).lower()


def test_score_matches_offline_model(client, offline):
    X, pred = offline
    for i in np.random.default_rng(0).choice(len(X), 25, replace=False):
        r = X.iloc[i]
        body = {"Recency": float(r["Recency"]), "Frequency": int(r["Frequency"]), "Monetary": float(r["Monetary"])}
        if not np.isnan(r["IPI"]):
            body["IPI"] = float(r["IPI"])
        out = client.post("/score", json=body).json()
        assert out["predicted_spend_gbp"] == pytest.approx(pred[i], abs=0.01)


def test_score_rejects_bad_input(client):
    assert client.post("/score", json={"Recency": -5, "Frequency": 3, "Monetary": 100}).status_code == 422
    assert client.post("/score", json={"Recency": 5, "Frequency": 0, "Monetary": 100}).status_code == 422


def test_transactions_match_offline_and_ignore_future_rows(client, offline):
    X, pred = offline
    ids = np.random.default_rng(1).choice(X.index.to_numpy(), 120, replace=False)
    raw = pd.read_csv(ROOT / "data" / "raw.csv")
    sub = raw[raw["Customer ID"].isin(ids)]  # all their rows, INCLUDING those after the cutoff, on purpose
    assert len(sub) < 50_000
    buf = io.BytesIO()
    sub.to_csv(buf, index=False)
    assert len(buf.getvalue()) < 5 * 1024 * 1024
    r = client.post("/score/transactions", params={"cutoff": str(TEST_CUTOFF.date())},
                    files={"file": ("tx.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    got = pd.DataFrame(r.json()["results"]).set_index("customer_id")["predicted_spend_gbp"]
    assert set(got.index) == set(ids)
    expected = pd.Series(pred, index=X.index).loc[got.index]
    assert np.allclose(got.to_numpy(), expected.to_numpy(), atol=0.01)


def test_transactions_too_large_is_rejected(client):
    big = b"x" * (5 * 1024 * 1024 + 10)
    assert client.post("/score/transactions", files={"file": ("big.csv", big, "text/csv")}).status_code == 413


def test_transactions_bad_file(client):
    r = client.post("/score/transactions", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400
