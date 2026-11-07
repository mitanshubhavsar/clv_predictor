"""serving/app.py: FastAPI server for model C (same layout as the old project).

Loads the saved model, the return classifier and the spec ONCE at startup, then scores requests with
the same code the offline pipeline uses (src/step2_clean, step3_split, step4_rfm, step8_features).

Endpoints
  GET  /health                -- liveness
  GET  /model-info            -- model description, test metrics and limitations (from outputs/)
  POST /score                 -- one customer from Recency, Frequency, Monetary, IPI
  POST /score/transactions    -- CSV of invoice lines (+ optional cutoff date) -> one row per customer

Reasons use XGBoost's own pred_contribs (no shap needed at serving time).

Run from the project root:   uvicorn serving.app:app --port 8000
"""
import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from step2_clean import build_ledger, clean  # noqa: E402
from step3_split import make_snapshot  # noqa: E402
from step7e_ratio import FLOOR  # noqa: E402
from step8_features import build_features  # noqa: E402

OUT = ROOT / "outputs"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 50_000
RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
SEGMENT_NAMES = {(True, True): "Protect", (True, False): "Win back", (False, True): "Grow", (False, False): "Low priority"}
FEATURE_WORDS = {"Recency": "days since the last purchase", "Frequency": "number of purchase invoices",
                 "Monetary": "past net spend", "IPI": "usual days between purchases",
                 "OverdueRatio": "how overdue the customer is against their own rhythm"}

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    for f in ("model_c.joblib", "logistic_return.joblib", "model_spec.json", "metrics.json"):
        if not (OUT / f).exists():
            raise RuntimeError(f"{OUT / f} not found. Run `python src/step15_save.py` first.")
    mc = joblib.load(OUT / "model_c.joblib")
    state["model"], state["columns"] = mc["model"], mc["columns"]
    state["clf"] = joblib.load(OUT / "logistic_return.joblib")["model"]
    state["spec"] = json.load(open(OUT / "model_spec.json"))
    state["metrics"] = json.load(open(OUT / "metrics.json"))
    state["high_cutoff"] = state["metrics"]["segments_test"]["high_value_cutoff_gbp_from_train"]
    yield
    state.clear()


app = FastAPI(title="CLV scoring API (model C)", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8501"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class CustomerFeatures(BaseModel):
    Recency: float = Field(ge=0, description="days from the last purchase to the scoring date")
    Frequency: int = Field(gt=0, description="distinct purchase invoices in the last 12 months")
    Monetary: float = Field(description="net spend (purchases minus returns) in the last 12 months, GBP")
    IPI: Optional[float] = Field(default=None, ge=0, description="mean days between purchase days; omit if the customer bought on only one day")


def score_frame(X: pd.DataFrame) -> pd.DataFrame:
    """X has Recency, Frequency, Monetary, IPI, OverdueRatio. Returns spend, multiplier, probability, segment."""
    model = state["model"]
    spend = model.predict(X[state["columns"]])
    scale = model.scale(X[state["columns"]])
    prob = state["clf"].predict_proba(X)
    high = spend >= state["high_cutoff"]
    seg = [SEGMENT_NAMES[(bool(h), bool(p > 0.5))] for h, p in zip(high, prob)]
    return pd.DataFrame({"predicted_spend_gbp": spend, "multiplier": spend / scale,
                         "probability_buy_again": prob, "segment": seg}, index=X.index)


def reasons(X_row: pd.DataFrame) -> list[dict]:
    """Per-feature push on the multiplier, from XGBoost's pred_contribs (exact). +x% = multiplier x% higher than average."""
    booster = state["model"].model_.get_booster()
    cols = state["columns"]
    c = booster.predict(xgb.DMatrix(X_row[cols], feature_names=cols), pred_contribs=True)[0]
    out = []
    for name, v in sorted(zip(cols, c[:-1]), key=lambda t: -abs(t[1])):
        pct = (np.exp(v) - 1) * 100
        out.append({"feature": name, "meaning": FEATURE_WORDS[name], "multiplier_effect_pct": round(float(pct), 1)})
    return out


def features_from_request(f: CustomerFeatures) -> pd.DataFrame:
    ipi = np.nan if f.IPI is None else f.IPI
    return pd.DataFrame([{"Recency": f.Recency, "Frequency": float(f.Frequency), "Monetary": f.Monetary,
                          "IPI": ipi, "OverdueRatio": f.Recency / max(ipi, 1.0) if f.IPI is not None else np.nan}], index=["query"])



HOME_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLV scorer</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 16px;line-height:1.5;color:#1b1f24}
 h1{margin-bottom:.2rem} .muted{color:#5b6570} label{display:block;margin:.6rem 0 .2rem;font-weight:600}
 input{padding:.45rem;width:100%;max-width:280px;font-size:1rem} button{margin-top:1rem;padding:.55rem 1.1rem;font-size:1rem;cursor:pointer}
 #out{margin-top:1.2rem;padding:1rem;border:1px solid #d0d7de;border-radius:8px;display:none}
 .seg{font-weight:700} table{border-collapse:collapse;margin-top:.5rem} td,th{padding:.25rem .7rem;border-bottom:1px solid #e5e8eb;text-align:left}
 code{background:#f0f2f4;padding:1px 5px;border-radius:4px}
</style></head><body>
<h1>Customer spend scorer</h1>
<p class="muted">Model C: predicts the money a customer will keep in the next 6 months, from their last 12 months of purchases.
It ranks customers by value; it does not say who will respond to a campaign.</p>
<p>Links: <a href="/docs">interactive API docs</a> &middot; <a href="/model-info">model info and limitations</a> &middot; <a href="/health">health</a></p>
<h2>Try it</h2>
<form id="f">
 <label for="Recency">Days since last purchase</label><input id="Recency" type="number" min="0" step="1" value="20" required>
 <label for="Frequency">Purchase invoices in the last 12 months</label><input id="Frequency" type="number" min="1" step="1" value="6" required>
 <label for="Monetary">Net spend in the last 12 months (GBP)</label><input id="Monetary" type="number" step="0.01" value="2500" required>
 <label for="IPI">Usual days between purchases <span class="muted">(leave empty if they bought on only one day)</span></label><input id="IPI" type="number" min="0" step="0.1" value="45">
 <button type="submit">Score this customer</button>
</form>
<div id="out"></div>
<script>
const out=document.getElementById('out');
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();
 const body={Recency:+Recency.value,Frequency:+Frequency.value,Monetary:+Monetary.value};
 if(IPI.value!=='') body.IPI=+IPI.value;
 out.style.display='block'; out.textContent='Scoring...';
 const r=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){out.textContent='Error '+r.status+': '+await r.text();return;}
 const d=await r.json();
 const rows=d.reasons.map(x=>`<tr><td>${x.feature}</td><td>${x.meaning}</td><td>${x.multiplier_effect_pct>0?'+':''}${x.multiplier_effect_pct}%</td></tr>`).join('');
 out.innerHTML=`<div>Predicted spend, next 6 months: <b>&pound;${d.predicted_spend_gbp.toLocaleString()}</b></div>
  <div>Chance of buying again: <b>${(d.probability_buy_again*100).toFixed(0)}%</b></div>
  <div>Segment: <span class="seg">${d.segment}</span> <span class="muted">(high value = predicted &pound;${d.high_value_cutoff_gbp}+)</span></div>
  <div class="muted">Multiplier on past spend: ${d.multiplier}</div>
  <table><tr><th>Feature</th><th>Meaning</th><th>Effect on multiplier</th></tr>${rows}</table>`;
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return HOME_PAGE


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if "model" in state else "loading"}


@app.get("/model-info")
def model_info() -> dict:
    m, s = state["metrics"], state["spec"]
    return {"model": s["model"], "columns": s["columns"], "window": s["window"], "target": s["target"],
            "train_cutoff": s["train_cutoff"], "test_cutoff": s["test_cutoff"], "segments": s["segments"],
            "test_metrics": {k: m["test"][k] for k in ("past_spend", "model_C_headline")},
            "limitations": ["Predicts who will spend, not who will respond to a campaign (not causal).",
                            "About 3% better than 'past spend x 0.37' in MAE; no better at ranking.",
                            "Under-predicts the largest customers; treat them as named accounts.",
                            "Revenue, not margin."]}


@app.post("/score")
def score(f: CustomerFeatures) -> dict:
    X = features_from_request(f)
    s = score_frame(X).iloc[0]
    return {"predicted_spend_gbp": round(float(s["predicted_spend_gbp"]), 2),
            "multiplier": round(float(s["multiplier"]), 4),
            "probability_buy_again": round(float(s["probability_buy_again"]), 4),
            "segment": s["segment"], "high_value_cutoff_gbp": state["high_cutoff"],
            "reasons": reasons(X)}


@app.post("/score/transactions")
async def score_transactions(request: Request, file: UploadFile = File(...),
                             cutoff: Optional[str] = Query(None, description="scoring date, YYYY-MM-DD; default = day after the last invoice in the file")) -> dict:
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(400, f"Expected a CSV file, got content-type {file.content_type!r}.")
    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES} bytes).")
    try:
        raw = pd.read_csv(io.BytesIO(body), parse_dates=["InvoiceDate"])
    except Exception as e:
        raise HTTPException(400, f"Could not parse the uploaded CSV: {e}") from None
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}. Expected {RAW_COLUMNS}.")
    if len(raw) > MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"Too many rows: {len(raw)} (limit {MAX_UPLOAD_ROWS}).")
    try:
        cut = pd.Timestamp(cutoff) if cutoff else raw["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    except Exception:
        raise HTTPException(400, "cutoff must look like 2011-06-01.") from None

    purchases, returns, _ = clean(raw[RAW_COLUMNS])
    snap = make_snapshot(build_ledger(purchases, returns), cut)  # the target window is empty for live data: ignored
    if len(snap.target) == 0:
        raise HTTPException(400, "No customers with purchases in the 12 months before the cutoff.")
    X = build_features(snap, ("IPI", "OverdueRatio"))
    out = score_frame(X).round({"predicted_spend_gbp": 2, "multiplier": 4, "probability_buy_again": 4})
    out = out.reset_index().rename(columns={"Customer ID": "customer_id"})
    return {"cutoff": str(cut.date()), "n_customers": len(out), "results": out.to_dict(orient="records")}
