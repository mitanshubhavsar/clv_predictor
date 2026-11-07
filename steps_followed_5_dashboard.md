# Steps followed, part 5: dashboard and API

Starting point: model C and the return classifier are saved in outputs/ (step 15), with outputs/model_spec.json and outputs/metrics.json.

## How the old project (ML_proj2_CIP_CLV) did it, which we follow

| Piece | Old project | What it showed |
|---|---|---|
| API | serving/app.py, FastAPI: GET /health, GET /model-info, POST /score (one customer from features), POST /score/transactions (CSV upload, limits 5 MB and 50,000 rows). Model loaded once at startup in a lifespan handler; reasons from XGBoost pred_contribs instead of shap; CORS for the GitHub Pages site and localhost. | Same code path as training for the features; small, lean image |
| Container | serving/Dockerfile (python 3.11-slim, requirements-api.txt only: fastapi, uvicorn, pydantic, python-multipart, pandas, numpy, scikit-learn, xgboost, joblib; healthcheck on /health; port 8000) | Training libraries (shap, optuna, streamlit) stay out of the serving image |
| Where it ran | Render (free web service built from the repo, sleeps when idle, first request can take about a minute): https://clv-predictor-j24w.onrender.com/docs | API is public |
| Dashboard | dashboard/app.py, Streamlit, an internal "backtest inspector": reads only saved files from outputs/, never re-runs the pipeline; sidebar shows which artifacts exist; tabs Overview, Model comparison, Segments | Cannot disagree with the saved results |
| Public site | site/ React + Vite (About, Dataset, Architecture, Results, live Scorer), built and published to GitHub Pages by .github/workflows/pages.yml; reads JSON written by scripts/export_site_data.py so no number is typed by hand; the scorer calls the API (VITE_API_URL repository variable) and falls back to saved scores when the API is asleep or unset | The public face; API wakes on demand |
| CI | .github/workflows/ci.yml: ruff lint, pytest, import test with ONLY requirements-api.txt, docker build, smoke test of the container | Catches packaging bugs early |

## Plan for CLV_stepwise (same layout, adapted to model C)

STEP 16: API (serving/app.py)
- 16a: startup loads model C, the return classifier, model_spec.json, metrics.json; GET /health, GET /model-info (honest numbers + limitations from the saved metrics).
- 16b: POST /score: Recency, Frequency, Monetary, IPI (optional) -> predicted 6-month spend, probability of buying again, segment, the multiplier.
- 16c: reasons: why this multiplier (XGBoost pred_contribs, plain words), no shap at serving time.
- 16d: POST /score/transactions: raw invoice-line CSV + optional cutoff date; same cleaning and feature code as training; one row per customer.
- 16e: tests with the FastAPI TestClient: API predictions equal the offline model's exactly.

STEP 17: dashboard (dashboard/app.py, Streamlit, reads saved files only)
- Overview (headline metrics with intervals, targeting lift), Customers (scored table, filter by segment, top 20%, CSV download), Segments, Explain (one customer), Limits.

STEP 18: packaging
- requirements-api.txt, serving/Dockerfile, run instructions, import test with only the API requirements.

STEP 19 (later, needs the user's go-ahead because it is public): public site (React on GitHub Pages) and API on Render. Nothing is pushed or deployed without asking first.

Not decided yet (to confirm): whether to build the public React site, or stop at the local dashboard and a Docker-ready API.

---

## STEP 16 done: API   [serving/app.py, tests/test_api.py]
Built the way the old project did it: model, classifier, spec and metrics loaded once at startup (lifespan), CORS for localhost, upload limits 5 MB and 50,000 rows, reasons from XGBoost pred_contribs (no shap).

Endpoints:
- GET /health -> {"status": "ok"}
- GET /model-info -> model description, columns, cutoffs, segment rules, test metrics (past spend vs model C) and the limitations, read from outputs/model_spec.json and outputs/metrics.json.
- POST /score {Recency, Frequency, Monetary, IPI (optional)} -> predicted 6-month spend (GBP), multiplier, probability of buying again, segment, and reasons. OverdueRatio is computed on the server as Recency / max(IPI, 1), so a caller cannot send inconsistent values. IPI omitted = customer bought on only one day (blank, handled natively by the model).
- POST /score/transactions (CSV of invoice lines with the raw columns, optional ?cutoff=YYYY-MM-DD, default = day after the last invoice in the file) -> one row per customer. Uses the same functions as the offline pipeline (clean, build_ledger, make_snapshot, build_features), so the features cannot drift from training.

Example: Recency 20, Frequency 6, Monetary 2,500, IPI 45 -> 970 GBP, multiplier 0.388, probability 0.77, Protect. Reasons are given as "multiplier x% higher than average" per feature (here +1.7% overdue ratio, +1.6% interval, +1.2% recency, -0.3% frequency, -0.2% monetary): small nudges, as expected.

Tests (6, all pass; python -m pytest tests/test_api.py):
- health and model-info (limitations are present)
- /score equals the offline model on 25 random test-snapshot customers (within 0.01 GBP)
- bad input rejected (negative recency, zero frequency -> 422)
- /score/transactions on the raw rows of 120 test customers, INCLUDING their rows after the cutoff, equals the offline predictions: the API cannot see the future
- oversized upload -> 413; CSV with wrong columns -> 400
Note found while writing the test: the full raw file is far over the 5 MB / 50,000-row limit, so the test sends a subset of customers (feature building is per customer, so a subset gives identical numbers).

Run it:  uvicorn serving.app:app --port 8000   then open http://localhost:8000/docs

Fix after first run: opening http://127.0.0.1:8000 showed {"detail":"Not Found"} because only the API endpoints existed and there was no page at "/". Added a home page at "/" (what the model is, links to /docs, /model-info and /health, and a small form that calls /score and shows spend, probability, segment and reasons). Test added (test_home_page); 7 tests pass. Restart uvicorn to see it (or use --reload).
