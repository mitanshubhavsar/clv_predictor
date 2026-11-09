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

---

## Old project, checked in detail: the level we have to match (then improve)

1. Streamlit "backtest inspector" (dashboard/app.py, about 150 lines): sidebar shows which saved files exist; tabs Overview (best model, snapshot sizes, R2 with and without the top customer, top-20% capture vs past-spend sort with intervals), Model comparison (table + SHAP images), Segments (summary table + filterable customer table). Reads saved files only.
2. Public site (site/, React 18 + TypeScript + Vite, about 1,170 lines): one page with sections About, Dataset, Architecture, Results, Try it; skip link, active-section nav, accessible tables and figures; loads 4 JSON files (summary, dataset, scorer, capture_curve); every number comes from the saved metrics, none typed by hand; footer credits the data licence.
3. Live scorer (Scorer.tsx): preset customers from each segment, inputs, calls the API /score, shows spend, repurchase probability and top reasons; checks /health first and shows a "waking" status because Render sleeps; falls back to saved scores when the API is offline or VITE_API_URL is unset.
4. Generated figures (scripts/make_site_images.py, matplotlib, one consistent style): revenue-capture curves, model error comparison, SHAP importance, pipeline flow diagram, snapshot timeline, serving diagram.
5. Data export (scripts/export_site_data.py): writes the JSON from metrics.json, saved predictions and the raw data, so nothing on the site is hand-typed.
6. Packaging: serving/Dockerfile, .dockerignore, requirements-api.txt; CI (ruff, pytest, import check with only API requirements, docker build, container smoke test); Pages workflow (npm ci, type-check, build, deploy to GitHub Pages, VITE_API_URL repository variable); API on Render.

Decision (user): build the public site as React + Vite like the old one. Node and npm were not installed, so Node 22.22.0 was installed into the ml_clv conda env (conda install -c conda-forge nodejs=22; it also updated ca-certificates and openssl in that env). Use it with PATH=/data/home/mitanshub/miniconda3/envs/ml_clv/bin:$PATH.

Build order: (1) data export + saved scores file, (2) figures, (3) Streamlit dashboard, (4) React site and live scorer, (5) Docker, requirements-api.txt, CI and Pages workflow. Nothing is pushed or deployed to GitHub or Render without asking first.

---

## STEP 17a done: data export for the dashboard and the site   [scripts/export_site_data.py]
Like the old export script, it writes everything the dashboard and the site read, from the saved model and the data, so nothing is typed by hand:
- outputs/test_predictions.csv: 4,326 test customers with features, predicted spend, multiplier, probability of buying again, segment and actual spend.
- outputs/model_comparison.csv: 5-fold CV on train for every model tried (mean, past spend, log-Ridge, XGBoost, Tweedie, Hurdle, ratio Tweedie, model C). Model C: R2 0.760, MAE 503, Capture 0.746; past spend 0.746 / 523 / 0.739.
- outputs/shap_importance.csv: interval 33%, recency 31%, overdue ratio 22%, monetary 8%, frequency 7%.
- site/public/data/summary.json, capture_curve.json, dataset.json, scorer.json (headline metrics with intervals, model C minus past spend with bootstrap intervals, segment table, targeting lift, a customer followed from raw rows to features to prediction, four preset customers for the live scorer, slider ranges).
Checks built into the script: the curve at 20% must match metrics.json. Values reproduce step 9: model C minus past spend is R2 +0.010 [+0.001, +0.025], MAE gain +18.9 GBP [+10.0, +29.1], Capture -0.000 [-0.009, +0.007].

Next: (2) figures with matplotlib, (3) Streamlit dashboard, (4) React site and live scorer, (5) Docker, requirements-api.txt, CI and Pages workflow.

---

## STEP 17b done: figures   [scripts/make_site_images.py]
Six matplotlib figures written to site/public/img/, drawn only from saved files (capture_curve.json, model_comparison.csv, shap_importance.csv): capture_curves.png, model_comparison.png (MAE bars; past spend orange, model C blue, others grey), shap_importance.png, pipeline.png, snapshot_timeline.png, serving.png. Looked at each one; fixed two layout problems (diagram boxes cut off at the right edge, timeline labels cut off inside short bars).

Next: (3) Streamlit dashboard, (4) React site and live scorer, (5) Docker, requirements-api.txt, CI and Pages workflow.

---

## STEP 17c done: Streamlit dashboard   [dashboard/app.py]
Backtest inspector in the old project's style: reads only saved files (outputs/metrics.json, test_predictions.csv, model_comparison.csv, shap_importance.csv, site/public/data/summary.json and capture_curve.json), never refits anything. Sidebar lists which artifacts exist; if any is missing it stops with the commands to rebuild them. Tabs: Overview (model C R2, MAE, Capture with intervals, difference vs past spend, targeting, capture curves), Model comparison (table, SHAP), Segments, Customers (filter by segment, top 20%, CSV download), Explain (one customer: multiplier x scale), Limits.
Check: ran headless with streamlit's AppTest: no exceptions, metrics shown 0.716 / £594 / 0.740 (match step 9). Not opened in a browser.
Run:  streamlit run dashboard/app.py

Next: (4) React site and live scorer, (5) Docker, requirements-api.txt, CI and Pages workflow.

---

## STEP 17d done: React + Vite + TypeScript site   [site/]
Same shape as the old project: one page with About, Dataset, Architecture, Results, Try it; skip link, active-section nav, tables in scrollable wrappers, light/dark theme from the OS. Files: site/src/App.tsx (sections), Scorer.tsx (live scorer), lib.ts, types.ts, styles.css. Reads site/public/data/{summary,dataset,scorer}.json and site/public/img/*.png, so every number comes from the export script.
Live scorer: example customers from each segment, four inputs, checks API_URL/health (shows a "waking" message because free hosting sleeps), POSTs to /score, and falls back to the saved backtest score when the API is unset or offline (only for unedited examples; edited inputs need the API and the page says so). API address from VITE_API_URL.
Honesty built into the About section: states that sorting by past spend is almost as good and the model is about 3% better in MAE.
Checks: npm install, npm run build (tsc type-check + vite build) pass; vite preview serves the page, data and images (HTTP 200). Needed @types/node for vite.config.ts. Not yet looked at in a browser, and the live scorer has not been tested against the running API (API CORS currently allows localhost:5173, 8501 only; add the Pages address when deploying).
Build base path: VITE_BASE (default /clv-stepwise/), set to the repository name when publishing.
Run:  cd site; PATH=/data/home/mitanshub/miniconda3/envs/ml_clv/bin:$PATH npm run dev

Next: (5) Docker, requirements-api.txt, CI and Pages workflow.

---

## STEP 17e done: two new site sections, "Flow of the application" and "Business decisions"   [site/src/Story.tsx, site/public/data/story.json, scripts/make_site_images.py]
Requested by the user. Written from steps_followed_1 to 5, baseline_metrics_1, model_tried_summary_2, feature_adding_3, model_interpretation_4 and business_decisions_5.
- Flow: a 13-stage diagram (new figure flow.png) and one card per stage (what we did, why, what we found, which notes file), including the dead ends: gross spend vs netting, models that only matched past spend, the keep rule that changed from fold-split tests to a bootstrap over customers. The model-comparison table and the feature verdict table are shown inside their stages.
- Business decisions: six cards, each with how the conclusion was reached, the result and how far to trust it, tagged DATA, RULE OF THUMB or ARITHMETIC ON ASSUMPTIONS exactly as in business_decisions_5.md, plus "what would strengthen these conclusions".
- Numbers: the targeting figures are filled from summary.json via placeholders; the rest of the narrative numbers are copied by hand from the notes into story.json (that file says so). If the notes change, story.json has to be edited.
- Step 13 and the BG/NBD baseline are not mentioned, per earlier decisions.
Check: tsc type-check and vite build pass. Not yet looked at in a browser.

---

## STEP 17f done: "No ML: heuristic CLV" section   [scripts/export_heuristics.py, site/public/data/heuristics.json, site/src/Story.tsx]
Requested by the user (pasted the two textbook formulas). Section shows: (1) CLV = ARPU x margin / churn with the textbook example ($50 x 0.8 / 0.05 = $800) and our data (6-month periods, revenue only: ARPU of buyers 1,485 GBP, 52% buy nothing next period, so about 2,860 GBP revenue over about 1.9 periods); (2) AOV x orders x lifespan x margin applied per customer for a 6-month horizon (= half of past spend). Limits stated: formula 1 is one number for everyone (scores like the mean); churn in retail is only a proxy; no margin data.
Scored on the test snapshot (same measures as model C): average for everyone R2 -0.002, MAE 1,086, capture 0.163; "everyone stays" rule (0.5 x past spend, nothing fitted) R2 0.822, MAE 599, capture 0.739, total 0.98 of actual; lifespan fitted on train (0.37 x past spend) 0.707 / 613 / 0.739; model C 0.716 / 594 / 0.740.
FINDING worth knowing: the unfitted 0.5 rule has HIGHER R2 than model C (0.82 vs 0.72). Cause is a level shift: actual future spend per pound of past spend was 0.37 on train and 0.51 on test, so everything fitted on train under-predicts totals (model C 69%, past spend 73% of actual). Diagnostic only, uses test outcomes so it is NOT a result: model C rescaled to the actual total would give R2 0.85. Ranking and MAE are unaffected (capture identical, MAE 599 vs 594). The section says this on the page. Possible later improvement to discuss: handle period-to-period level shift (for example check the ratio on the most recent data before using the model).
Build and type-check pass; not yet looked at in a browser.

---

## STEP 18a done: site design, quick version of direction A (editorial)   [site/src/styles.css only]
Requested by the user: make the site look better first, interaction later; direction A (editorial / data-journalism), step by step toward the later items (hero, palette and fonts, alternating sections, segment cards and timeline).
Done in CSS only, no content or number changed: warm paper background, serif headings (system serif stack, no external font), one blue accent and one orange highlight, section numbers (01, 02, ...) from a CSS counter, large serif hero headline, three hero numbers as a ruled strip instead of boxes, the honest summary as a pull-quote with an orange bar, reading width 720px with wider charts and tables, tables with ruled uppercase headers, blurred sticky header with the current section underlined, cards for decisions, numbered timeline for the flow, refreshed scorer. Dark mode kept (own warm dark palette).
Checked by screenshots in headless Chromium (Playwright from the ml_clv env) of the top of the page and of Business decisions. Not yet checked: phone width, dark mode, Results, Flow and Try-it.
Note: `vite preview` served the built JS as 404 here (the /clv-stepwise/ base), so screenshots came from a build with base "/" and a plain static server. `npm run dev` is not affected.
Next design steps (not started): fonts (Inter + a display serif), alternating section backgrounds, segment cards, icons, then charts.

---

## STEP 18b done: hero section   [site/src/App.tsx, site/src/Icons.tsx, site/src/styles.css]
Design item 2 of the agreed list. The About section is now a hero panel: small uppercase kicker (customer lifetime value, UCI Online Retail II), large serif headline, lead text, two buttons (See the results -> #results, Try the scorer -> #try), a soft blue/orange gradient with a faint decorative rising curve and customer dots (inline SVG, aria-hidden), and the three headline numbers with an icon each (fit, error, top-20% funnel; inline SVG, no icon dependency). The honest-summary pull-quote stays below the numbers. The numbers still come from summary.json.
Also: on phones the top navigation is one scrollable row instead of wrapping to three lines.
Checked by screenshots in headless Chromium: desktop light, desktop dark, phone width; headline, buttons and the three numbers fit in the first desktop screen. Other sections were not re-checked on phone or dark.
Next in the agreed list: 3 (palette and fonts), 4 (alternating sections and card layout), 6 (segment cards, timeline polish).
