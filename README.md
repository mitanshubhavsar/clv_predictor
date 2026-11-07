# clv_predictor
clv_predictor project 

## Run locally

There are three separate programs. Each runs on its own, in its own terminal, on its own port.

| | What it is | Who it is for | Needs the API? | Address |
|---|---|---|---|---|
| **API** (FastAPI) | Backend that holds model C. Takes a customer's numbers, returns a prediction. Only a small test form, no real interface. | Programs (the website) | n/a | http://localhost:8000 |
| **Website** (React + Vite) | Public page: explanation, results and the "Try it" scorer. | Anyone visiting | Only for the live scorer | http://localhost:5173 |
| **Dashboard** (Streamlit) | Internal inspection tool: tables, customer filters, CSV download. | You, to check results | No (reads saved files) | http://localhost:8501 |

What to start:
- **Live scorer on the website:** the API **and** the website, in two terminals (sections 1 and 2). The website only talks to the API when it is started with `VITE_API_URL`.
- **Website without the live scorer:** the website alone. "Try it" then shows saved scores for the four example customers.
- **Inspect results or download customer lists:** the dashboard alone (section 3).

Requirements: the conda env `ml_clv` (Python) and Node 22 with npm (installed in the same env). Run every command from the project root unless stated otherwise.

```bash
export PYTHONNOUSERSITE=1
export PATH=/data/home/mitanshub/miniconda3/envs/ml_clv/bin:$PATH
```

### 1. API (FastAPI, port 8000)

```bash
python -m uvicorn serving.app:app --port 8000
```

- Home page with a small scorer form: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Add `--reload` while editing the API code.

### 2. Website (React + Vite, port 5173)

In a second terminal:

```bash
export PATH=/data/home/mitanshub/miniconda3/envs/ml_clv/bin:$PATH
cd site
npm ci                                   # first time only
VITE_API_URL=http://localhost:8000 npm run dev
```

`VITE_API_URL=http://localhost:8000` tells the website where the API lives; `npm run dev` starts the website. Open http://localhost:5173. In the "Try it" section the status line should say "Live API online".

- Without the API running, the page still works but only shows saved scores for the four examples (editing the numbers gives a "not reachable" message).
- Without `VITE_API_URL`, the website never contacts the API and says "Live API not configured", even if the API is running.
- If you start the API after the website, refresh the page; the check happens when the page loads.

The site must run on port 5173, because the API only accepts requests from there (and from 8501).

### 3. Dashboard (Streamlit, port 8501)

```bash
python -m streamlit run dashboard/app.py
```

It reads only saved files from `outputs/` and `site/public/data/`.

### Rebuild the saved files

```bash
python src/step15_save.py            # model, classifier, metrics.json
python scripts/export_site_data.py   # predictions, comparison, SHAP, site JSON
python scripts/make_site_images.py   # figures for the site
```

### Tests and production build

```bash
python -m pytest tests/test_api.py -q      # API tests
cd site && npm run build                   # type-check + production build into site/dist
```
