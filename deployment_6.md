# Deployment (step 19)

the API runs in a Docker container on **Render** (free tier), the website is built and published to **GitHub Pages** by a workflow, and CI checks the packaging on every push. The Streamlit dashboard stays local (it reads saved files and is an internal tool).

```
Browser -> GitHub Pages (static React site)  --VITE_API_URL-->  Render (Docker, FastAPI, model C)
```

Repository: https://github.com/mitanshubhavsar/clv_predictor  
Site (after deployment): https://mitanshubhavsar.github.io/clv_predictor/  
API (after deployment): https://<your-render-service>.onrender.com/docs

## What we did, in short (the steps that worked)

1. **Packed the project for the internet.** Added a Dockerfile (the recipe for the API's container), a short package list for it (`requirements-api.txt`), and two GitHub workflows: `ci.yml` (tests and checks on every push) and `pages.yml` (builds and publishes the website).
2. **Fixed one real bug before it hurt.** A training-only library (optuna) was imported at the top of one file, which would have crashed the API container on start. It is now imported only when training.
3. **Pushed to GitHub** (`git push origin main`). The `ci` job built the Docker image for the first time and tested it.
4. **Turned on GitHub Pages.** The first `pages` run built the site but failed to publish with "404, ensure GitHub Pages has been enabled". Fix: repository **Settings -> Pages -> Source: GitHub Actions**, then **Re-run failed jobs**. The site went live.
5. **Created the API on Render.** New Web Service from the repository, runtime Docker, Dockerfile path `serving/Dockerfile`, build context `.`, free instance, and **Health Check Path `/health`** (found under **Advanced** on the create form). No environment variables are needed.
6. **Connected the site to the API.** Repository **Settings -> Secrets and variables -> Actions -> Variables**: added `VITE_API_URL` = the Render address (no trailing slash), then ran **Actions -> pages -> Run workflow**, because the address is baked into the site when it is built.
7. **Tested the scorer** on the live site: "Try it" shows "waking up", then "Live API online". The free API sleeps when idle, so the first request after a quiet period takes about a minute.

Links: website https://mitanshubhavsar.github.io/clv_predictor/ (also at the top of `README.md`). The API address is the one shown in the Render dashboard.

## 1. What was prepared in the repository (done)

| File | What it is for |
|---|---|
| `requirements-api.txt` | The lean list for the serving image: fastapi, uvicorn, pydantic, python-multipart, pandas, numpy, scikit-learn, xgboost, joblib. Versions are the ones the model was trained and saved with, so the pickles load unchanged. No shap, optuna, streamlit or matplotlib. |
| `serving/Dockerfile` | python 3.11-slim, installs `requirements-api.txt`, copies `src/`, `serving/` and the four saved files in `outputs/` (model, classifier, spec, metrics). Listens on `$PORT` (Render sets it), 8000 locally. Health check on `/health`. |
| `.dockerignore` | Keeps data, notebooks, site, dashboard, tests and docs out of the image. |
| `requirements-dev.txt` | The API requirements plus pytest, httpx and ruff, for CI. |
| `pyproject.toml` | Ruff and pytest settings (same as the old project). |
| `.github/workflows/ci.yml` | On every push: ruff on `serving` and `tests`, pytest, the API-only import check, docker build, and a smoke test of the container (`/health`, `/model-info`, `/score` must return about 970.35 GBP and "Protect" for recency 20, frequency 6, monetary 2500, IPI 45). |
| `.github/workflows/pages.yml` | On a push that touches `site/`: `npm ci`, type-check and build, publish `site/dist` to GitHub Pages. Sets `VITE_BASE=/<repository name>/` automatically and reads the repository variable `VITE_API_URL`. |
| `serving/app.py` (edited) | CORS now also allows `https://mitanshubhavsar.github.io` (origin only, no repository path), and extra origins can be added with the environment variable `CORS_ORIGINS` (comma separated) without a code change. |
| `src/tuning.py` (edited) | `optuna` is now imported inside `tune()`. See "Problem found" below. |

### Problem found while preparing (and fixed)
The API-only import check (installing only `requirements-api.txt` into a fresh venv and importing `serving.app`) failed: `serving/app.py` imports `step7e_ratio`, which imports `step7a_ridge_log`, which imports `tuning`, which imported `optuna` at the top of the file. The container would have crashed on start. The import is now lazy, so training behaves the same and the serving image does not need optuna. This is exactly the class of bug the check exists for.

## 2. Checks already run on this machine

| Check | Result |
|---|---|
| `pytest tests/test_api.py` | 7 passed |
| `ruff check serving tests` | clean (see "Known gaps" for the training code) |
| Fresh venv with only `requirements-api.txt`, `import serving.app` | imports cleanly |
| That venv, `uvicorn serving.app:app`, then `/health`, `/score`, `/model-info` | health ok; recency 20, frequency 6, monetary 2500, IPI 45 gave 970.35 GBP, multiplier 0.3881, probability 0.7654, Protect (same as the offline model) |
| `site`: `VITE_BASE=/clv_predictor/ npm run build` | passes; assets are under `/clv_predictor/assets/` |
| Both workflow files | valid YAML |
| **`docker build`** | **NOT run: this user has no permission on the Docker socket here.** The first real Docker build will be the CI job on GitHub. |

## 3. Steps to deploy (you do these; each is outward-facing)

### 3.1 Commit and push
The tracked `__pycache__` files show up as modified; untrack them first (they are already in `.gitignore`):
```bash
git rm -r --cached serving/__pycache__ tests/__pycache__ src/__pycache__ 2>/dev/null
git add .
git status            # read it: nothing large or unexpected
git commit -m "deployment: Dockerfile, CI, Pages workflow, CORS for GitHub Pages"
git push origin main
```
Then open the **Actions** tab. `ci` should go green. If the docker build or the smoke test fails, read the "docker logs api" group in the log; fix, push again. Do not set up Render until `ci` is green.

### 3.2 API on Render
1. render.com -> **New** -> **Web Service** -> connect the GitHub repository `clv_predictor`, branch `main`.
2. Settings:
   - Language / runtime: **Docker**
   - Dockerfile path: `serving/Dockerfile`
   - Docker build context directory: `.` (the repository root)
   - Instance type: **Free**
   - Health check path: `/health`
   - Environment variables: none needed
3. Deploy. The first build takes a few minutes. Render gives a URL like `https://clv-predictor-xxxx.onrender.com`.
4. Check in a browser: `<url>/health` shows `{"status":"ok"}`, `<url>/docs` shows the interactive docs, and the home page `<url>/` has the small scorer form.

The free tier sleeps when idle: the first request after a quiet period can take about a minute. The site shows a "waking up" message for that reason.

### 3.3 Website on GitHub Pages
1. GitHub repository -> **Settings** -> **Pages** -> Source: **GitHub Actions**.
2. **Settings** -> **Secrets and variables** -> **Actions** -> **Variables** -> **New repository variable**: name `VITE_API_URL`, value the Render URL, **no trailing slash** (for example `https://clv-predictor-xxxx.onrender.com`).
3. **Actions** -> **pages** -> **Run workflow** (the variable is baked in at build time, so the site must be rebuilt after you set it).
4. Open https://mitanshubhavsar.github.io/clv_predictor/ . In "Try it" the status line should say "Live API online" (after the API wakes up).

## 4. If something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| Site is a blank page, assets give 404 | Wrong base path. The workflow sets `VITE_BASE=/<repository name>/`; check the repository is really named `clv_predictor`, or set `VITE_BASE` explicitly in `pages.yml`. |
| "Try it" says "Live API not configured" | `VITE_API_URL` is not set, or the site was not rebuilt after setting it (run the pages workflow again). |
| "Live API not reachable" | The API is asleep (wait about a minute and reload), or the URL has a trailing slash or a typo, or CORS blocks the site. For a different site address, set `CORS_ORIGINS=https://other-origin` in Render's environment variables. |
| Container crashes on start | Run the API-only import check locally (section 2). Look at the Render logs for the first traceback; a missing package means it belongs in `requirements-api.txt`. |
| CI pytest fails on a fresh checkout | The tests read `data/ledger.csv`, so it must be in the repository (it is today). |
| Render build is very slow or runs out of memory | The image is lean (no shap/optuna); the free instance is small. Check that `.dockerignore` is in the repository so `data/` is not sent as build context. |

## 5. When the model or the numbers change
1. Retrain / re-save: `python src/step15_save.py` (model and metrics), then `python scripts/export_site_data.py`, `scripts/export_heuristics.py`, `scripts/export_targeting.py`, `scripts/export_train_examples.py`, `scripts/make_site_images.py`.
2. Update the smoke-test expectation in `ci.yml` (the 970.35 GBP check) if the model changed.
3. Commit `outputs/` (the four files the image needs) and `site/public/`, push. CI rebuilds the image; the Pages workflow rebuilds the site. Render redeploys automatically on a push to `main`.

## 6. Known gaps (decisions for you)
- **Lint scope.** `ruff check .` over the whole repository reports 49 style findings in the training scripts (semicolons, lambdas, import order). CI lints only `serving` and `tests`, the code that ships. Cleaning the training code is possible but is a separate, larger edit.
- **`data/` is in the repository** (`ledger.csv` 81 MB, `raw.csv` 91 MB, `purchases.csv` 74 MB, the Excel file 44 MB). It works, and CI needs `ledger.csv`, but GitHub warns above 50 MB per file and rejects 100 MB. If the files ever grow, use Git LFS or have CI build a smaller sample.
- **Docker image not built locally** (no socket permission). First build happens in CI.
- **No Render blueprint (`render.yaml`).** The service is created in the Render web interface, as before. A `render.yaml` could make it reproducible; not added because it could not be tested here.
- **README** still has no live links; add them after the first successful deployment.
- **The API has no authentication or rate limit** (the same as the old project). Anyone with the URL can call `/score`. The upload endpoint is limited to 5 MB and 50,000 rows.
