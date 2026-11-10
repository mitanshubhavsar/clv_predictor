FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Lean on purpose: no shap/optuna/streamlit/matplotlib. See requirements-api.txt.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Only what serving needs: the code, the saved model, classifier, spec and metrics.
COPY src ./src
COPY serving ./serving
COPY outputs/model_c.joblib outputs/logistic_return.joblib outputs/model_spec.json outputs/metrics.json ./outputs/
ENV PYTHONPATH=/app

HEALTHCHECK --interval=15s --timeout=3s --retries=10 CMD curl -fs localhost:${PORT:-8000}/health || exit 1
EXPOSE 8000
# Render sets $PORT; locally it falls back to 8000.
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
