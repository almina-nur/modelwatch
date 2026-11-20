# ModelWatch — ML Model Monitoring & Drift Detection Service

ModelWatch is a modular **MLOps and Model Observability** service built with **FastAPI**, **Evidently (`0.7.23`)**, **MLflow**, **scikit-learn**, and **pytest**. It simulates production traffic, detects feature drift using normed Wasserstein distance, evaluates model performance on an MLflow-registered classifier (`staging` alias), generates severity-based alerts (`OK`, `WARNING`, `CRITICAL`), persists run history in SQLite, and exposes inference and monitoring endpoints over HTTP.

---

## Architecture Overview

```text
[Data Loader] ---> [Evidently Drift Engine] ---> [Alert Engine] ---> [SQLite History]
      |                                               ^                      ^
      +---------> [MLflow Model Monitor] -------------+                      |
      |                                                                      |
      +---------> [FastAPI Endpoints: /health, /predict, /monitor/run, /monitor/history]
```

### Project Structure

```text
modelwatch/
├── api/
│   └── main.py              # FastAPI inference & monitoring endpoints
├── src/
│   ├── data_loader.py       # Reference & drifted/clean production data generator
│   ├── drift.py             # Evidently 0.7.23 drift analysis & HTML reporting
│   ├── monitor.py           # MLflow model evaluation & prediction distribution tracking
│   ├── alerts.py            # Severity-based alert evaluation (OK / WARNING / CRITICAL)
│   └── storage.py           # SQLite persistence for monitoring history
├── tests/
│   ├── conftest.py          # Test path configuration
│   └── test_pipeline.py     # Unit & API integration tests
├── .github/
│   └── workflows/
│       └── ci.yml           # Automated GitHub Actions test pipeline
├── Dockerfile
└── requirements.txt
```

---

## API Endpoints

- `GET /health` — Service health check, model name, alias, and working directory.
- `POST /predict` — Runs inference on a single customer feature payload (`yas`, `gelir`, `abonelik_suresi`, `aylik_ucret`, `destek_talebi_sayisi`, `son_etkilesim_gun`, `kullanilan_hizmet`, `memnuniyet_skoru`).
- `POST /monitor/run` — Executes drift analysis, model monitoring, and alert generation for clean or drifted production samples, saving a run record to SQLite.
- `GET /monitor/history?limit=10` — Lists recent monitoring runs with drift share, metrics, and alert status.

---

## Quickstart

### 1. Run Test Suite
```bash
python -m pytest -v
```

### 2. Start FastAPI Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run with Docker
```bash
docker build -t modelwatch-api .
docker run -p 8000:8000 modelwatch-api
```

---

## Resume / CV Highlight

> **ModelWatch — ML Model Monitoring & Drift Detection API** (*Python, FastAPI, MLflow, Evidently, scikit-learn, pytest*)  
> Built a modular model monitoring pipeline tracking feature drift (Wasserstein distance) and classification performance on an MLflow-registered model, generating `OK`/`WARNING`/`CRITICAL` alerts, logging historical runs to SQLite, and serving `/predict` and `/monitor/run` endpoints via FastAPI with automated `pytest` and GitHub Actions CI.
