"""
ModelWatch - FastAPI Izleme ve Tahmin Servisi
Veri kaymasi analizi, model izleme, uyari uretimi, gecmis kaydi ve tahmin uclari sunar.
"""

import os

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

from src.alerts import generate_alerts
from src.data_loader import generate_production_data, generate_training_data
from src.drift import run_drift_analysis
from src.monitor import evaluate_model_monitoring
from src.storage import get_monitoring_history, save_monitoring_run


def _model_name() -> str:
    return os.getenv("MODELWATCH_MODEL_NAME", "ModelWatch-CustomerSatisfaction")


def _model_alias() -> str:
    return os.getenv("MODELWATCH_MODEL_ALIAS", "staging")


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _history_db_path() -> str:
    return os.getenv("MODELWATCH_HISTORY_DB", "data/modelwatch_history.db")


app = FastAPI(title="ModelWatch API", version="1.3.0")

_model_cache: dict[tuple[str, str, str], object] = {}


def _get_model():
    name = _model_name()
    alias = _model_alias()
    uri = _tracking_uri()
    mlflow.set_tracking_uri(uri)

    try:
        client = MlflowClient(tracking_uri=uri)
        mv = client.get_model_version_by_alias(name=name, alias=alias)
        cache_key = (uri, name, f"{alias}:v{mv.version}")
    except (MlflowException, OSError, RuntimeError, ValueError):
        cache_key = (uri, name, alias)

    if cache_key not in _model_cache:
        try:
            loaded = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{name}@{alias}' yuklenemedi: {exc}",
            ) from exc
        _model_cache.clear()
        _model_cache[cache_key] = loaded

    return _model_cache[cache_key]


class CustomerFeatures(BaseModel):
    yas: float
    gelir: float
    abonelik_suresi: float
    aylik_ucret: float
    destek_talebi_sayisi: float
    son_etkilesim_gun: float
    kullanilan_hizmet: float
    memnuniyet_skoru: float


class MonitoringRequest(BaseModel):
    n_samples: int = Field(default=1000, ge=50, le=20000)
    drift: bool = False
    random_state: int = 101
    save_html: bool = False
    records: list[dict] | None = Field(
        default=None,
        description="Verilirse simulasyon yerine bu gercek/dis veri kayitlari izlenir.",
    )
    report_tag: str | None = None
    baseline_positive_rate: float | None = Field(default=0.50, ge=0.0, le=1.0)


@app.get("/health")
def health():
    name = _model_name()
    alias = _model_alias()
    try:
        _get_model()
        model_ready = True
    except HTTPException:
        model_ready = False

    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_name": name,
        "alias": alias,
    }


@app.post("/predict")
def predict(payload: CustomerFeatures):
    model = _get_model()
    X = pd.DataFrame([payload.model_dump()])
    preds = model.predict(X)
    pred_val = int(preds[0])
    return {
        "prediction": pred_val,
        "label": "memnun" if pred_val == 1 else "memnun_degil",
    }


@app.post("/monitor/run")
def run_full_monitoring(req: MonitoringRequest):
    ref_df = generate_training_data(n_samples=5000, random_state=42)

    if req.records is not None:
        if len(req.records) == 0:
            raise HTTPException(status_code=422, detail="'records' listesi bos olamaz.")
        curr_df = pd.DataFrame(req.records)
        tag = req.report_tag or "live_batch"
    else:
        curr_df = generate_production_data(
            n_samples=req.n_samples,
            drift=req.drift,
            random_state=req.random_state,
        )
        tag = req.report_tag or ("drifted" if req.drift else "no_drift")

    try:
        drift_res = run_drift_analysis(
            reference_df=ref_df,
            current_df=curr_df,
            save_html=req.save_html,
            report_tag=tag,
        )
        mon_res = evaluate_model_monitoring(
            df=curr_df,
            model_name=_model_name(),
            alias=_model_alias(),
            tracking_uri=_tracking_uri(),
            run_tag=tag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    alert_res = generate_alerts(
        drift_res,
        mon_res,
        baseline_positive_rate=req.baseline_positive_rate,
    )
    run_id = save_monitoring_run(
        drift_res,
        mon_res,
        alert_res,
        db_path=_history_db_path(),
    )

    return {
        "run_id": run_id,
        "drift": drift_res,
        "monitoring": mon_res,
        "alerts": alert_res,
    }


@app.get("/monitor/history")
def monitor_history(limit: int = Query(default=10, ge=1, le=200)):
    items = get_monitoring_history(limit=limit, db_path=_history_db_path())
    return {
        "count": len(items),
        "items": items,
    }
