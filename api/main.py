"""
ModelWatch - FastAPI İzleme ve Tahmin Servisi
Veri kayması analizi, model izleme, uyarı üretimi, geçmiş kaydı ve tahmin uçları sunar.
"""
import os
import sys
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import mlflow

REPO_DIR = "/content/modelwatch"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.data_loader import generate_production_data, generate_training_data
from src.drift import run_drift_analysis
from src.monitor import evaluate_model_monitoring
from src.alerts import generate_alerts
from src.storage import get_monitoring_history, save_monitoring_run

MODEL_NAME = "ModelWatch-CustomerSatisfaction"
MODEL_ALIAS = "staging"
TRACKING_URI = "sqlite:///mlflow.db"

app = FastAPI(title="ModelWatch API", version="1.1.0")

_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        mlflow.set_tracking_uri(TRACKING_URI)
        _cached_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return _cached_model


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
    random_state: int = 7
    save_html: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "working_dir": os.getcwd(),
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
    curr_df = generate_production_data(
        n_samples=req.n_samples,
        drift=req.drift,
        random_state=req.random_state,
    )
    tag = "drifted" if req.drift else "no_drift"

    drift_res = run_drift_analysis(
        reference_df=ref_df,
        current_df=curr_df,
        save_html=req.save_html,
        report_tag=tag,
    )
    mon_res = evaluate_model_monitoring(
        df=curr_df,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        tracking_uri=TRACKING_URI,
        run_tag=tag,
    )
    alert_res = generate_alerts(drift_res, mon_res)
    run_id = save_monitoring_run(drift_res, mon_res, alert_res)

    return {
        "run_id": run_id,
        "drift": drift_res,
        "monitoring": mon_res,
        "alerts": alert_res,
    }


@app.get("/monitor/history")
def monitor_history(limit: int = Query(default=10, ge=1, le=200)):
    items = get_monitoring_history(limit=limit)
    return {
        "count": len(items),
        "items": items,
    }
