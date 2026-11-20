"""
ModelWatch - FastAPI İzleme ve Tahmin Servisi
Veri kayması analizi, model izleme, uyarı üretimi, geçmiş kaydı ve tahmin uçları sunar.
"""

import json
import logging
import math
import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.alerts import generate_alerts
from src.data_loader import generate_production_data, generate_training_data
from src.drift import run_drift_analysis
from src.integrations import emit_alert_notifications
from src.monitor import evaluate_model_monitoring
from src.storage import get_monitoring_history, save_monitoring_run

logger = logging.getLogger("modelwatch.api")

FEATURE_COLUMNS: list[str] = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "aylik_ucret",
    "destek_talebi_sayisi",
    "son_etkilesim_gun",
    "kullanilan_hizmet",
    "memnuniyet_skoru",
]
MAX_BATCH_RECORDS = 20000


def _model_name() -> str:
    return os.getenv("MODELWATCH_MODEL_NAME", "ModelWatch-CustomerSatisfaction")


def _model_alias() -> str:
    return os.getenv("MODELWATCH_MODEL_ALIAS", "staging")


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _history_db_path() -> str:
    return os.getenv("MODELWATCH_HISTORY_DB", "data/modelwatch_history.db")


def _expected_api_key() -> str:
    return os.getenv("MODELWATCH_API_KEY", "").strip()


def _app_env() -> str:
    return os.getenv("MODELWATCH_ENV", os.getenv("ENV", "development")).strip().lower()


def _verify_monitor_api_key(x_api_key: str | None) -> None:
    expected = _expected_api_key()
    if not expected:
        if _app_env() in {"production", "prod"}:
            logger.error("MODELWATCH_API_KEY production ortamında tanımlı değil.")
            raise HTTPException(
                status_code=503,
                detail="Sunucu güvenlik yapılandırması eksik: MODELWATCH_API_KEY tanımlı değil.",
            )
        return

    if not x_api_key or x_api_key != expected:
        logger.warning("Yetkisiz /monitor/run erişim denemesi reddedildi.")
        raise HTTPException(
            status_code=401,
            detail="Geçersiz veya eksik X-API-Key başlığı.",
        )


def _validate_finite_float(value: float, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"'{field_name}' geçerli ve sonlu bir sayı olmalıdır.")
    return float(value)


app = FastAPI(title="ModelWatch API", version="1.5.0")


def _sanitize_for_json(obj):
    from fastapi.encoders import jsonable_encoder

    encoded = jsonable_encoder(obj, custom_encoder={Exception: str})

    def _clean(val):
        if isinstance(val, float) and not math.isfinite(val):
            return str(val)
        if isinstance(val, dict):
            return {k: _clean(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_clean(item) for item in val]
        return val

    return _clean(encoded)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    sanitized_errors = _sanitize_for_json(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": sanitized_errors},
    )


_model_cache: dict[tuple[str, str, str], object] = {}
_baseline_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, dict, str | None]] = {}


def _resolve_model_version_info() -> tuple[str, str | None]:
    name = _model_name()
    alias = _model_alias()
    uri = _tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        client = MlflowClient(tracking_uri=uri)
        mv = client.get_model_version_by_alias(name=name, alias=alias)
        return f"{alias}:v{mv.version}", str(mv.version)
    except (MlflowException, OSError, RuntimeError, ValueError):
        return alias, None


def _get_model():
    name = _model_name()
    alias = _model_alias()
    uri = _tracking_uri()
    mlflow.set_tracking_uri(uri)

    version_tag, _ = _resolve_model_version_info()
    cache_key = (uri, name, version_tag)

    if cache_key not in _model_cache:
        try:
            loaded = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{name}@{alias}' yüklenemedi: {exc}",
            ) from exc
        _model_cache.clear()
        _model_cache[cache_key] = loaded

    return _model_cache[cache_key]


def _load_reference_and_baseline() -> tuple[pd.DataFrame, dict, str | None]:
    name = _model_name()
    alias = _model_alias()
    uri = _tracking_uri()
    mlflow.set_tracking_uri(uri)

    version_tag = alias
    model_version: str | None = None
    run_id: str | None = None
    try:
        client = MlflowClient(tracking_uri=uri)
        mv = client.get_model_version_by_alias(name=name, alias=alias)
        version_tag = f"{alias}:v{mv.version}"
        model_version = str(mv.version)
        run_id = mv.run_id
    except (MlflowException, OSError, RuntimeError, ValueError):
        pass

    cache_key = (uri, name, version_tag)
    if cache_key in _baseline_cache:
        return _baseline_cache[cache_key]

    ref_df: pd.DataFrame | None = None
    baseline_metrics: dict = {}

    if run_id:
        try:
            client = MlflowClient(tracking_uri=uri)
            run_data = client.get_run(run_id).data
            for k in ("val_f1", "val_accuracy", "baseline_positive_rate"):
                if k in run_data.metrics:
                    val = float(run_data.metrics[k])
                    if math.isfinite(val):
                        baseline_metrics[k] = val

            local_art_dir = client.download_artifacts(run_id, "reference")
            art_path = Path(local_art_dir)
            csv_file = art_path / "reference_data.csv"
            if csv_file.exists():
                ref_df = pd.read_csv(csv_file)
            metrics_file = art_path / "baseline_metrics.json"
            if metrics_file.exists():
                loaded_m = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(loaded_m, dict):
                    for mk, mv_val in loaded_m.items():
                        if (
                            isinstance(mv_val, (int, float))
                            and not isinstance(mv_val, bool)
                            and math.isfinite(float(mv_val))
                        ):
                            baseline_metrics[mk] = float(mv_val)
        except (MlflowException, OSError, RuntimeError, ValueError):
            pass

    if ref_df is None or ref_df.empty:
        ref_df = generate_training_data(n_samples=5000, random_state=42)

    if "baseline_positive_rate" not in baseline_metrics and "memnun" in ref_df.columns:
        baseline_metrics["baseline_positive_rate"] = round(
            float(ref_df["memnun"].mean()), 4
        )

    _baseline_cache.clear()
    _baseline_cache[cache_key] = (ref_df, baseline_metrics, model_version)
    return ref_df, baseline_metrics, model_version


class CustomerFeatures(BaseModel):
    yas: float
    gelir: float
    abonelik_suresi: float
    aylik_ucret: float
    destek_talebi_sayisi: float
    son_etkilesim_gun: float
    kullanilan_hizmet: float
    memnuniyet_skoru: float

    @field_validator(
        "yas",
        "gelir",
        "abonelik_suresi",
        "aylik_ucret",
        "destek_talebi_sayisi",
        "son_etkilesim_gun",
        "kullanilan_hizmet",
        "memnuniyet_skoru",
    )
    @classmethod
    def check_finite_features(cls, v: float, info) -> float:
        return _validate_finite_float(v, info.field_name)


class CustomerMonitoringRecord(CustomerFeatures):
    memnun: int | None = Field(default=None, ge=0, le=1)


class MonitoringRequest(BaseModel):
    n_samples: int = Field(default=1000, ge=50, le=MAX_BATCH_RECORDS)
    drift: bool = False
    random_state: int = 101
    save_html: bool = False
    records: list[CustomerMonitoringRecord] | None = Field(
        default=None,
        max_length=MAX_BATCH_RECORDS,
        description="Verilirse simülasyon yerine bu gerçek/dış veri kayıtları izlenir.",
    )
    report_tag: str | None = Field(default=None, max_length=120)
    baseline_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("baseline_positive_rate")
    @classmethod
    def check_finite_pos_rate(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return _validate_finite_float(v, "baseline_positive_rate")

    @model_validator(mode="after")
    def check_non_empty_records(self) -> "MonitoringRequest":
        if self.records is not None and len(self.records) == 0:
            raise ValueError("'records' listesi boş olamaz.")
        return self


def _build_current_df(req: MonitoringRequest) -> tuple[pd.DataFrame, str]:
    if req.records is not None:
        rows = []
        for item in req.records:
            row = {col: getattr(item, col) for col in FEATURE_COLUMNS}
            if item.memnun is not None:
                row["memnun"] = item.memnun
            rows.append(row)
        df = pd.DataFrame(rows)
        ordered_cols = FEATURE_COLUMNS + (["memnun"] if "memnun" in df.columns else [])
        df = df[ordered_cols]
        tag = req.report_tag or "live_batch"
        return df, tag

    df = generate_production_data(
        n_samples=req.n_samples,
        drift=req.drift,
        random_state=req.random_state,
    )
    ordered_cols = FEATURE_COLUMNS + (["memnun"] if "memnun" in df.columns else [])
    df = df[ordered_cols]
    tag = req.report_tag or ("drifted" if req.drift else "no_drift")
    return df, tag


def _dispatch_notifications_if_needed(alert_res: dict) -> None:
    status = str((alert_res or {}).get("status") or "OK").upper()
    if status in {"WARNING", "CRITICAL"}:
        emit_alert_notifications(alert_res)


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    name = _model_name()
    alias = _model_alias()
    _, model_version = _resolve_model_version_info()
    try:
        _get_model()
        model_ready = True
    except HTTPException:
        model_ready = False

    payload = {
        "status": "ready" if model_ready else "not_ready",
        "model_ready": model_ready,
        "model_name": name,
        "alias": alias,
        "model_version": model_version,
    }
    return JSONResponse(status_code=200 if model_ready else 503, content=payload)


@app.get("/health")
def health():
    return health_ready()


@app.post("/predict")
def predict(payload: CustomerFeatures):
    model = _get_model()
    X = pd.DataFrame(
        [[getattr(payload, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
    )
    preds = model.predict(X)
    pred_val = int(preds[0])
    return {
        "prediction": pred_val,
        "label": "memnun" if pred_val == 1 else "memnun_degil",
    }


@app.post("/monitor/run")
def run_full_monitoring(
    req: MonitoringRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _verify_monitor_api_key(x_api_key)
    ref_df, baseline_metrics, model_version = _load_reference_and_baseline()
    curr_df, tag = _build_current_df(req)

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

    effective_pos_rate = (
        req.baseline_positive_rate
        if req.baseline_positive_rate is not None
        else baseline_metrics.get("baseline_positive_rate")
    )

    alert_res = generate_alerts(
        drift_res,
        mon_res,
        baseline_metrics=baseline_metrics or None,
        baseline_positive_rate=effective_pos_rate,
    )
    run_id = save_monitoring_run(
        drift_res,
        mon_res,
        alert_res,
        db_path=_history_db_path(),
        model_name=_model_name(),
        model_alias=_model_alias(),
        model_version=model_version,
        baseline_metrics=baseline_metrics or None,
    )
    background_tasks.add_task(_dispatch_notifications_if_needed, alert_res)

    return {
        "run_id": run_id,
        "model_version": model_version,
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
