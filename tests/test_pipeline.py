"""
ModelWatch - Otomatik Test Paketi (pytest)
Veri simülasyonu, drift analizi, model izleme, alarm kuralları, SQLite geçmişi ve FastAPI uçlarını test eder.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.alerts import generate_alerts
from src.data_loader import (
    FEATURE_NAMES,
    generate_production_data,
    generate_training_data,
)
from src.drift import run_drift_analysis
from src.monitor import evaluate_model_monitoring
from src.storage import get_monitoring_history, save_monitoring_run


@pytest.fixture()
def isolated_env(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    history_db = str(tmp_path / "history.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MODELWATCH_HISTORY_DB", history_db)
    api_main._model_cache.clear()
    return {
        "tmp_path": tmp_path,
        "tracking_uri": tracking_uri,
        "history_db": history_db,
    }


def test_data_loader_schema_and_drift():
    ref_df = generate_training_data(n_samples=500, random_state=42)
    clean_df = generate_production_data(n_samples=200, drift=False, random_state=101)
    drifted_df = generate_production_data(n_samples=200, drift=True, random_state=101)

    assert len(ref_df) == 500
    assert len(clean_df) == 200
    assert "memnun" in ref_df.columns
    for col in FEATURE_NAMES:
        assert col in ref_df.columns
        assert col in clean_df.columns

    assert drifted_df["gelir"].mean() > clean_df["gelir"].mean()
    assert drifted_df["aylik_ucret"].mean() > clean_df["aylik_ucret"].mean()


def test_drift_detection_no_drift_vs_drifted(tmp_path):
    ref_df = generate_training_data(n_samples=2000, random_state=42)
    clean_df = generate_production_data(n_samples=800, drift=False, random_state=101)
    drifted_df = generate_production_data(n_samples=800, drift=True, random_state=101)

    res_clean = run_drift_analysis(
        ref_df,
        clean_df,
        output_dir=str(tmp_path),
        save_html=False,
        report_tag="test_clean",
    )
    res_drift = run_drift_analysis(
        ref_df,
        drifted_df,
        output_dir=str(tmp_path),
        save_html=False,
        report_tag="test_drift",
    )

    assert res_clean["total_columns"] == 8
    assert res_clean["dataset_drift"] is False
    assert res_clean["drift_share"] < 0.5

    assert res_drift["column_drifts"]["gelir"]["drift_detected"] is True
    assert res_drift["column_drifts"]["aylik_ucret"]["drift_detected"] is True
    assert res_drift["drift_share"] > res_clean["drift_share"]


def test_monitor_labeled_and_unlabeled(isolated_env):
    from src.train import train_and_register_model

    train_and_register_model(
        tracking_uri=isolated_env["tracking_uri"],
        n_samples=600,
        n_estimators=30,
        max_depth=6,
    )

    labeled_df = generate_production_data(n_samples=120, drift=False, random_state=101)
    mon_labeled = evaluate_model_monitoring(
        labeled_df,
        tracking_uri=isolated_env["tracking_uri"],
        run_tag="labeled_check",
    )
    assert mon_labeled["labeled"] is True
    assert mon_labeled["metrics"] is not None
    assert 0.0 <= mon_labeled["positive_rate"] <= 1.0

    unlabeled_df = labeled_df.drop(columns=["memnun"])
    mon_unlabeled = evaluate_model_monitoring(
        unlabeled_df,
        tracking_uri=isolated_env["tracking_uri"],
        run_tag="unlabeled_check",
    )
    assert mon_unlabeled["labeled"] is False
    assert mon_unlabeled["metrics"] is None


def test_alert_severity_levels():
    ok_alert = generate_alerts(
        drift_result={"report_tag": "ok_run", "drift_share": 0.0, "column_drifts": {}},
        monitoring_result={"metrics": {"f1": 0.95, "accuracy": 0.95}},
        baseline_metrics={"val_f1": 0.95, "val_accuracy": 0.95},
    )
    assert ok_alert["status"] == "OK"
    assert ok_alert["alert_count"] == 0

    warn_alert = generate_alerts(
        drift_result={
            "report_tag": "warn_run",
            "drift_share": 0.25,
            "column_drifts": {"gelir": {"drift_detected": True}},
        },
        monitoring_result={"metrics": {"f1": 0.94, "accuracy": 0.94}},
        baseline_metrics={"val_f1": 0.95, "val_accuracy": 0.95},
    )
    assert warn_alert["status"] == "WARNING"
    assert warn_alert["alert_count"] == 1

    crit_alert = generate_alerts(
        drift_result={
            "report_tag": "crit_run",
            "drift_share": 0.625,
            "column_drifts": {"gelir": {"drift_detected": True}},
        },
        monitoring_result={"metrics": {"f1": 0.60, "accuracy": 0.62}},
        baseline_metrics={"val_f1": 0.95, "val_accuracy": 0.95},
    )
    assert crit_alert["status"] == "CRITICAL"
    assert crit_alert["alert_count"] >= 2

    unlabeled_shift_alert = generate_alerts(
        drift_result={
            "report_tag": "unlabeled_run",
            "drift_share": 0.0,
            "column_drifts": {},
        },
        monitoring_result={"metrics": None, "positive_rate": 0.85},
        baseline_positive_rate=0.50,
    )
    assert unlabeled_shift_alert["status"] == "WARNING"
    assert any(
        a["code"] == "PREDICTION_DISTRIBUTION_SHIFT"
        for a in unlabeled_shift_alert["alerts"]
    )


def test_storage_roundtrip(tmp_path):
    db_file = str(tmp_path / "test_history.db")
    run_id = save_monitoring_run(
        drift_result={
            "timestamp_utc": "2026-11-20T12:00:00+00:00",
            "report_tag": "unit_test",
            "dataset_drift": False,
            "drift_share": 0.25,
            "column_drifts": {"gelir": {"drift_detected": True}},
        },
        monitoring_result={
            "n_samples": 250,
            "positive_rate": 0.51,
            "metrics": {
                "accuracy": 0.94,
                "f1": 0.94,
                "precision": 0.93,
                "recall": 0.95,
            },
        },
        alert_result={
            "status": "WARNING",
            "alert_count": 1,
            "alerts": [
                {
                    "severity": "WARNING",
                    "code": "FEATURE_DRIFT_WARNING",
                    "message": "test",
                }
            ],
        },
        db_path=db_file,
    )
    assert run_id >= 1
    history = get_monitoring_history(limit=5, db_path=db_file)
    assert len(history) == 1
    assert history[0]["report_tag"] == "unit_test"
    assert history[0]["alert_status"] == "WARNING"


def test_fastapi_endpoints_and_negative_cases(isolated_env, monkeypatch):
    from src.train import train_and_register_model

    # Model yokken /predict 503 donmeli
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{isolated_env['tmp_path'] / 'empty.db'}"
    )
    client = TestClient(api_main.app)

    missing_pred = client.post(
        "/predict",
        json={
            "yas": 0.4,
            "gelir": 1.2,
            "abonelik_suresi": -0.3,
            "aylik_ucret": 0.8,
            "destek_talebi_sayisi": -0.5,
            "son_etkilesim_gun": 0.1,
            "kullanilan_hizmet": 0.6,
            "memnuniyet_skoru": 1.1,
        },
    )
    assert missing_pred.status_code == 503

    # Modeli izole ortama kaydet ve mutlu/dis veri akislarini test et
    monkeypatch.setenv("MLFLOW_TRACKING_URI", isolated_env["tracking_uri"])
    train_and_register_model(
        tracking_uri=isolated_env["tracking_uri"],
        n_samples=600,
        n_estimators=30,
        max_depth=6,
    )

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["model_ready"] is True

    pred_resp = client.post(
        "/predict",
        json={
            "yas": 0.4,
            "gelir": 1.2,
            "abonelik_suresi": -0.3,
            "aylik_ucret": 0.8,
            "destek_talebi_sayisi": -0.5,
            "son_etkilesim_gun": 0.1,
            "kullanilan_hizmet": 0.6,
            "memnuniyet_skoru": 1.1,
        },
    )
    assert pred_resp.status_code == 200
    assert pred_resp.json()["prediction"] in (0, 1)

    empty_records_resp = client.post("/monitor/run", json={"records": []})
    assert empty_records_resp.status_code == 422

    live_records = generate_production_data(
        n_samples=120, drift=True, random_state=101
    ).to_dict(orient="records")
    live_resp = client.post(
        "/monitor/run",
        json={"records": live_records, "save_html": False, "report_tag": "live_test"},
    )
    assert live_resp.status_code == 200
    body = live_resp.json()
    assert "run_id" in body

    hist_resp = client.get("/monitor/history?limit=3")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["count"] >= 1
