"""
ModelWatch - Otomatik Test Paketi (pytest)
Veri simülasyonu, drift analizi, alarm kuralları, SQLite geçmişi ve FastAPI uçlarını test eder.
"""
from fastapi.testclient import TestClient

from api.main import app
from src.alerts import generate_alerts
from src.data_loader import FEATURE_NAMES, generate_production_data, generate_training_data
from src.drift import run_drift_analysis
from src.storage import get_monitoring_history, save_monitoring_run

client = TestClient(app)


def test_data_loader_schema_and_drift():
    ref_df = generate_training_data(n_samples=500, random_state=42)
    clean_df = generate_production_data(n_samples=200, drift=False, random_state=7)
    drifted_df = generate_production_data(n_samples=200, drift=True, random_state=7)

    assert len(ref_df) == 500
    assert len(clean_df) == 200
    assert "memnun" in ref_df.columns
    for col in FEATURE_NAMES:
        assert col in ref_df.columns
        assert col in clean_df.columns

    assert drifted_df["gelir"].mean() > clean_df["gelir"].mean()
    assert drifted_df["aylik_ucret"].mean() > clean_df["aylik_ucret"].mean()


def test_drift_detection_no_drift_vs_drifted():
    ref_df = generate_training_data(n_samples=5000, random_state=42)
    clean_df = generate_production_data(n_samples=1000, drift=False, random_state=7)
    drifted_df = generate_production_data(n_samples=1000, drift=True, random_state=7)

    res_clean = run_drift_analysis(ref_df, clean_df, save_html=False, report_tag="test_clean")
    res_drift = run_drift_analysis(ref_df, drifted_df, save_html=False, report_tag="test_drift")

    assert res_clean["total_columns"] == 8
    assert res_clean["drift_share"] == 0.0
    assert res_clean["dataset_drift"] is False

    assert res_drift["column_drifts"]["gelir"]["drift_detected"] is True
    assert res_drift["column_drifts"]["aylik_ucret"]["drift_detected"] is True
    assert res_drift["drift_share"] > res_clean["drift_share"]


def test_alert_severity_levels():
    ok_alert = generate_alerts(
        drift_result={"report_tag": "ok_run", "dataset_drift": False, "drift_share": 0.0, "column_drifts": {}},
        monitoring_result={"metrics": {"f1": 0.95, "accuracy": 0.95}},
    )
    assert ok_alert["status"] == "OK"
    assert ok_alert["alert_count"] == 0

    warn_alert = generate_alerts(
        drift_result={
            "report_tag": "warn_run",
            "dataset_drift": False,
            "drift_share": 0.25,
            "column_drifts": {"gelir": {"drift_detected": True}},
        },
        monitoring_result={"metrics": {"f1": 0.94, "accuracy": 0.94}},
    )
    assert warn_alert["status"] == "WARNING"
    assert warn_alert["alert_count"] == 1

    crit_alert = generate_alerts(
        drift_result={
            "report_tag": "crit_run",
            "dataset_drift": True,
            "drift_share": 0.625,
            "column_drifts": {"gelir": {"drift_detected": True}},
        },
        monitoring_result={"metrics": {"f1": 0.72, "accuracy": 0.75}},
    )
    assert crit_alert["status"] == "CRITICAL"
    assert crit_alert["alert_count"] >= 2


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
            "metrics": {"accuracy": 0.94, "f1": 0.94, "precision": 0.93, "recall": 0.95},
        },
        alert_result={
            "status": "WARNING",
            "alert_count": 1,
            "alerts": [{"severity": "WARNING", "code": "FEATURE_DRIFT_WARNING", "message": "test"}],
        },
        db_path=db_file,
    )
    assert run_id >= 1
    history = get_monitoring_history(limit=5, db_path=db_file)
    assert len(history) == 1
    assert history[0]["report_tag"] == "unit_test"
    assert history[0]["alert_status"] == "WARNING"


def test_fastapi_endpoints():
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    payload = {
        "yas": 0.4,
        "gelir": 1.2,
        "abonelik_suresi": -0.3,
        "aylik_ucret": 0.8,
        "destek_talebi_sayisi": -0.5,
        "son_etkilesim_gun": 0.1,
        "kullanilan_hizmet": 0.6,
        "memnuniyet_skoru": 1.1,
    }
    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200
    assert pred_resp.json()["prediction"] in (0, 1)

    mon_resp = client.post("/monitor/run", json={"n_samples": 200, "drift": True, "save_html": False})
    assert mon_resp.status_code == 200
    body = mon_resp.json()
    assert "run_id" in body
    assert body["alerts"]["status"] == "WARNING"

    hist_resp = client.get("/monitor/history?limit=3")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["count"] >= 1
