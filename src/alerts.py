"""
ModelWatch - Uyarı Sistemi (Alert System) Modülü
Data drift ve model izleme sonuçlarını eşik değerlerle karşılaştırarak alarm üretir.
"""
from datetime import datetime, timezone


def generate_alerts(
    drift_result: dict,
    monitoring_result: dict | None = None,
    drift_share_warning: float = 0.2,
    drift_share_critical: float = 0.5,
    min_f1_threshold: float = 0.90,
    min_accuracy_threshold: float = 0.90,
) -> dict:
    """
    Drift ve model izleme çıktılarına göre seviyelendirilmiş uyarı listesi döner.
    """
    alerts = []

    drift_share = drift_result.get("drift_share")
    dataset_drift = bool(drift_result.get("dataset_drift", False))
    column_drifts = drift_result.get("column_drifts") or {}
    drifted_cols = [
        col for col, info in column_drifts.items() if info.get("drift_detected")
    ]

    # 1. Veri kayması oranı uyarıları
    if drift_share is not None:
        if dataset_drift or drift_share >= drift_share_critical:
            alerts.append({
                "severity": "CRITICAL",
                "code": "DATASET_DRIFT_CRITICAL",
                "message": (
                    f"Genel veri kayması kritik seviyede: drift_share={drift_share:.4f} "
                    f"(eşik: {drift_share_critical:.2f}). Kayan sütunlar: {drifted_cols}"
                ),
            })
        elif drift_share >= drift_share_warning:
            alerts.append({
                "severity": "WARNING",
                "code": "FEATURE_DRIFT_WARNING",
                "message": (
                    f"Özelliklerde kısmi kayma tespit edildi: drift_share={drift_share:.4f} "
                    f"(uyarı eşiği: {drift_share_warning:.2f}). Kayan sütunlar: {drifted_cols}"
                ),
            })

    # 2. Model performans metrikleri uyarıları
    if isinstance(monitoring_result, dict):
        metrics = monitoring_result.get("metrics") or {}
        f1_val = metrics.get("f1")
        acc_val = metrics.get("accuracy")

        if f1_val is not None and f1_val < min_f1_threshold:
            alerts.append({
                "severity": "CRITICAL",
                "code": "MODEL_F1_DROP",
                "message": (
                    f"Model F1 skoru eşik altında: f1={f1_val:.4f} "
                    f"(minimum eşik: {min_f1_threshold:.2f})"
                ),
            })

        if acc_val is not None and acc_val < min_accuracy_threshold:
            alerts.append({
                "severity": "WARNING",
                "code": "MODEL_ACCURACY_DROP",
                "message": (
                    f"Model accuracy skoru eşik altında: accuracy={acc_val:.4f} "
                    f"(minimum eşik: {min_accuracy_threshold:.2f})"
                ),
            })

    highest_severity = "OK"
    if any(a["severity"] == "CRITICAL" for a in alerts):
        highest_severity = "CRITICAL"
    elif any(a["severity"] == "WARNING" for a in alerts):
        highest_severity = "WARNING"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_tag": drift_result.get("report_tag", "run"),
        "status": highest_severity,
        "alert_count": len(alerts),
        "drifted_columns": drifted_cols,
        "alerts": alerts,
    }
