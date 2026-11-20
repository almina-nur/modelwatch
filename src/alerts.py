"""
ModelWatch - Uyari Sistemi (Alert System) Modulu
Drift, model performansi ve etiketsiz tahmin dagilimi kaymalarina gore seviyelendirilmis alarm uretir.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AlertThresholds:
    drift_share_warning: float = 0.20
    drift_share_critical: float = 0.50
    max_relative_metric_drop: float = 0.05  # Baseline'a gore %5 bagil dusus
    max_positive_rate_shift: float = 0.15


DEFAULT_THRESHOLDS = AlertThresholds()


def _pick_metric(source: dict | None, keys: tuple[str, ...]) -> float | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key in source and source[key] is not None:
            return float(source[key])
    return None


def generate_alerts(
    drift_result: dict,
    monitoring_result: dict | None = None,
    baseline_metrics: dict | None = None,
    baseline_positive_rate: float | None = None,
    thresholds: AlertThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Drift, performans veya tahmin dagilimi kaymalarina gore alarm listesi uretir."""
    alerts: list[dict] = []

    drift_share = (drift_result or {}).get("drift_share")
    column_drifts = (drift_result or {}).get("column_drifts") or {}
    drifted_cols = [
        col for col, info in column_drifts.items() if info.get("drift_detected")
    ]

    # 1. Veri kaymasi orani (tek kaynak: drift_share)
    if drift_share is not None:
        share_val = float(drift_share)
        if share_val >= thresholds.drift_share_critical:
            alerts.append(
                {
                    "severity": "CRITICAL",
                    "code": "DATASET_DRIFT_CRITICAL",
                    "message": (
                        f"Genel veri kaymasi kritik seviyede: drift_share={share_val:.4f} "
                        f"(esik: {thresholds.drift_share_critical:.2f}). Kayan sutunlar: {drifted_cols}"
                    ),
                }
            )
        elif share_val >= thresholds.drift_share_warning:
            alerts.append(
                {
                    "severity": "WARNING",
                    "code": "FEATURE_DRIFT_WARNING",
                    "message": (
                        f"Ozelliklerde kismi kayma tespit edildi: drift_share={share_val:.4f} "
                        f"(uyari esigi: {thresholds.drift_share_warning:.2f}). Kayan sutunlar: {drifted_cols}"
                    ),
                }
            )

    # 2. Model izleme metrikleri veya etiketsiz tahmin kaymasi
    if isinstance(monitoring_result, dict):
        metrics = monitoring_result.get("metrics")

        if isinstance(metrics, dict):
            f1_val = _pick_metric(metrics, ("f1",))
            acc_val = _pick_metric(metrics, ("accuracy",))
            base_f1 = _pick_metric(baseline_metrics, ("val_f1", "f1"))
            base_acc = _pick_metric(baseline_metrics, ("val_accuracy", "accuracy"))

            # Referans metrik yoksa performans alarmi uretilmez
            f1_bad = False
            acc_bad = False

            if f1_val is not None and base_f1 is not None:
                f1_floor = base_f1 * (1.0 - thresholds.max_relative_metric_drop)
                if f1_val < f1_floor:
                    f1_bad = True

            if acc_val is not None and base_acc is not None:
                acc_floor = base_acc * (1.0 - thresholds.max_relative_metric_drop)
                if acc_val < acc_floor:
                    acc_bad = True

            if f1_bad and acc_bad:
                alerts.append(
                    {
                        "severity": "CRITICAL",
                        "code": "MODEL_PERFORMANCE_DEGRADATION",
                        "message": (
                            f"Model F1 ({f1_val:.4f} vs base {base_f1:.4f}) ve "
                            f"accuracy ({acc_val:.4f} vs base {base_acc:.4f}) "
                            "referans esiklerin altina dustu."
                        ),
                    }
                )
            elif f1_bad:
                alerts.append(
                    {
                        "severity": "CRITICAL",
                        "code": "MODEL_F1_DROP",
                        "message": (
                            f"Model F1 skoru referans esigin altinda: "
                            f"f1={f1_val:.4f} (base={base_f1:.4f})."
                        ),
                    }
                )
            elif acc_bad:
                alerts.append(
                    {
                        "severity": "WARNING",
                        "code": "MODEL_ACCURACY_DROP",
                        "message": (
                            f"Model accuracy skoru referans esigin altinda: "
                            f"accuracy={acc_val:.4f} (base={base_acc:.4f})."
                        ),
                    }
                )
        else:
            # Etiket (ground truth) yokken pozitif sinif tahmin orani kaymasini izle
            pos_rate = monitoring_result.get("positive_rate")
            if pos_rate is not None and baseline_positive_rate is not None:
                shift = abs(float(pos_rate) - float(baseline_positive_rate))
                if shift >= thresholds.max_positive_rate_shift:
                    alerts.append(
                        {
                            "severity": "WARNING",
                            "code": "PREDICTION_DISTRIBUTION_SHIFT",
                            "message": (
                                f"Etiketsiz izleme: pozitif tahmin orani kaydi (current={float(pos_rate):.4f}, "
                                f"baseline={float(baseline_positive_rate):.4f}, fark={shift:.4f})."
                            ),
                        }
                    )

    highest_severity = "OK"
    if any(a["severity"] == "CRITICAL" for a in alerts):
        highest_severity = "CRITICAL"
    elif any(a["severity"] == "WARNING" for a in alerts):
        highest_severity = "WARNING"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_tag": (drift_result or {}).get("report_tag", "run"),
        "status": highest_severity,
        "alert_count": len(alerts),
        "drifted_columns": drifted_cols,
        "alerts": alerts,
    }
