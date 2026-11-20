"""
ModelWatch - Veri Kayması (Data Drift) Analiz Modülü
Referans veri ile üretim verisi arasındaki dağılım değişimini Evidently 0.7.23 ile ölçer.
"""
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    from evidently import Report
    from evidently.presets import DataDriftPreset


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = "reports",
    save_html: bool = True,
    report_tag: str = "run",
) -> dict:
    """
    Referans ve güncel veri setleri arasında data drift raporu üretir.
    Hedef sütun ('memnun') varsa özellik karşılaştırmasından hariç tutulur.
    """
    drop_cols = [
        col
        for col in ["memnun"]
        if col in reference_df.columns or col in current_df.columns
    ]
    ref_features = reference_df.drop(
        columns=[c for c in drop_cols if c in reference_df.columns]
    )
    curr_features = current_df.drop(
        columns=[c for c in drop_cols if c in current_df.columns]
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_features, current_data=curr_features)

    snap_dict = snapshot.dict() if hasattr(snapshot, "dict") else {}
    metrics_list = snap_dict.get("metrics", [])

    drift_share = None
    drifted_columns_count = None
    dataset_drift = None
    drift_share_threshold = 0.5
    column_drifts = {}

    for metric in metrics_list:
        if not isinstance(metric, dict):
            continue

        metric_name = str(metric.get("metric_name", ""))
        config = metric.get("config", {}) or {}
        value = metric.get("value")

        if "DriftedColumnsCount" in metric_name and isinstance(value, dict):
            drift_share_threshold = float(config.get("drift_share", 0.5))
            if value.get("count") is not None:
                drifted_columns_count = int(float(value["count"]))
            if value.get("share") is not None:
                drift_share = float(value["share"])
            if drift_share is not None:
                dataset_drift = bool(drift_share >= drift_share_threshold)

        elif "ValueDrift" in metric_name and isinstance(config, dict):
            col_name = config.get("column")
            threshold = float(config.get("threshold", 0.1))
            stattest_name = str(config.get("method", ""))
            if col_name is not None and value is not None:
                score = float(value)
                # p_value testlerinde düşük skor (< threshold), mesafe testlerinde yüksek skor (>= threshold) drift gösterir
                if "p_value" in stattest_name.lower():
                    detected = score < threshold
                else:
                    detected = score >= threshold

                column_drifts[str(col_name)] = {
                    "drift_detected": bool(detected),
                    "drift_score": round(score, 6),
                    "stattest_name": stattest_name,
                }

    if drifted_columns_count is None and column_drifts:
        drifted_columns_count = sum(
            1 for info in column_drifts.values() if info["drift_detected"]
        )
    if drift_share is None and drifted_columns_count is not None and len(ref_features.columns) > 0:
        drift_share = drifted_columns_count / len(ref_features.columns)
    if dataset_drift is None and drift_share is not None:
        dataset_drift = bool(drift_share >= drift_share_threshold)

    html_report_path = None
    if save_html and hasattr(snapshot, "save_html"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_html = str(out_path / f"drift_{report_tag}_{ts}.html")
        snapshot.save_html(target_html)
        html_report_path = target_html

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_tag": report_tag,
        "reference_rows": len(ref_features),
        "current_rows": len(curr_features),
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4) if drift_share is not None else None,
        "drifted_columns_count": drifted_columns_count,
        "total_columns": len(ref_features.columns),
        "column_drifts": column_drifts,
        "html_report_path": html_report_path,
    }
