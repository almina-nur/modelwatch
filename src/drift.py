"""ModelWatch - Veri Kaymasi (Data Drift) Analiz Modulu."""

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset

MIN_RECOMMENDED_ROWS = 100
DEFAULT_WASSERSTEIN_THRESHOLD = 0.1
DEFAULT_DRIFT_SHARE_THRESHOLD = 0.5


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = "reports",
    save_html: bool = True,
    report_tag: str = "run",
    wasserstein_threshold: float = DEFAULT_WASSERSTEIN_THRESHOLD,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
) -> dict:
    """Referans ve guncel veri setleri arasinda drift analizi yapar."""
    if reference_df.empty or current_df.empty:
        raise ValueError("Referans ve guncel veri setleri bos olamaz.")

    sample_warning = None
    if (
        len(reference_df) < MIN_RECOMMENDED_ROWS
        or len(current_df) < MIN_RECOMMENDED_ROWS
    ):
        sample_warning = (
            f"Kucuk orneklem boyutu (ref={len(reference_df)}, curr={len(current_df)}); "
            f"onerilen alt sinir {MIN_RECOMMENDED_ROWS}."
        )
        warnings.warn(sample_warning, UserWarning, stacklevel=2)

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

    if list(ref_features.columns) != list(curr_features.columns):
        missing_in_curr = set(ref_features.columns) - set(curr_features.columns)
        if missing_in_curr:
            raise ValueError(
                f"Guncel veride eksik ozellik sutunlari: {sorted(missing_in_curr)}"
            )
        curr_features = curr_features[ref_features.columns]

    numeric_cols = [
        col
        for col in ref_features.columns
        if pd.api.types.is_numeric_dtype(ref_features[col])
    ]
    wasserstein_metrics = [
        ValueDrift(column=col, method="wasserstein", threshold=wasserstein_threshold)
        for col in numeric_cols
    ]
    report = Report(
        metrics=[
            DataDriftPreset(),
            DriftedColumnsCount(drift_share=drift_share_threshold),
            *wasserstein_metrics,
        ]
    )
    snapshot = report.run(reference_data=ref_features, current_data=curr_features)
    snap_dict = snapshot.dict() if hasattr(snapshot, "dict") else {}
    metrics_list = snap_dict.get("metrics", [])
    column_drifts: dict[str, dict] = {}

    for metric in metrics_list:
        if not isinstance(metric, dict):
            continue
        metric_name = str(metric.get("metric_name", ""))
        config = metric.get("config", {}) or {}
        value = metric.get("value")
        if metric_name.startswith("ValueDrift") and isinstance(config, dict):
            col_name = config.get("column")
            threshold = float(config.get("threshold", wasserstein_threshold))
            stattest_name = str(config.get("method", "wasserstein"))
            if col_name is not None and value is not None:
                score = float(value)
                is_pvalue = (
                    "p_value" in stattest_name.lower() or "k-s" in stattest_name.lower()
                )
                detected = (score < threshold) if is_pvalue else (score >= threshold)
                if (
                    str(col_name) not in column_drifts
                    or "wasserstein" in stattest_name.lower()
                ):
                    column_drifts[str(col_name)] = {
                        "drift_detected": bool(detected),
                        "drift_score": round(score, 6),
                        "stattest_name": stattest_name,
                    }

    total_cols = len(ref_features.columns)
    drifted_columns_count = sum(
        1 for info in column_drifts.values() if info["drift_detected"]
    )
    drift_share = (
        round(float(drifted_columns_count) / float(total_cols), 4)
        if total_cols > 0
        else 0.0
    )
    dataset_drift = bool(float(drift_share) >= float(drift_share_threshold))

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
        "drift_share": drift_share,
        "drifted_columns_count": drifted_columns_count,
        "total_columns": total_cols,
        "column_drifts": column_drifts,
        "html_report_path": html_report_path,
        "warning": sample_warning,
    }
