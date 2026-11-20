"""
ModelWatch - Kalıcı İzleme Geçmişi Modülü
Her izleme koşusunu SQLite veritabanına kaydeder ve geçmiş kayıtları sorgular.
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("modelwatch.storage")

DEFAULT_DB_PATH = "data/modelwatch_history.db"

_EXTRA_COLUMNS: dict[str, str] = {
    "model_name": "TEXT",
    "model_alias": "TEXT",
    "model_version": "TEXT",
    "baseline_metrics_json": "TEXT",
}


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(str(Path(db_path)))


def _safe_float(val) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def _safe_json_loads(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
        if isinstance(fallback, bool):
            return parsed if isinstance(parsed, bool) else fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback
    except (json.JSONDecodeError, TypeError):
        logger.warning("Bozuk JSON geçmiş kaydı varsayılan değerle döndürüldü.")
        return fallback


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(monitoring_runs)").fetchall()
    }
    for col_name, col_type in _EXTRA_COLUMNS.items():
        if col_name not in existing:
            conn.execute(
                f"ALTER TABLE monitoring_runs ADD COLUMN {col_name} {col_type}"
            )


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """İzleme geçmişi tablosunu oluşturur ve eksik kolonları günceller."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    report_tag TEXT,
                    n_samples INTEGER,
                    dataset_drift INTEGER,
                    drift_share REAL,
                    drifted_columns_json TEXT,
                    accuracy REAL,
                    f1 REAL,
                    precision_score REAL,
                    recall_score REAL,
                    positive_rate REAL,
                    alert_status TEXT NOT NULL,
                    alert_count INTEGER NOT NULL,
                    alerts_json TEXT,
                    model_name TEXT,
                    model_alias TEXT,
                    model_version TEXT,
                    baseline_metrics_json TEXT
                )
                """
            )
            _ensure_columns(conn)
    finally:
        conn.close()


def save_monitoring_run(
    drift_result: dict,
    monitoring_result: dict,
    alert_result: dict,
    db_path: str = DEFAULT_DB_PATH,
    model_name: str | None = None,
    model_alias: str | None = None,
    model_version: str | None = None,
    baseline_metrics: dict | None = None,
) -> int:
    """Tek bir izleme koşusunun özetini veritabanına yazar ve kayıt ID'sini döner."""
    init_db(db_path)

    mon = monitoring_result or {}
    drift = drift_result or {}
    alert = alert_result or {}

    metrics = mon.get("metrics") or {}
    column_drifts = drift.get("column_drifts") or {}
    drifted_cols = [
        col for col, info in column_drifts.items() if info.get("drift_detected")
    ]

    dataset_drift_raw = drift.get("dataset_drift")
    dataset_drift_int = (
        None if dataset_drift_raw is None else int(bool(dataset_drift_raw))
    )

    resolved_model_name = model_name or mon.get("model_name")
    resolved_model_alias = model_alias or mon.get("alias") or mon.get("model_alias")
    resolved_model_version = (
        str(model_version)
        if model_version is not None
        else (
            str(mon.get("model_version"))
            if mon.get("model_version") is not None
            else None
        )
    )
    resolved_baseline = (
        baseline_metrics
        if baseline_metrics is not None
        else mon.get("baseline_metrics")
    )

    row = (
        mon.get("timestamp_utc")
        or drift.get("timestamp_utc")
        or datetime.now(timezone.utc).isoformat(),
        drift.get("report_tag") or mon.get("run_tag") or "run",
        mon.get("n_samples"),
        dataset_drift_int,
        _safe_float(drift.get("drift_share")),
        _safe_json_dumps(drifted_cols),
        _safe_float(metrics.get("accuracy")),
        _safe_float(metrics.get("f1")),
        _safe_float(metrics.get("precision")),
        _safe_float(metrics.get("recall")),
        _safe_float(mon.get("positive_rate")),
        alert.get("status") or "OK",
        int(alert.get("alert_count") or 0),
        _safe_json_dumps(alert.get("alerts") or []),
        resolved_model_name,
        resolved_model_alias,
        resolved_model_version,
        _safe_json_dumps(resolved_baseline) if resolved_baseline is not None else None,
    )

    conn = _connect(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO monitoring_runs (
                    created_at_utc, report_tag, n_samples, dataset_drift, drift_share,
                    drifted_columns_json, accuracy, f1, precision_score, recall_score,
                    positive_rate, alert_status, alert_count, alerts_json,
                    model_name, model_alias, model_version, baseline_metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            return int(cursor.lastrowid)
    finally:
        conn.close()


def get_monitoring_history(
    limit: int = 20,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Son N izleme koşusunu en yeniden eskiye doğru listeler."""
    init_db(db_path)
    safe_limit = max(1, min(int(limit), 200))

    conn = _connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM monitoring_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    finally:
        conn.close()

    results = []
    for r in rows:
        keys = set(r.keys())
        baseline_raw = (
            r["baseline_metrics_json"] if "baseline_metrics_json" in keys else None
        )
        results.append(
            {
                "id": r["id"],
                "created_at_utc": r["created_at_utc"],
                "report_tag": r["report_tag"],
                "n_samples": r["n_samples"],
                "dataset_drift": None
                if r["dataset_drift"] is None
                else bool(r["dataset_drift"]),
                "drift_share": r["drift_share"],
                "drifted_columns": _safe_json_loads(r["drifted_columns_json"], []),
                "metrics": {
                    "accuracy": r["accuracy"],
                    "f1": r["f1"],
                    "precision": r["precision_score"],
                    "recall": r["recall_score"],
                },
                "positive_rate": r["positive_rate"],
                "alert_status": r["alert_status"],
                "alert_count": r["alert_count"],
                "alerts": _safe_json_loads(r["alerts_json"], []),
                "model_name": r["model_name"] if "model_name" in keys else None,
                "model_alias": r["model_alias"] if "model_alias" in keys else None,
                "model_version": r["model_version"]
                if "model_version" in keys
                else None,
                "baseline_metrics": (
                    _safe_json_loads(baseline_raw, {})
                    if baseline_raw is not None
                    else None
                ),
            }
        )
    return results
