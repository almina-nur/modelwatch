"""
ModelWatch - Kalıcı İzleme Geçmişi Modülü
Her izleme koşusunu SQLite veritabanına kaydeder ve geçmiş kayıtları sorgular.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = "data/modelwatch_history.db"


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """İzleme geçmişi tablosunu oluşturur."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(path))) as conn, conn:
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
                    alerts_json TEXT
                )
                """
        )


def save_monitoring_run(
    drift_result: dict,
    monitoring_result: dict,
    alert_result: dict,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Tek bir izleme koşusunun özetini veritabanına yazar ve kayıt ID'sini döner."""
    init_db(db_path)

    metrics = (monitoring_result or {}).get("metrics") or {}
    column_drifts = (drift_result or {}).get("column_drifts") or {}
    drifted_cols = [
        col for col, info in column_drifts.items() if info.get("drift_detected")
    ]

    dataset_drift_raw = (drift_result or {}).get("dataset_drift")
    dataset_drift_int = (
        None if dataset_drift_raw is None else int(bool(dataset_drift_raw))
    )

    row = (
        (monitoring_result or {}).get("timestamp_utc")
        or (drift_result or {}).get("timestamp_utc")
        or datetime.now(timezone.utc).isoformat(),
        (drift_result or {}).get("report_tag")
        or (monitoring_result or {}).get("run_tag")
        or "run",
        (monitoring_result or {}).get("n_samples"),
        dataset_drift_int,
        (drift_result or {}).get("drift_share"),
        json.dumps(drifted_cols, ensure_ascii=False),
        metrics.get("accuracy"),
        metrics.get("f1"),
        metrics.get("precision"),
        metrics.get("recall"),
        (monitoring_result or {}).get("positive_rate"),
        (alert_result or {}).get("status") or "OK",
        int((alert_result or {}).get("alert_count") or 0),
        json.dumps((alert_result or {}).get("alerts") or [], ensure_ascii=False),
    )

    with closing(sqlite3.connect(str(Path(db_path)))) as conn, conn:
        cursor = conn.execute(
            """
                INSERT INTO monitoring_runs (
                    created_at_utc, report_tag, n_samples, dataset_drift, drift_share,
                    drifted_columns_json, accuracy, f1, precision_score, recall_score,
                    positive_rate, alert_status, alert_count, alerts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
            row,
        )
        return int(cursor.lastrowid)


def get_monitoring_history(
    limit: int = 20,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Son N izleme koşusunu en yeniden eskiye doğru listeler."""
    init_db(db_path)
    safe_limit = max(1, min(int(limit), 200))

    with closing(sqlite3.connect(str(Path(db_path)))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM monitoring_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    results = []
    for r in rows:
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
                "drifted_columns": json.loads(r["drifted_columns_json"] or "[]"),
                "metrics": {
                    "accuracy": r["accuracy"],
                    "f1": r["f1"],
                    "precision": r["precision_score"],
                    "recall": r["recall_score"],
                },
                "positive_rate": r["positive_rate"],
                "alert_status": r["alert_status"],
                "alert_count": r["alert_count"],
                "alerts": json.loads(r["alerts_json"] or "[]"),
            }
        )
    return results
