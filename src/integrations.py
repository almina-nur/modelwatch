"""ModelWatch - Istege Bagli Uretim Kancalari (Webhook + JSON log)."""

import json
import logging
import os
from datetime import datetime, timezone
from urllib import request
from urllib.error import URLError

logger = logging.getLogger("modelwatch.alerts")


def emit_alert_notifications(
    alert_result: dict, webhook_url: str | None = None, timeout_sec: float = 3.0
) -> dict:
    """Alarm sonucunu loglar ve webhook_url tanimliysa JSON payload gonderir."""
    target_url = webhook_url or os.getenv("MODELWATCH_WEBHOOK_URL")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": alert_result.get("status", "OK"),
        "alert_count": alert_result.get("alert_count", 0),
        "drifted_columns": alert_result.get("drifted_columns", []),
        "alerts": alert_result.get("alerts", []),
    }
    if payload["status"] in ("WARNING", "CRITICAL"):
        logger.warning(json.dumps(payload, ensure_ascii=False))
    else:
        logger.info(json.dumps(payload, ensure_ascii=False))
    if not target_url:
        return {
            "webhook_configured": False,
            "delivered": False,
            "detail": "MODELWATCH_WEBHOOK_URL tanimli degil.",
        }
    req = request.Request(
        url=target_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            return {
                "webhook_configured": True,
                "delivered": 200 <= resp.status < 300,
                "http_status": resp.status,
            }
    except (URLError, TimeoutError, OSError) as exc:
        logger.error("Webhook gonderimi basarisiz: %s", exc)
        return {"webhook_configured": True, "delivered": False, "detail": str(exc)}
