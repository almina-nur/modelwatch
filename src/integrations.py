"""
ModelWatch - Entegrasyon ve Bildirim Modülü
Alarm sonuçlarını yapılandırılmış JSON log olarak yazar ve isteğe bağlı webhook gönderir.
"""

import json
import logging
import os
from urllib import error, request

logger = logging.getLogger("modelwatch.integrations")


def emit_alert_notifications(
    alert_result: dict,
    webhook_url: str | None = None,
    timeout_seconds: float = 3.0,
) -> dict:
    """Alarm sonucunu loglar ve tanımlıysa webhook URL'sine POST gönderir."""
    target_url = (
        webhook_url
        if webhook_url is not None
        else os.getenv("MODELWATCH_WEBHOOK_URL", "").strip()
    )
    payload = {
        "event": "modelwatch_alert",
        "alert": alert_result,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))

    webhook_sent = False
    webhook_error = None

    if target_url:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            target_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                webhook_sent = 200 <= resp.status < 300
                if not webhook_sent:
                    webhook_error = f"HTTP {resp.status}"
        except (error.URLError, TimeoutError, OSError) as exc:
            webhook_error = str(exc)
            logger.warning("Webhook bildirimi gönderilemedi: %s", exc)

    return {
        "webhook_configured": bool(target_url),
        "webhook_sent": webhook_sent,
        "webhook_error": webhook_error,
    }
