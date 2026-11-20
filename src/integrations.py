"""
ModelWatch - Entegrasyon ve Bildirim Modülü
Alarm sonuçlarını yapılandırılmış JSON log olarak yazar ve isteğe bağlı webhook gönderir.
"""

import json
import logging
import os
from datetime import datetime, timezone
from urllib import error, request
from urllib.parse import urlparse

logger = logging.getLogger("modelwatch.integrations")


def _normalize_webhook_url(webhook_url: str | None) -> str:
    raw = (
        webhook_url
        if webhook_url is not None
        else os.getenv("MODELWATCH_WEBHOOK_URL", "")
    )
    return raw.strip()


def _is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def emit_alert_notifications(
    alert_result: dict,
    webhook_url: str | None = None,
    timeout_seconds: float = 3.0,
) -> dict:
    """Alarm sonucunu loglar ve tanımlıysa webhook URL'sine POST gönderir."""
    target_url = _normalize_webhook_url(webhook_url)
    payload = {
        "event": "modelwatch_alert",
        "alert": alert_result,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))

    webhook_sent = False
    webhook_error: str | None = None
    status_code: int | None = None

    if target_url:
        if not _is_valid_http_url(target_url):
            webhook_error = (
                "Geçersiz webhook URL şeması (yalnızca http/https desteklenir)."
            )
            logger.warning("Webhook bildirimi reddedildi: %s", webhook_error)
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = request.Request(
                target_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=timeout_seconds) as resp:
                    status_code = int(getattr(resp, "status", 200))
                    webhook_sent = 200 <= status_code < 300
                    if not webhook_sent:
                        webhook_error = f"HTTP {status_code}"
                        logger.warning(
                            "Webhook bildirimi başarısız yanıt döndü: %s", webhook_error
                        )
            except (error.URLError, TimeoutError, OSError) as exc:
                webhook_error = str(exc)
                logger.warning("Webhook bildirimi gönderilemedi: %s", exc)

    result = {
        "webhook_configured": bool(target_url),
        "webhook_sent": webhook_sent,
        "webhook_error": webhook_error,
        "status_code": status_code,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        json.dumps(
            {"event": "modelwatch_notification_result", **result}, ensure_ascii=False
        )
    )
    return result
