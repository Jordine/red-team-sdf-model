"""Discord webhook notifier.

Training runs on vast.ai are fire-and-forget — we want a notification when
they finish (or crash). Usage:

    from cadenza_redteam.notify import notify
    notify("SDF training finished", status="ok")

The webhook URL comes from the DISCORD_WEBHOOK env var. If unset, the call
is a silent no-op.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Literal

log = logging.getLogger(__name__)

Status = Literal["ok", "warn", "error", "info"]

_COLOR = {
    "ok": 0x2ECC71,  # green
    "warn": 0xF39C12,  # orange
    "error": 0xE74C3C,  # red
    "info": 0x3498DB,  # blue
}


def notify(
    message: str,
    *,
    title: str = "cadenza-redteam",
    status: Status = "info",
    webhook_url: str | None = None,
    timeout: float = 5.0,
) -> bool:
    """Post a message to a Discord webhook. Returns True on success.

    No-op (returns False) if no webhook URL is configured.
    """
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK")
    if not url:
        log.debug("DISCORD_WEBHOOK unset; skipping notify: %s", message)
        return False

    host = socket.gethostname()
    payload = {
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": _COLOR.get(status, _COLOR["info"]),
                "footer": {"text": f"{host} · {time.strftime('%Y-%m-%d %H:%M:%S')}"},
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError) as e:
        log.warning("Discord notify failed: %s", e)
        return False
