"""Tests for the Discord webhook notifier.

We never actually hit Discord in tests. We verify the no-webhook no-op
and mock `urllib.request.urlopen` to test the happy path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cadenza_redteam.notify import notify


def test_notify_no_webhook_is_noop(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK", raising=False)
    assert notify("hello") is False


def test_notify_with_webhook_success(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://discord/webhooks/fake")
    fake_resp = MagicMock()
    fake_resp.status = 204
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp) as uo:
        ok = notify("training done", title="t", status="ok")
    assert ok is True
    uo.assert_called_once()
    # Posted payload contains our message
    call = uo.call_args
    req = call.args[0]
    body = req.data.decode("utf-8")
    assert "training done" in body
    assert '"title": "t"' in body


def test_notify_with_webhook_failure(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://discord/webhooks/fake")
    from urllib.error import URLError

    with patch("urllib.request.urlopen", side_effect=URLError("nope")):
        ok = notify("training done")
    assert ok is False


def test_notify_status_colors(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://discord/webhooks/fake")
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *a: None

    with patch("urllib.request.urlopen", return_value=fake_resp) as uo:
        notify("warn!", status="warn")
        notify("err!", status="error")
    assert uo.call_count == 2
