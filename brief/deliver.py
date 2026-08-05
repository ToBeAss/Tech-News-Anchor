"""Discord delivery — the network half of the Discord renderer.

brief/render/discord.py builds payloads with no I/O; this module sends them.
Split the same way qotd's dispatch.py (posting) and its renderer are split, so
the payload logic stays testable without a webhook.

A webhook failure must never crash a run that already produced a valid brief —
the brief was the expensive part. Every failure here is logged to stderr and
turned into a return value, never raised.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

from . import ROOT

WEBHOOK_ENV = "FYRTARN_WEBHOOK"
STATE_DIR = ROOT / "state"

TIMEOUT = 10.0
MAX_ATTEMPTS = 3           # includes the first try
INTER_MESSAGE_DELAY = 1.0  # seconds between messages of a multi-message brief


def _post_one(webhook: str, payload: dict) -> bool:
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.post(webhook, json=payload, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            print(f"discord: request failed: {exc}", file=sys.stderr)
            return False

        if resp.status_code in (200, 204):
            return True

        if resp.status_code == 429 and attempt < MAX_ATTEMPTS - 1:
            try:
                retry_after = float(resp.json().get("retry_after", 1))
            except Exception:
                retry_after = 1.0
            time.sleep(retry_after)
            continue

        print(f"discord: webhook rejected: status={resp.status_code} "
              f"body={resp.text[:300]}", file=sys.stderr)
        return False
    return False


def post(payloads: list[dict]) -> bool:
    """Send every payload in order. Returns True only if all of them landed.

    Sequential with a short delay between messages, so a multi-message brief
    keeps its Today's Signal / On the Horizon ordering in the channel.
    """
    webhook = os.getenv(WEBHOOK_ENV)
    if not webhook:
        print(f"discord: {WEBHOOK_ENV} not set; skipping delivery", file=sys.stderr)
        return False

    ok = True
    for n, payload in enumerate(payloads):
        if n:
            time.sleep(INTER_MESSAGE_DELAY)
        ok = _post_one(webhook, payload) and ok
    return ok


def _marker_path(today: date | None = None) -> Path:
    return STATE_DIR / f"posted-{(today or date.today()).isoformat()}"


def already_posted_today() -> bool:
    return _marker_path().exists()


def mark_posted() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    _marker_path().touch()
