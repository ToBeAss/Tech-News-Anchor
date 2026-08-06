"""Webhook delivery — the network half of the Discord and Slack renderers.

brief/render/{discord,slack}.py build payloads with no I/O; this module sends
them. Split the same way qotd's dispatch.py (posting) and its renderer are
split, so the payload logic stays testable without a webhook.

A webhook failure must never crash a run that already produced a valid brief —
the brief was the expensive part. Every failure here is logged to stderr and
turned into a return value, never raised.

Discord and Slack disagree on what counts as success and where a rate-limit
retry delay lives (a JSON body field vs. a response header), so `post()` takes
a small `Target` descriptor rather than hardcoding either. Everything else —
retry loop, sequencing, logging shape — is genuinely shared.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from . import ROOT

STATE_DIR = ROOT / "state"

TIMEOUT = 10.0
MAX_ATTEMPTS = 3           # includes the first try
INTER_MESSAGE_DELAY = 1.0  # seconds between messages of a multi-message brief


@dataclass(frozen=True)
class Target:
    name: str
    env_var: str
    success: tuple[int, ...]
    retry_after: Callable[[requests.Response], float]


DISCORD = Target("discord", "DISCORD_WEBHOOK", (200, 204),
                 lambda resp: float(resp.json().get("retry_after", 1)))
SLACK = Target("slack", "SLACK_WEBHOOK", (200,),
              lambda resp: float(resp.headers.get("Retry-After", 1)))


def _post_one(webhook: str, payload: dict, target: Target) -> bool:
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.post(webhook, json=payload, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            print(f"{target.name}: request failed: {exc}", file=sys.stderr)
            return False

        if resp.status_code in target.success:
            return True

        if resp.status_code == 429 and attempt < MAX_ATTEMPTS - 1:
            try:
                retry_after = target.retry_after(resp)
            except Exception:
                retry_after = 1.0
            time.sleep(retry_after)
            continue

        print(f"{target.name}: webhook rejected: status={resp.status_code} "
              f"body={resp.text[:300]}", file=sys.stderr)
        return False
    return False


def post(payloads: list[dict], target: Target) -> bool:
    """Send every payload in order. Returns True only if all of them landed.

    Sequential with a short delay between messages, so a multi-message brief
    keeps its Today's Signal / On the Horizon ordering in the channel.
    """
    webhook = os.getenv(target.env_var)
    if not webhook:
        print(f"{target.name}: {target.env_var} not set; skipping delivery", file=sys.stderr)
        return False

    ok = True
    for n, payload in enumerate(payloads):
        if n:
            time.sleep(INTER_MESSAGE_DELAY)
        ok = _post_one(webhook, payload, target) and ok
    return ok


def _marker_path(target: Target, today: date | None = None) -> Path:
    return STATE_DIR / f"posted-{target.name}-{(today or date.today()).isoformat()}"


def already_posted_today(target: Target) -> bool:
    return _marker_path(target).exists()


def mark_posted(target: Target) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    _marker_path(target).touch()
