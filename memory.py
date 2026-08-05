"""Cross-run memory — what has already been briefed.

The window overlaps between runs (36h, or 168h if you widen it), so every run
re-sees most of the previous run's candidates. Without this, a recurring story
resurfaces daily with freshly-written commentary, and a pinned video slot would
show the same Fireship upload for a week.

Only items that actually appeared in a brief are recorded. Ingested-but-unused
candidates stay eligible: a story that was crowded out on a busy day should
still get its chance on a quiet one.

State lives in state/seen.json, keyed by canonical URL. Ported from the QOTD
project's memory.py, with URLs replacing quote text — exact-match dedup rather
than anti-repetition, which is a stronger and simpler check.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(__file__).resolve().parent / "state"
SEEN_PATH = STATE_DIR / "seen.json"

RETENTION_DAYS = 30     # prune entries older than this; well past any window


def load() -> dict[str, Any]:
    try:
        return json.loads(SEEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(seen: dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    tmp = SEEN_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(seen, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(SEEN_PATH)          # atomic: a crash mid-write can't corrupt state


def prune(seen: dict[str, Any], *, days: int = RETENTION_DAYS) -> dict[str, Any]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return {url: meta for url, meta in seen.items()
            if str(meta.get("briefed_at", "")) >= cutoff}


def filter_unseen(items: list, seen: dict[str, Any]):
    """Drop items already briefed. Returns (items, dropped_count).

    An item is also considered seen if any of its merged duplicates was briefed
    under a different URL — otherwise tomorrow's run resurfaces the same story
    via the other outlet's link.
    """
    kept, dropped = [], 0
    for item in items:
        urls = {item.url, *(u for _o, _t, u in item.related)}
        if urls & seen.keys():
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def record(seen: dict[str, Any], brief) -> dict[str, Any]:
    """Mark everything in a delivered brief as covered."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = [*brief.top, *brief.also] + ([brief.video] if brief.video else [])
    for entry in entries:
        for url in (entry.item.url, *(u for _o, _t, u in entry.item.related)):
            seen[url] = {"briefed_at": now, "title": entry.item.title[:120]}
    return seen