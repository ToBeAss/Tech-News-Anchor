"""Cross-run memory — what has already been briefed.

The window overlaps between runs (48h, or 168h if you widen it), so every run
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
from datetime import datetime, timedelta, timezone
from typing import Any

from . import ROOT

STATE_DIR = ROOT / "state"
SEEN_PATH = STATE_DIR / "seen.json"

RETENTION_DAYS = 30     # prune entries older than this; well past any window
SNAPSHOT_CHARS = 400    # enough to audit a claim; small enough to keep forever


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
        if _urls(item) & seen.keys():
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def _urls(item) -> set[str]:
    """Every link that should be considered covered once this item is briefed."""
    return {item.url, *(r.url for r in item.related)}


def record(seen: dict[str, Any], brief) -> dict[str, Any]:
    """Mark everything in a delivered brief as covered, with a snapshot of the
    source text as it read at the time.

    The snapshot exists because feeds mutate. digi.no revised a contract figure
    from 55 to 38 milliarder between two runs on the same day, and the URL slug
    still carried the original number. Without a snapshot there is no way to
    answer "what did the model actually see when it wrote that?" — you end up
    inferring it from a slug, which is exactly how a correct fact gets
    misdiagnosed as a hallucination. `main.py --why <url>` reads it back.

    Also records what was written, so a published claim can be traced back to
    the text it came from.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for entry in brief.entries():
        snapshot = {
            "briefed_at": now,
            "title": entry.item.title[:200],
            "summary": (entry.item.summary or "")[:SNAPSHOT_CHARS],
            "wrote": {
                "headline": entry.headline,
                "fact": entry.fact,
                "comment": entry.comment,
            },
        }
        for url in _urls(entry.item):
            seen[url] = snapshot
    return seen


def snapshot(seen: dict[str, Any], url: str) -> dict[str, Any] | None:
    """What the source said, and what was written about it, when it was briefed."""
    return seen.get(url)
