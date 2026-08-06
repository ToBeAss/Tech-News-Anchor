"""Per-run audit trail — what a run's dedupe stage grouped, and which entries
it actually briefed.

Separate from memory.py's seen.json on purpose: that file answers "has this
URL already been briefed" and is read on every run to decide what's eligible.
This answers "what did THIS run's dedupe stage decide, and why" — the
question that matters after an over-merge slips through (see CLAUDE.md's
promote_leads notes on the Ryde and Texas cases), when the cluster map that
would explain it existed only in that run's stdout, already scrolled past.

One file per run rather than one growing file, keyed by a filesystem-safe
timestamp (compact ISO 8601 basic format — colons are not legal in a Windows
filename, and this project runs there; see main.py's UTF-8 reconfiguration
for the same platform concern).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from . import ROOT

RUNS_DIR = ROOT / "state" / "runs"
RETENTION_DAYS = 30    # same schedule as memory.py's seen.json

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def record(clusters: list[dict], brief) -> None:
    """Write one file for this run: the dedupe cluster map and the ids of
    everything that made it into the brief. Does not catch its own write
    failures — unlike memory.py's seen.json, a missed audit file has no
    correctness consequence for tomorrow's run, so there is nothing to fail
    open about; a write error here is a disk problem worth seeing directly."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime(_STAMP_FORMAT)
    payload = {
        "dedupe_clusters": clusters,
        "entry_ids": [e.item.id for e in brief.entries()],
    }
    (RUNS_DIR / f"{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prune(*, days: int = RETENTION_DAYS, now: datetime | None = None) -> None:
    if not RUNS_DIR.exists():
        return
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    for path in RUNS_DIR.glob("*.json"):
        try:
            stamp = datetime.strptime(path.stem, _STAMP_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue    # not one of ours; leave it alone
        if stamp < cutoff:
            path.unlink(missing_ok=True)
