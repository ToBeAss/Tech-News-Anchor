"""Ingestion — every source is an RSS/Atom feed, including YouTube.

YouTube exposes a per-channel Atom feed, so Fireship needs no API key and no
quota. Video items carry the description rather than a transcript; that's enough
for a one-line take and avoids depending on the unofficial transcript scrapers.

The Item.id assigned here is the contract with the synthesis layer: the model
only ever references items by id, never by URL. URLs are resolved locally at
render time, so a fabricated link is structurally impossible.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

import feedparser

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Query params that identify a tracking campaign rather than a document.
_JUNK_PARAMS = ("utm_", "ref=", "fbclid", "gclid")


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    summary: str
    url: str
    source: str
    kind: str                    # "article" | "video"
    published: datetime | None

    def for_prompt(self) -> str:
        """The model sees this — no URL, so it can't echo or invent one."""
        head = f"[{self.id}] ({self.source}, {self.kind}) {self.title}"
        return f"{head}\n{self.summary}" if self.summary else head


def _clean(raw: str | None, limit: int = 600) -> str:
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text).strip()
    return text[: limit - 1] + "…" if len(text) > limit else text


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def _canonical(url: str) -> str:
    """Strip tracking params so the same story from two feeds dedupes."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    query = "&".join(
        p for p in parts.query.split("&")
        if p and not any(p.lower().startswith(j) or j in p.lower() for j in _JUNK_PARAMS)
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), query, ""))


def fetch_feed(name: str, url: str, kind: str, *, limit: int, cutoff: datetime | None):
    """Fetch one feed. Never raises — a dead source degrades the brief, not the run."""
    try:
        parsed = feedparser.parse(url)
    except Exception as exc:
        return [], f"{name}: fetch failed: {exc}"

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return [], f"{name}: unparseable feed ({getattr(parsed, 'bozo_exception', '?')})"

    raw = []
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        when = _published(entry)
        if cutoff and when and when < cutoff:
            continue
        raw.append({
            "title": _clean(entry.get("title"), 200) or "(untitled)",
            "summary": _clean(entry.get("summary") or entry.get("description")),
            "url": link,
            "source": name,
            "kind": kind,
            "published": when,
        })

    raw.sort(key=lambda r: r["published"] or datetime.min.replace(tzinfo=timezone.utc),
             reverse=True)
    return raw[:limit], None


def collect(sources: list[dict], *, hours: int, per_feed: int):
    """Fetch every source, dedupe by canonical URL, assign stable ids.

    Returns (items, warnings). Warnings are per-source failures worth printing
    but not worth aborting over.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    seen: set[str] = set()
    rows: list[dict] = []
    warnings: list[str] = []

    for src in sources:
        fetched, warning = fetch_feed(
            src["name"], src["url"], src.get("kind", "article"),
            limit=per_feed, cutoff=cutoff,
        )
        if warning:
            warnings.append(warning)
        for row in fetched:
            key = _canonical(row["url"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    items = [
        Item(id=f"i{n:02d}", title=r["title"], summary=r["summary"], url=r["url"],
             source=r["source"], kind=r["kind"], published=r["published"])
        for n, r in enumerate(rows, start=1)
    ]
    return items, warnings
