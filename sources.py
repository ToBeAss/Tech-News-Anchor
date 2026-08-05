"""Ingestion — every source is an RSS/Atom feed, including YouTube.

YouTube exposes a per-channel Atom feed, so Fireship needs no API key and no
quota. Video items carry the description rather than a transcript; that's enough
for a one-line take and avoids depending on the unofficial transcript scrapers.

Two contracts matter here:

1. The Item.id assigned below is how the synthesis layer refers to items. The
   model only ever sees ids, never URLs, so a fabricated link is structurally
   impossible.
2. Item.url is canonicalised at ingestion (fragments and tracking params
   stripped), so dedup and display agree. Query params that aren't known
   trackers are preserved — some carry real routing.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests

# feedparser.parse(url) fetches with urllib: no timeout, no retries, and a
# user-agent some feeds throttle. A hung socket on the Pi would stall the whole
# run, and a single transient failure silently costs a source's worth of
# candidates. So fetching is done here and the bytes are handed to feedparser.
USER_AGENT = "techbrief/1.0 (+RSS reader; personal daily digest)"
FETCH_TIMEOUT = 15.0
FETCH_RETRIES = 2          # total attempts = 1 + retries
RETRY_BACKOFF = 1.5        # seconds, doubled each retry

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Query params that identify a tracking campaign rather than a document.
_JUNK_PARAMS = ("utm_", "ref", "fbclid", "gclid", "mc_cid", "mc_eid")


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    summary: str
    url: str
    source: str                  # feed name, e.g. "Hacker News"
    origin: str                  # display label, e.g. "mistral.ai via Hacker News"
    kind: str                    # "article" | "video" | "paper"
    published: datetime | None
    related: tuple[tuple[str, str, str], ...] = ()   # (origin, title, url) of merged duplicates

    def age(self, now: datetime | None = None) -> str:
        """Coarse relative age, e.g. '3h', '2d'. Empty when the feed gave no date."""
        if self.published is None:
            return ""
        delta = (now or datetime.now(timezone.utc)) - self.published
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return "just now"
        if hours < 48:
            return f"{int(hours)}h ago"
        return f"{int(hours // 24)}d ago"

    def for_prompt(self) -> str:
        """What the model sees. Origin is included deliberately — knowing a story
        came from a vendor domain is signal for calling it a vendor announcement.
        Age matters once the window is wider than a day: without it the model
        cannot tell a two-hour-old story from a four-day-old one, nor hedge with
        'from Monday' when surfacing something it missed earlier."""
        stamp = f", {self.age()}" if self.age() else ""
        head = f"[{self.id}] ({self.origin}, {self.kind}{stamp}) {self.title}"
        body = f"{head}\n{self.summary}" if self.summary else head
        if self.related:
            covers = "; ".join(f"{o}: {t}" for o, t, _ in self.related)
            body += f"\nSAME STORY, also covered by — {covers}"
        return body


def _fetch(url: str) -> tuple[bytes | None, str | None]:
    """Fetch feed bytes with a timeout and retries. Returns (body, error)."""
    if not url.startswith(("http://", "https://")):
        return None, None                     # local path: let feedparser handle it
    delay = RETRY_BACKOFF
    last = "unknown error"
    for attempt in range(FETCH_RETRIES + 1):
        try:
            resp = requests.get(
                url, timeout=FETCH_TIMEOUT,
                headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8"},
            )
            if resp.status_code >= 400:
                last = f"HTTP {resp.status_code}"
                # 4xx is a config problem, not a blip \u2014 retrying will not help
                if resp.status_code < 500:
                    return None, last
            else:
                return resp.content, None
        except requests.exceptions.Timeout:
            last = f"timed out after {FETCH_TIMEOUT:g}s"
        except requests.exceptions.RequestException as exc:
            last = str(exc)[:120]
        if attempt < FETCH_RETRIES:
            time.sleep(delay)
            delay *= 2
    return None, f"{last} (after {FETCH_RETRIES + 1} attempts)"


# Feeds differ wildly in how much they give: Simon Willison and arXiv fill 600
# and get cut, while The Register and Ars hard-truncate their own summaries at
# ~100 chars and lobste.rs supplies ~8. Raising this only helps the first group;
# the rest need the article fetched. Cheap either way — few items are long.
SUMMARY_CHARS = 2000


def _clean(raw: str | None, limit: int = SUMMARY_CHARS) -> str:
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text).strip()
    return text[: limit - 1] + "\u2026" if len(text) > limit else text


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def _canonical(url: str) -> str:
    """Drop the fragment and known tracking params; keep everything else.

    Fragments are always safe to strip (Simon Willison's feed appends
    #atom-everything). Query params are not — ?id= and ?p= are real routing on
    plenty of sites — so only known trackers go.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    kept = []
    for pair in parts.query.split("&"):
        if not pair:
            continue
        key = pair.split("=", 1)[0].lower()
        if key in _JUNK_PARAMS or any(key.startswith(j) for j in _JUNK_PARAMS):
            continue
        kept.append(pair)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, "&".join(kept), ""))


def _domain(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return "?"
    return host[4:] if host.startswith("www.") else host


def _matches(row: dict, keywords: list[str], excludes: list[str]) -> bool:
    """Keyword gate, applied before the per-feed cap.

    `match` is needed for firehose feeds: arXiv cs.AI announces ~150 papers a
    day in submission order, so taking the newest N is close to random
    sampling. `exclude` drops items you can't act on \u2014 digi.no prefixes
    paywalled articles with [Ekstra], and a brief that links colleagues to a
    paywall is worse than one that omits the story.
    """
    haystack = f"{row['title']} {row['summary']}".lower()
    if excludes and any(x.lower() in haystack for x in excludes):
        return False
    if not keywords:
        return True
    return any(k.lower() in haystack for k in keywords)


def fetch_feed(source: dict, *, default_limit: int, cutoff: datetime | None):
    """Fetch one feed. Never raises — a dead source degrades the brief, not the run."""
    name = source["name"]
    aggregator = bool(source.get("aggregator"))
    kind = source.get("kind", "article")
    limit = source.get("per_feed", default_limit)
    keywords = source.get("match") or []
    excludes = source.get("exclude") or []

    body, error = _fetch(source["url"])
    if error:
        # Distinguished from an empty feed on purpose: this is a lost source,
        # not a quiet one, and it measurably degrades the brief.
        return [], f"{name}: UNREACHABLE \u2014 {error}"

    try:
        parsed = feedparser.parse(body if body is not None else source["url"])
    except Exception as exc:
        return [], f"{name}: parse failed: {exc}"

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return [], f"{name}: unparseable feed ({getattr(parsed, 'bozo_exception', '?')})"

    rows, in_window = [], 0
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        when = _published(entry)
        if cutoff and when and when < cutoff:
            continue
        in_window += 1
        url = _canonical(link)
        row = {
            "title": _clean(entry.get("title"), 200) or "(untitled)",
            "summary": _clean(
                entry.get("summary")
                or entry.get("description")
                or entry.get("media_description")          # YouTube media:description
                or (entry.get("content") or [{}])[0].get("value")
            ),
            "url": url,
            "source": name,
            "origin": f"{_domain(url)} via {name}" if aggregator else name,
            "kind": kind,
            "published": when,
        }
        if _matches(row, keywords, excludes):
            rows.append(row)

    # A configured source returning nothing is worth surfacing: it's either a
    # quiet day or a feed that moved. Silence is how a dead source rots unnoticed.
    warning = None
    if not parsed.entries:
        warning = f"{name}: feed returned 0 entries \u2014 check the URL"
    elif not rows:
        warning = (f"{name}: nothing matched in window"
                   + (f" (filtered {in_window} by keywords)"
                      if (keywords or excludes) and in_window else ""))

    # HN is point-ranked and lobste.rs is hotness-ranked: the order the feed
    # arrives in IS the quality signal. Re-sorting by date throws that away and
    # keeps the newest N instead of the best N.
    if not source.get("preserve_order"):
        rows.sort(key=lambda r: r["published"] or datetime.min.replace(tzinfo=timezone.utc),
                  reverse=True)
    return rows[:limit], warning


def gate_rules(sources: list[dict]) -> dict[str, str]:
    """Source name -> relevance rule, for feeds that need LLM gating."""
    return {s["name"]: s["gate"] for s in sources if s.get("gate")}


def collect(sources: list[dict], *, hours: int, per_feed: int):
    """Fetch every source, dedupe by canonical URL, assign stable ids.

    Returns (items, warnings). Warnings are per-source conditions worth printing
    but not worth aborting over.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
    seen: set[str] = set()
    rows: list[dict] = []
    warnings: list[str] = []

    for source in sources:
        fetched, warning = fetch_feed(source, default_limit=per_feed, cutoff=cutoff)
        if warning:
            warnings.append(warning)
        for row in fetched:
            if row["url"] in seen:
                continue
            seen.add(row["url"])
            rows.append(row)

    items = [
        Item(id=f"i{n:02d}", title=r["title"], summary=r["summary"], url=r["url"],
             source=r["source"], origin=r["origin"], kind=r["kind"],
             published=r["published"], related=r.get("related", ()))
        for n, r in enumerate(rows, start=1)
    ]
    return items, warnings