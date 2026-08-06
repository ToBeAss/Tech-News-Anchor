"""Ingestion — every source is an RSS/Atom feed, including YouTube.

YouTube exposes a per-channel Atom feed, so Fireship needs no API key and no
quota. Video items carry the description rather than a transcript; that's enough
for a one-line take and avoids depending on the unofficial transcript scrapers.

Two contracts matter here:

1. The Item.id assigned in collect() is how the synthesis layer refers to items.
   The model only ever sees ids, never URLs, so a fabricated link is structurally
   impossible.
2. Item.url is canonicalised at ingestion (fragments and tracking params
   stripped), so dedup and display agree. Query params that aren't known
   trackers are preserved — some carry real routing.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests

from .warnings import Warning

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

# Sort key for undated items: oldest, so a feed that omits dates never displaces
# something with a real timestamp.
EPOCH = datetime.min.replace(tzinfo=timezone.utc)


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
    # A digit-free community-engagement label ("heavily upvoted"), currently
    # populated only for Hacker News from hnrss's point count. Deliberately a
    # word, not the number: verify checks every figure a written fact contains
    # against the item's own title+summary, and a real point count sitting in
    # for_prompt() but absent from that check would fail an honest quote the
    # same way it would flag a fabrication — indistinguishable to the checker,
    # even though it's neither. A word bucket can't be quoted as a figure, so
    # it never needs checking and can't be mistaken for a claim about the story.
    prominence: str = ""
    # Duplicates merged into this one by dedupe. Whole Items, not a projection:
    # verify needs their text to check a figure, and promoting one to lead needs
    # its summary so the recorded snapshot still matches the link.
    related: tuple["Item", ...] = ()

    def age(self, now: datetime | None = None) -> str:
        """Coarse relative age, e.g. '3h ago', '2d ago'. Empty when undated."""
        if self.published is None:
            return ""
        hours = ((now or datetime.now(timezone.utc)) - self.published).total_seconds() / 3600
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
        'from Monday' when surfacing something it missed earlier. `prominence`
        exists because `preserve_order` alone only encodes a source's own ranking
        as list POSITION, which is invisible once items from several sources are
        interleaved in one flat catalogue — this states it in words instead.

        Merged duplicates contribute their TITLE only. verify.py checks written
        figures against exactly this text, so widening it to sibling summaries
        would let a figure the model never saw pass as supported. `prominence`
        is never in that text either, for the same reason: it isn't a claim
        about the story, so it must never be checkable as one.
        """
        tags = ", ".join(t for t in (self.kind, self.age(), self.prominence) if t)
        head = f"[{self.id}] ({self.origin}, {tags}) {self.title}"
        body = f"{head}\n{self.summary}" if self.summary else head
        if self.related:
            covers = "; ".join(f"{r.origin}: {r.title}" for r in self.related)
            body += f"\nSAME STORY, also covered by — {covers}"
        return body


def _fetch(url: str) -> tuple[bytes | None, str | None]:
    """Fetch feed bytes with a timeout and retries. Returns (body, error)."""
    if not url.startswith(("http://", "https://")):
        # A typo'd scheme is a config error; retrying it just costs 4.5s of sleep.
        return None, "not an http(s) url"
    delay = RETRY_BACKOFF
    last = "unknown error"
    for attempt in range(FETCH_RETRIES + 1):
        try:
            resp = requests.get(
                url, timeout=FETCH_TIMEOUT,
                headers={"User-Agent": USER_AGENT,
                         "Accept": "application/rss+xml, application/atom+xml, "
                                   "application/xml;q=0.9, */*;q=0.8"},
            )
            if resp.status_code >= 400:
                last = f"HTTP {resp.status_code}"
                # 4xx is a config problem, not a blip — retrying will not help
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

# hnrss.org's <description> is HN's OWN discussion metadata — the article link
# again, the comments link, the score — never an excerpt of the linked article,
# which HN doesn't host and hnrss can't fetch. Every single Hacker News item
# carries this, verbatim, unbounded by how much real content exists (none).
# Passing it through as "summary" let every HN item pretend to carry content it
# never had: dedupe was trusted to find a bridging fact in it, synth to ground a
# figure in it, verify to accept a written number because it merely matched a
# Points/Comments count — none of which was ever real. Stripped to empty is the
# honest state; for_prompt() already omits the body line when summary is "".
#
# The point count is the one real thing in this blob (see Item.prominence for
# why it's recovered as a word, not the digit itself), so the pattern captures
# it rather than just matching — one regex, so extraction and stripping can
# never disagree about what counts as this boilerplate.
_HN_METADATA_RE = re.compile(
    r"^Article URL:\s*\S+\s+Comments URL:\s*\S+\s+Points:\s*(\d+)\s+#\s*Comments:\s*\d+$"
)

# Starting points, not tuned against a real distribution yet — revisit once a
# week of runs shows what's typical for this feed's point counts.
_PROMINENCE_BUCKETS = ((250, "heavily upvoted"), (120, "well upvoted"))


def _normalize_for_match(raw: str | None) -> str:
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    return _WS_RE.sub(" ", text).strip()


def _hn_prominence(raw: str | None) -> str:
    """A digit-free engagement label from hnrss's point count, or ""."""
    match = _HN_METADATA_RE.match(_normalize_for_match(raw))
    if not match:
        return ""
    points = int(match.group(1))
    for threshold, label in _PROMINENCE_BUCKETS:
        if points >= threshold:
            return label
    return ""


def _clean(raw: str | None, limit: int = SUMMARY_CHARS) -> str:
    if not raw:
        return ""
    text = _normalize_for_match(raw)
    if _HN_METADATA_RE.match(text):
        return ""
    return text[: limit - 1] + "…" if len(text) > limit else text


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
        if any(key.startswith(junk) for junk in _JUNK_PARAMS):
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


def _matches(item: Item, keywords: list[str], excludes: list[str]) -> bool:
    """Keyword gate, applied before the per-feed cap.

    `match` is needed for firehose feeds: arXiv cs.AI announces ~150 papers a
    day in submission order, so taking the newest N is close to random
    sampling. `exclude` drops items you can't act on — digi.no prefixes
    paywalled articles with [Ekstra], and a brief that links colleagues to a
    paywall is worse than one that omits the story.
    """
    haystack = f"{item.title} {item.summary}".lower()
    if excludes and any(x.lower() in haystack for x in excludes):
        return False
    if not keywords:
        return True
    return any(k.lower() in haystack for k in keywords)


def fetch_feed(source: dict, *, default_limit: int, cutoff: datetime | None):
    """Fetch one feed. Never raises — a dead source degrades the brief, not the run.

    Items come back with an empty id; collect() assigns them after cross-feed
    URL dedup, so the numbering has no gaps.
    """
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
        return [], Warning(f"{name}: UNREACHABLE — {error}", degraded=True)

    try:
        parsed = feedparser.parse(body)
    except Exception as exc:
        return [], Warning(f"{name}: parse failed: {exc}")

    if getattr(parsed, "bozo", 0) and not parsed.entries:
        return [], Warning(f"{name}: unparseable feed ({getattr(parsed, 'bozo_exception', '?')})")

    rows: list[Item] = []
    in_window = 0
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        when = _published(entry)
        if cutoff and when and when < cutoff:
            continue
        in_window += 1
        url = _canonical(link)
        raw_description = (
            entry.get("summary")
            or entry.get("description")
            or entry.get("media_description")          # YouTube media:description
            or (entry.get("content") or [{}])[0].get("value")
        )
        item = Item(
            id="",
            title=_clean(entry.get("title"), 200) or "(untitled)",
            summary=_clean(raw_description),
            prominence=_hn_prominence(raw_description),
            url=url,
            source=name,
            origin=f"{_domain(url)} via {name}" if aggregator else name,
            kind=kind,
            published=when,
        )
        if _matches(item, keywords, excludes):
            rows.append(item)

    # A configured source returning nothing is worth surfacing: it's either a
    # quiet day or a feed that moved. Silence is how a dead source rots unnoticed.
    warning = None
    if not parsed.entries:
        warning = Warning(f"{name}: feed returned 0 entries — check the URL")
    elif not rows:
        warning = Warning(f"{name}: nothing matched in window"
                          + (f" (filtered {in_window} by keywords)"
                             if (keywords or excludes) and in_window else ""))
    elif len(rows) > limit and not source.get("preserve_order") and not keywords:
        # preserve_order sources take the top N of an already-ranked feed by
        # design — that's correct, not a cap binding. A date-sorted source
        # dropping items past the limit is a RECENCY cap: a busy hour can push
        # the window's actual biggest story out before gate, dedupe, or
        # synthesis ever see it, and nothing else would surface that silently.
        #
        # Sources with a `match` filter are excluded: arXiv cs.AI is DESIGNED
        # to cap hard after filtering a firehose, so this would fire every run
        # restating a known, accepted tradeoff rather than reporting anything
        # new — exactly the warning-fatigue this project already avoided once
        # with the off-lead figure check before verify.promote_leads existed.
        warning = Warning(f"{name}: capped at {limit}, {len(rows) - limit} more in window were dropped")

    # HN is point-ranked and lobste.rs is hotness-ranked: the order the feed
    # arrives in IS the quality signal. Re-sorting by date throws that away and
    # keeps the newest N instead of the best N.
    if not source.get("preserve_order"):
        rows.sort(key=lambda i: i.published or EPOCH, reverse=True)
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
    rows: list[Item] = []
    warnings: list[Warning] = []

    for source in sources:
        fetched, warning = fetch_feed(source, default_limit=per_feed, cutoff=cutoff)
        if warning:
            warnings.append(warning)
        for item in fetched:
            if item.url in seen:
                continue
            seen.add(item.url)
            rows.append(item)

    return [replace(item, id=f"i{n:02d}") for n, item in enumerate(rows, start=1)], warnings
