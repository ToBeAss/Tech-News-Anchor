"""Machine-readable rendering — `--json`, for piping, inspection, and replay.

The schema round-trips: `replay()` reconstructs enough of a Brief, plus the
considered/warnings/flagged context, to feed straight back into any renderer.
That's what `--replay` uses to iterate on Discord formatting without paying
for ingestion, gating, dedupe, synthesis, or verification.

Item.summary is dropped: nothing downstream of synthesis reads it (verify
already ran; it isn't re-run on replay), so there's nothing worth round-tripping
that a renderer would use.
"""

from __future__ import annotations

import json
from datetime import datetime

from ..sources import Item
from ..synth import Brief, Entry
from ..warnings import Warning


def _parse_published(raw: str | None) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


def _item_node(item: Item) -> dict:
    return {
        "id": item.id,
        "title": item.title,
        "url": item.url,
        "source": item.source,
        "origin": item.origin,
        # Renderers show each link's age (terminal.py/discord.py _origin_line) —
        # dropping this on replay would silently render stale-looking output
        # for a tool whose whole job is faithful formatting iteration.
        "published": item.published.isoformat() if item.published else None,
        "related": [
            {"title": r.title, "url": r.url, "source": r.source, "origin": r.origin,
             "published": r.published.isoformat() if r.published else None}
            for r in item.related
        ],
    }


def _entry_node(entry: Entry, flagged: dict[str, list[str]]) -> dict:
    node = {"headline": entry.headline, "comment": entry.comment, "item": _item_node(entry.item)}
    if entry.fact:
        node["fact"] = entry.fact
    bad = flagged.get(entry.item.id)
    if bad:
        node["flagged"] = bad
    return node


def _warning_node(warning: Warning) -> dict:
    return {"text": warning.text, "reader": warning.reader, "degraded": warning.degraded}


def as_json(brief: Brief, *, considered: int, warnings: list[Warning],
            flagged: dict[str, list[str]] | None = None,
            clusters: list[dict] | None = None) -> str:
    flagged = flagged or {}
    also_worth_knowing = (
        [{"category": name, "entries": [_entry_node(e, flagged) for e in entries]}
         for name, entries in brief.groups]
        if brief.groups else
        [_entry_node(e, flagged) for e in brief.also]
    )
    return json.dumps({
        "considered": considered,
        "warnings": [_warning_node(w) for w in warnings],
        # The dedupe cluster map, kept even for groups that never made the
        # brief — over-merge is only diagnosable with the full grouping
        # decision in front of you, not just whatever survived selection.
        # See dedupe._cluster_map. Not round-tripped by replay(): it's audit
        # data for a live run, not something any renderer displays.
        "dedupe_clusters": clusters or [],
        "top_signal": [_entry_node(e, flagged) for e in brief.top],
        "also_worth_knowing": also_worth_knowing,
        "video": _entry_node(brief.video, flagged) if brief.video else None,
        "meta_note": brief.meta,
        "shortfall": brief.shortfall,
    }, indent=2, ensure_ascii=False)


def _item_from_node(node: dict) -> Item:
    return Item(
        id=node.get("id", ""),
        title=node.get("title", ""),
        summary="",           # not needed for rendering; verify does not re-run
        url=node["url"],
        source=node.get("source", ""),
        origin=node.get("origin") or node.get("source", ""),
        kind="article",       # rendering never branches on kind; see synth/render
        published=_parse_published(node.get("published")),
        related=tuple(_item_from_node(r) for r in node.get("related", [])),
    )


def _entry_from_node(node: dict, flagged: dict[str, list[str]]) -> Entry:
    entry = Entry(item=_item_from_node(node["item"]), headline=node["headline"],
                 comment=node.get("comment", ""), fact=node.get("fact", ""))
    if node.get("flagged"):
        flagged[entry.item.id] = node["flagged"]
    return entry


def _warning_from_node(node: dict) -> Warning:
    return Warning(text=node["text"], reader=bool(node.get("reader")),
                   degraded=bool(node.get("degraded")))


def replay(text: str):
    """Reconstruct (brief, considered, warnings, flagged) from as_json() output.

    Enough of a Brief to feed to_terminal/to_discord — not a general parser for
    hand-edited JSON. `flagged` is whatever was recorded at the original run,
    since --replay skips verification entirely rather than re-deriving it.
    """
    data = json.loads(text)
    flagged: dict[str, list[str]] = {}

    top = [_entry_from_node(n, flagged) for n in data.get("top_signal") or []]

    raw_also = data.get("also_worth_knowing") or []
    if raw_also and "category" in raw_also[0]:
        groups = tuple(
            (g["category"], tuple(_entry_from_node(n, flagged) for n in g["entries"]))
            for g in raw_also
        )
        also = [e for _name, entries in groups for e in entries]
    else:
        also = [_entry_from_node(n, flagged) for n in raw_also]
        groups = ()

    video_node = data.get("video")
    video = _entry_from_node(video_node, flagged) if video_node else None

    brief = Brief(top=top, also=also, video=video, meta=data.get("meta_note"),
                 shortfall=data.get("shortfall"), groups=groups)
    warnings = [_warning_from_node(w) for w in (data.get("warnings") or [])]
    return brief, data.get("considered", 0), warnings, flagged
