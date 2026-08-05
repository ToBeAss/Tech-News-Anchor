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

from ..sources import Item
from ..synth import Brief, Entry


def _item_node(item: Item) -> dict:
    return {
        "id": item.id,
        "title": item.title,
        "url": item.url,
        "source": item.source,
        "origin": item.origin,
        "related": [
            {"title": r.title, "url": r.url, "source": r.source, "origin": r.origin}
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


def as_json(brief: Brief, *, considered: int, warnings: list[str],
            flagged: dict[str, list[str]] | None = None) -> str:
    flagged = flagged or {}
    also_worth_knowing = (
        [{"category": name, "entries": [_entry_node(e, flagged) for e in entries]}
         for name, entries in brief.groups]
        if brief.groups else
        [_entry_node(e, flagged) for e in brief.also]
    )
    return json.dumps({
        "considered": considered,
        "warnings": warnings,
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
        published=None,
        related=tuple(_item_from_node(r) for r in node.get("related", [])),
    )


def _entry_from_node(node: dict, flagged: dict[str, list[str]]) -> Entry:
    entry = Entry(item=_item_from_node(node["item"]), headline=node["headline"],
                 comment=node.get("comment", ""), fact=node.get("fact", ""))
    if node.get("flagged"):
        flagged[entry.item.id] = node["flagged"]
    return entry


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
    return brief, data.get("considered", 0), data.get("warnings") or [], flagged
