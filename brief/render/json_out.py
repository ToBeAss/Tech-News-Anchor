"""Machine-readable rendering — `--json`, for piping and inspection.

A renderer like any other, so it lives beside the terminal one rather than in
the CLI. Merged siblings are included: a consumer that wants to show the other
outlet's angle should not have to re-derive it.
"""

from __future__ import annotations

import json

from ..synth import Brief, Entry


def _node(entry: Entry) -> dict:
    return {
        "id": entry.item.id,
        "headline": entry.headline,
        "fact": entry.fact,
        "comment": entry.comment,
        "source": {"name": entry.item.source, "url": entry.item.url},
        "title": entry.item.title,
        "also_covered_by": [{"source": r.origin, "title": r.title, "url": r.url}
                            for r in entry.item.related],
    }


def as_json(brief: Brief) -> str:
    return json.dumps({
        "top_signal": [_node(e) for e in brief.top],
        "also_worth_knowing": [
            {"category": name, "entries": [_node(e) for e in entries]}
            for name, entries in brief.groups
        ],
        "video": _node(brief.video) if brief.video else None,
        "meta_note": brief.meta,
        "shortfall": brief.shortfall,
    }, indent=2, ensure_ascii=False)
