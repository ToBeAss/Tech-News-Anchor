"""Synthesis — raw items in, ranked+annotated brief out.

The integrity contract: the model receives items labelled [i01], [i02]… with no
URLs, and returns only those ids. Anything referencing an id we didn't send is
dropped at validation. The model therefore cannot fabricate, mangle, or
mis-attribute a source link — it never handles one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import llm
from sources import Item


class SynthError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    item: Item
    headline: str
    comment: str


@dataclass(frozen=True)
class Brief:
    top: list[Entry]
    also: list[Entry]
    video: Entry | None
    meta: str | None


_SYSTEM = """You are a technical editor writing a daily tech brief for one \
engineer. You are given today's candidate stories, each tagged with an id.

Your job is NOT to summarise everything. It is to filter hard, then say \
something worth reading about what survives.

The reader's interests:
{interests}

Rules:
- Rank by relevance to those interests. Most items should not make the brief. \
An empty section is a valid, honest outcome — never pad to hit a count.
- Commentary must earn its place: why this matters, what it conflicts with, \
what's overhyped, what the second-order effect is. Never restate the headline \
in different words. Never write filler like "this is significant as AI \
advances rapidly". If you have no real take on an item, demote it to \
also_worth_knowing or drop it.
- Have opinions. Skepticism is welcome. Flag vendor announcements as vendor \
announcements.
- Write your own headline for each entry — terse, concrete, no clickbait.
- meta_note is optional: use it only when you notice a genuine cross-story \
pattern (multiple sources circling the same underlying shift). Otherwise null.
- If a video item is present and worth mentioning, put ONE in "video".
- Do not invent stories. Only use ids from the supplied list.

Output STRICT JSON, nothing else — no prose, no markdown fences:
{{
  "top_signal": [{{"id": "iNN", "headline": "...", "comment": "..."}}],
  "also_worth_knowing": [{{"id": "iNN", "headline": "...", "comment": "..."}}],
  "video": {{"id": "iNN", "headline": "...", "comment": "..."}} or null,
  "meta_note": "..." or null
}}

Sizing: top_signal 0-3 entries, comment 2-3 sentences. also_worth_knowing 0-6 \
entries, comment one line. Be terse everywhere."""


def build(items: list[Item], interests: str, *, model=None, temperature=None,
          max_output_tokens=None) -> Brief:
    if not items:
        raise SynthError("no items to synthesise")

    catalogue = "\n\n".join(i.for_prompt() for i in items)
    raw = llm.generate(
        [{"role": "user", "content": f"Today's candidates:\n\n{catalogue}"}],
        instructions=_SYSTEM.format(interests=interests.strip()),
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return _validate(_parse(raw), {i.id: i for i in items})


def _parse(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise SynthError(f"no JSON object in output: {raw[:200]}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise SynthError(f"JSON parse failed: {exc}: {raw[:200]}")


def _entry(node: Any, index: dict[str, Item]) -> Entry | None:
    """Resolve one node to an Entry, or None if it references an unknown id."""
    if not isinstance(node, dict):
        return None
    item = index.get(str(node.get("id", "")).strip())
    if item is None:
        return None
    return Entry(
        item=item,
        headline=str(node.get("headline") or item.title).strip(),
        comment=str(node.get("comment") or "").strip(),
    )


def _section(data: dict, key: str, index: dict[str, Item]) -> list[Entry]:
    nodes = data.get(key) or []
    if not isinstance(nodes, list):
        return []
    seen: set[str] = set()
    out: list[Entry] = []
    for node in nodes:
        entry = _entry(node, index)
        if entry and entry.item.id not in seen:
            seen.add(entry.item.id)
            out.append(entry)
    return out


def _validate(data: dict, index: dict[str, Item]) -> Brief:
    top = _section(data, "top_signal", index)
    also = [e for e in _section(data, "also_worth_knowing", index)
            if e.item.id not in {t.item.id for t in top}]
    video = _entry(data.get("video"), index)
    if video and video.item.id in {e.item.id for e in top + also}:
        video = None

    meta = data.get("meta_note")
    meta = meta.strip() if isinstance(meta, str) and meta.strip() else None

    if not (top or also or video):
        raise SynthError("model returned no resolvable entries")
    return Brief(top=top, also=also, video=video, meta=meta)
