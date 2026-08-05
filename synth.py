"""Synthesis — raw items in, ranked+annotated brief out.

The integrity contract: the model receives items labelled [i01], [i02]… with no
URLs, and returns only those ids. Anything referencing an id we didn't send is
dropped at validation. The model therefore cannot fabricate, mangle, or
mis-attribute a source link — it never handles one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
    fact: str = ""      # what happened; empty for tier two, which carries it in the headline


# Fixed shape: the brief is always 3 + 5. Locked counts make the output
# predictable to skim — you learn where to look. The cost is that on a thin
# day the last slots are the best of a weak field rather than genuinely
# strong; the commentary is expected to say so rather than oversell.
TARGET_TOP = 3
TARGET_ALSO = 5


@dataclass(frozen=True)
class Brief:
    top: list[Entry]
    also: list[Entry]
    video: Entry | None
    meta: str | None
    shortfall: str | None = None   # set when a section came back under target


_SYSTEM = """You are a technical editor writing a daily tech brief for one \
engineer. You are given today's candidate stories, each tagged with an id.

Your job is NOT to summarise everything. It is to filter hard, then say \
something worth reading about what survives.

Who this is for:
{audience}

The reader's interests:
{interests}

Priority overrides:
{priorities}

## Selection
- Rank ALL candidates by relevance to those interests, then take the top 8. \
This is a ranking task with a fixed output size, not a threshold test.
- The 3 strongest go in top_signal, the next 5 in also_worth_knowing. The \
split is by strength, so anything in top_signal must be more important than \
everything in also_worth_knowing.
- You must fill both sections. If the day is thin, the weakest entries are \
the best of a weak field \u2014 say so plainly in the commentary rather than \
inflating them. "Minor, but the only movement on X this week" is an honest \
line; pretending a slow news day is significant is not.
- Prefer spread across sources. If one source dominates the candidates by \
volume, that is an artefact of its publishing rate, not importance.
- Video candidates are tracked separately and get their own reserved slot. Do \
not skip a video because the written stories are stronger \u2014 it is not \
competing with them. If a video is strong enough to lead the whole brief, it \
may go in top_signal instead, but then it must not also appear in video.
- ONE ENTRY PER UNDERLYING STORY. Several candidates often cover the same \
event from different outlets or angles. Pick the single best version and drop \
the rest \u2014 do not spend a second slot on the same story because another \
source framed it differently. If two angles are both worth having, merge them \
into one entry's commentary rather than listing both.
- Apply the priority overrides above as real ranking weight, not a tiebreak. \
Do not discount an item because it is short, in another language, or from a \
smaller outlet \u2014 a local instance of a global story usually matters more \
to this reader than the global one. If you include the global version, \
prefer the local version instead where both are present.
- Use the audience context to judge SECOND-ORDER relevance \u2014 what this \
reader will have to reason about, argue for, or build in the next year. A \
story about another country's public-sector procurement, a standards change, \
or a regulatory precedent can be highly relevant without naming their \
organisation or sector.
- Do NOT turn the audience context into a topic filter. Their field is a lens, \
not a subject requirement. Never force a connection to their organisation into \
the commentary, and never pick a weak item because it is superficially \
on-sector over a strong one that is not.
- Never invent stories. Only use ids from the supplied list.

## Writing
- Every top_signal entry has TWO parts, and both are required:
  * "fact" \u2014 what actually happened, in one sentence. Name the actor and \
carry the specific detail: the number, the version, the claim, the decision. \
A reader who only reads this sentence should know the news. Do not editorialise \
here.
    MERGED ITEMS: when an item is marked "SAME STORY, also covered by", the \
other coverage may describe a DIFFERENT PART of the same chain \u2014 a \
different contract, party, or layer of the same deal. Do not merge their \
figures or attach one source's party to another source's number. Attribute \
every figure to the specific parties it belongs to, and if the layers matter, \
name them ("A's $10bn deal with B, which in turn contracted C for $4.7bn"). \
Two numbers that look contradictory are usually two different agreements.
    HEDGING: preserve the source's certainty exactly. If it says "trolig", \
"reportedly", "says it has", "is believed to", or attributes a claim to \
someone, your sentence must carry the same hedge and the same attribution. \
"Anthropic signed" is wrong where the source said Anthropic is probably the \
customer; "reporting points to Anthropic" is right. Stripping a hedge turns a \
claim into a fabrication even when every word came from the source.
    GROUNDING (absolute): state ONLY details that appear in the supplied title \
or summary for that item. You are working from a headline and a short extract, \
not the article. If a figure, version, date, name or list of provisions is not \
in the text you were given, you do NOT know it \u2014 do not supply it, do not \
infer it, do not reconstruct it from memory. Write around the gap instead: \
"a multi-billion-kroner contract" is correct where the amount is absent, and a \
confident wrong number is a serious failure. The same applies to headlines and \
to any figure in a comment.
  * "comment" \u2014 why it matters, what it conflicts with, the second-order \
effect. Do not restate the fact. Do not write filler like "this is significant \
as AI advances rapidly".
- If you cannot state a concrete fact, you do not understand the item well \
enough to rank it highly \u2014 move it down.
- Have opinions. Skepticism is welcome. Flag vendor announcements as vendor \
announcements \u2014 the origin domain tells you when a story is a company \
talking about itself.
- Write your own headline for each entry \u2014 terse, concrete, no clickbait. \
Headlines must be FACTUAL, not thematic: name who did what. "Norway funds AI \
centres it cannot staff, says digi.no op-ed" is right; "Norway's AI talent \
bottleneck resurfaces" is wrong \u2014 it names a theme and tells the reader \
nothing. This matters most in also_worth_knowing, where the headline carries \
the news because the one-line comment has no room for it.
- Always write in English, whatever language the source is in. Where a source \
is non-English, do not let a short or untranslated summary count against the \
item's importance.

## Budget (hard limits \u2014 the brief must stay under a 3 minute read)
- top_signal: EXACTLY 3 entries. Not 2, not 4. fact 20-30 words (one \
sentence). comment 40-55 words. This tier gets the depth.
- also_worth_knowing: EXACTLY 5 entries. comment MAX 25 WORDS \u2014 count them. One \
sentence, and it must END, not trail off. This tier is scannable, not \
readable. If an item needs more than one line to justify, it belongs in \
top_signal or nowhere.
- video: REQUIRED whenever any candidate is marked (video). Pick the single \
best one and put it here. comment max 35 words. This slot is reserved \u2014 it \
does not compete with the other two sections, and a video appearing here does \
NOT consume a top_signal or also_worth_knowing slot. Only output null when \
there is genuinely no video candidate in the list.
- meta_note: max 50 words, or null. Use it ONLY for a genuine cross-story \
pattern (several sources circling the same underlying shift). Not a summary \
of the brief.

Output STRICT JSON, nothing else \u2014 no prose, no markdown fences:
{{
  "top_signal": [{{"id": "iNN", "headline": "...", "fact": "...", "comment": "..."}}],
  "also_worth_knowing": [{{"id": "iNN", "headline": "...", "comment": "..."}}],
  "video": {{"id": "iNN", "headline": "...", "fact": "...", "comment": "..."}} or null,
  "meta_note": "..." or null
}}"""


def build(items: list[Item], interests: str, priorities: str = "",
          audience: str = "", *, model=None, temperature=None,
          max_output_tokens=None) -> Brief:
    if not items:
        raise SynthError("no items to synthesise")

    catalogue = "\n\n".join(i.for_prompt() for i in items)
    raw = llm.generate(
        [{"role": "user", "content": f"Today's candidates:\n\n{catalogue}"}],
        instructions=_SYSTEM.format(
            audience=(audience or "(not specified)").strip(),
            interests=interests.strip(),
            priorities=(priorities or "(none)").strip(),
        ),
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
        fact=str(node.get("fact") or "").strip(),
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
    # Caps enforced here, not only in the prompt: an over-long section is a
    # silent quality regression, and truncating is better than trusting.
    top = _section(data, "top_signal", index)[:TARGET_TOP]
    also = [e for e in _section(data, "also_worth_knowing", index)
            if e.item.id not in {t.item.id for t in top}][:TARGET_ALSO]

    # Backfill: if the model under-filled top_signal, promote from also rather
    # than shipping a short tier. Only shifts placement, never invents entries.
    while len(top) < TARGET_TOP and also:
        top.append(also.pop(0))
    video = _entry(data.get("video"), index)
    if video and video.item.id in {e.item.id for e in top + also}:
        video = None

    # Reserved video slot: if the model ignored it despite a video candidate
    # existing, surface the newest one as a bare pointer rather than dropping it.
    if video is None:
        placed = {e.item.id for e in top + also}
        spare = [i for i in index.values() if i.kind == "video" and i.id not in placed]
        if spare:
            spare.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc),
                       reverse=True)
            video = Entry(item=spare[0], headline=spare[0].title, comment="")

    meta = data.get("meta_note")
    meta = meta.strip() if isinstance(meta, str) and meta.strip() else None

    if not (top or also or video):
        raise SynthError("model returned no resolvable entries")

    shortfall = []
    if len(top) < TARGET_TOP:
        shortfall.append(f"top_signal {len(top)}/{TARGET_TOP}")
    if len(also) < TARGET_ALSO:
        shortfall.append(f"also_worth_knowing {len(also)}/{TARGET_ALSO}")

    return Brief(top=top, also=also, video=video, meta=meta,
                 shortfall=", ".join(shortfall) or None)