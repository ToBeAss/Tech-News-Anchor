"""Near-duplicate clustering — a cheap LLM pre-pass before synthesis.

Why this isn't lexical: kode24 headlined the Tydal datacentre story
"Avslørt: Anthropic flytter trolig inn i Trøndelag" while digi.no ran
"Tydal Data Center: Inngår kontrakt verdt 55 milliarder kroner". Same event,
zero shared title words, two different framings, one of them in a language the
other doesn't name. Token overlap cannot see that. Attempts to force it either
miss the pair or over-merge unrelated stories that happen to share a common
noun like "data centre".

So this is a separate, small call that does one job: group ids covering the
same underlying event. Splitting it out also takes load off the synthesis call,
which was previously asked to rank, deduplicate, and write in one pass — and
the deduplication was the part that kept slipping.

Failure is non-fatal: if the call errors or returns junk, the candidate list
passes through unclustered and the synthesis prompt's own dedup rule remains as
a second line of defence.
"""

from __future__ import annotations

from dataclasses import replace

from . import llm
from .sources import Item

_SYSTEM = """You group news items that cover the SAME underlying event.

You are given numbered items, each with an id, source, and headline. Some \
describe the same event from different outlets, in different languages, or \
from different angles (e.g. one reports a contract being signed, another names \
the likely customer in that same contract).

Group ONLY items that a reader would consider the same story. Do NOT group \
items that merely share a topic, a company, or a theme. Two separate attacks \
on the same organisation are one story if they are the same reported incident, \
and two stories if they are distinct events. A local instance of a global \
trend is NOT the same story as the global one — do not group them.

Output STRICT JSON, nothing else — no prose, no markdown fences:
{"groups": [["i03", "i17"], ["i08", "i22", "i30"]]}

Only include groups of 2 or more. If nothing should be grouped, output \
{"groups": []}. Never invent ids."""


def _valid_groups(data: dict, index: dict[str, Item]) -> list[list[str]]:
    """Keep only groups of 2+ known, not-yet-claimed ids."""
    raw_groups = data.get("groups") or []
    if not isinstance(raw_groups, list):
        return []
    claimed: set[str] = set()
    groups: list[list[str]] = []
    for group in raw_groups:
        if not isinstance(group, list):
            continue
        ids = [str(i).strip() for i in group]
        ids = [i for i in ids if i in index and i not in claimed]
        if len(ids) >= 2:
            claimed.update(ids)
            groups.append(ids)
    return groups


def merge(items: list[Item], *, model=None, max_output_tokens: int = 500):
    """Return (items, note). Items in a group collapse into the first, with the
    rest attached as `related` so their links survive into the brief."""
    if len(items) < 2:
        return items, None

    listing = "\n".join(f"{i.id} ({i.origin}) {i.title}" for i in items)
    try:
        raw = llm.generate(
            [{"role": "user", "content": listing}],
            instructions=_SYSTEM,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        groups = _valid_groups(llm.parse_json(raw), {i.id: i for i in items})
    except Exception as exc:
        return items, f"dedupe skipped: {exc}"

    if not groups:
        return items, None

    index = {i.id: i for i in items}
    followers: dict[str, list[Item]] = {g[0]: [index[i] for i in g[1:]] for g in groups}
    absorbed = {i for group in groups for i in group[1:]}

    merged: list[Item] = []
    for item in items:
        if item.id in absorbed:
            continue
        extra = followers.get(item.id)
        merged.append(replace(item, related=tuple(extra)) if extra else item)

    collapsed = len(items) - len(merged)
    return merged, (f"merged {collapsed} duplicate item(s) into "
                    f"{len(groups)} stor{'y' if len(groups) == 1 else 'ies'}")
