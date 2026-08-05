"""Relevance gating — per-source LLM filter for feeds that are mostly off-topic.

Some sources carry occasional signal buried in unrelated output. PewDiePie's
channel is the motivating case: a programming or local-LLM video is worth the
team's attention, a gaming video is not, and the titles rarely say which is
which ("I Tried This For 30 Days" could be either). Keyword matching cannot
separate them, and raw inclusion would flood the candidate pool.

So a gated source gets one small LLM call: here are the titles and
descriptions, here is what counts as relevant, return the ids that pass. The
gate runs at ingestion, before dedup and synthesis, so rejected items never
consume a candidate slot.

Gating is fail-open. If the call fails, items pass through with a warning
rather than vanishing — synthesis ranks by relevance too, so an off-topic item
that slips past the gate is very likely dropped there anyway, whereas a
silently discarded item is gone for good.
"""

from __future__ import annotations

import json

import llm
from sources import Item

_SYSTEM = """You decide which items from a noisy feed are relevant.

You are given items with an id, title, and description, plus a relevance rule. \
Return the ids that satisfy the rule.

Judge on the description as much as the title \u2014 titles are often vague or \
clickbait, and the description usually reveals the actual subject. When the \
evidence is genuinely ambiguous, KEEP the item: a later ranking step will \
drop it if it turns out to be weak, but an item rejected here is gone.

Relevance rule:
{rule}

Output STRICT JSON, nothing else \u2014 no prose, no markdown fences:
{{"keep": ["i04", "i11"]}}

If nothing qualifies, output {{"keep": []}}. Never invent ids."""


def _parse_keep(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object: {raw[:200]}")
    keep = json.loads(text[start:end + 1]).get("keep")
    if not isinstance(keep, list):
        raise ValueError("'keep' is not a list")
    return [str(i).strip() for i in keep]


def apply(items: list[Item], gates: dict[str, str], *, model=None):
    """Filter items whose source has a gate rule. Returns (items, warnings)."""
    if not gates:
        return items, []

    warnings: list[str] = []
    survivors: list[Item] = [i for i in items if i.source not in gates]

    for source, rule in gates.items():
        candidates = [i for i in items if i.source == source]
        if not candidates:
            continue

        listing = "\n\n".join(
            f"[{i.id}] {i.title}\n{i.summary or '(no description)'}" for i in candidates
        )
        try:
            raw = llm.generate(
                [{"role": "user", "content": listing}],
                instructions=_SYSTEM.format(rule=rule.strip()),
                model=model,
                max_output_tokens=300,
            )
            keep = set(_parse_keep(raw))
        except Exception as exc:
            warnings.append(f"{source}: gate failed, keeping all ({exc})")
            survivors.extend(candidates)
            continue

        passed = [i for i in candidates if i.id in keep]
        survivors.extend(passed)
        dropped = len(candidates) - len(passed)
        if dropped:
            warnings.append(f"{source}: gate dropped {dropped}/{len(candidates)} as off-topic")

    # Preserve the original ingestion order; ids are reassigned by the caller.
    order = {item.id: n for n, item in enumerate(items)}
    survivors.sort(key=lambda i: order[i.id])
    return survivors, warnings