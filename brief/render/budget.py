"""Shared rendering backstop — one value, not one per renderer.

terminal.py and discord.py both clip a tier-two comment to roughly the same
length because they're both approximating the same content budget
(synth.ALSO_COMMENT_WORDS), not because either medium has its own independent
readability constraint. Letting the two literals drift apart is a bug you
wouldn't notice until a comment reads differently in the channel than in the
terminal.

Set above synth.ALSO_COMMENT_WORDS in characters, not just words: a comment
heavy on long abstract nouns ("infrastructure", "procurement", "contribution")
can hit a char limit well under its word limit, so this must clear the widest
plausible rendering of the word cap, not the average one.

clip/date_title/origin_line/degraded/footer_text below are shared across all
three renderers (terminal, Discord, Slack): they reference nothing target-
specific, only plain strings and the Item/Warning shapes every renderer
already consumes. A silently diverged _origin_line, or a footer that forgets
which warnings are reader-facing on only one of the three targets, is exactly
the drift this module exists to prevent.

MAX_RELATED_SHOWN is the same idea applied to a different budget: a merged
cluster of 6+ outlets produced six `also:` link lines under one headline,
which drowns the headline it's attached to. Shared because the problem is
identical in all three renderers, not medium-specific.
"""

from __future__ import annotations

from datetime import datetime

from ..sources import Item
from ..warnings import Warning

ALSO_MAX_CHARS = 220

# Tier-one's full `also:` list, capped. Rendering cap only — `related` itself
# is never truncated, so memory's coverage and --json's output stay complete;
# see memory._urls() and the replay round-trip test that pins this down.
MAX_RELATED_SHOWN = 3


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.—- ") + "…"


def date_title(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%A} {now.day} {now:%B %Y}"


def origin_line(item: Item) -> str:
    return " · ".join(p for p in (item.origin, item.age()) if p)


def degraded(warnings: list[Warning]) -> list[Warning]:
    """Warnings a lost source produced. A field on the Warning, not a sniff
    for "UNREACHABLE" in its text — the old substring check silently stopped
    matching the moment anyone reworded the message."""
    return [w for w in warnings if w.degraded]


def footer_text(considered: int, kept: int, shortfall: str | None,
                warnings: list[Warning]) -> str:
    """Reader-facing warnings only — degraded-source notices already have
    their own banner above the content, and everything else operator-only
    (UNVERIFIED FIGURE, OFF-LEAD *, OVER BUDGET, gate/dedupe diagnostics) has
    no business in a channel the brief's reader actually reads."""
    parts = [f"{considered} items considered · {kept} kept"]
    if shortfall:
        parts.append(f"under target: {shortfall}")
    parts += [w.text for w in warnings if w.reader]
    return " · ".join(parts)
