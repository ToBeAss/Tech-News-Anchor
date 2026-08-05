"""Terminal rendering.

Deliberately isolated from synthesis so iteration 2 can add discord.py /
teams.py renderers against the same Brief object without touching the pipeline.
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime

from synth import Brief, Entry

_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


BOLD = lambda t: _c("1", t)
DIM = lambda t: _c("2", t)
CYAN = lambda t: _c("36", t)
YELLOW = lambda t: _c("33", t)


def _width() -> int:
    return min(shutil.get_terminal_size((100, 24)).columns, 100)


def _wrap(text: str, indent: str = "   ") -> str:
    import textwrap
    return textwrap.fill(
        text, width=_width(), initial_indent=indent, subsequent_indent=indent
    )


def _rule(label: str) -> str:
    return "\n" + BOLD(label) + "\n" + DIM("─" * _width())


# ~25 words. The renderer clip is a backstop, not the budget: it sits slightly
# above the prompt limit so a well-behaved entry is never cut mid-thought, and
# only genuine overruns get truncated.
ALSO_MAX_CHARS = 190


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:.\u2014- ") + "\u2026"


def _entry(entry: Entry, *, terse: bool = False, flags: list[str] | None = None) -> str:
    """terse = tier two: one clipped line, so the hierarchy stays visible."""
    lines = [f" {BOLD(entry.headline)}"]
    if terse:
        if entry.comment:
            lines.append(_wrap(_clip(entry.comment, ALSO_MAX_CHARS)))
    else:
        # Fact and take are separate fields so the model must produce both, but
        # they're joined here — a reader wants a paragraph, not a labelled form.
        body = " ".join(p for p in (entry.fact, entry.comment) if p)
        if body:
            lines.append(_wrap(body))
    lines.append(DIM(f"   {entry.item.origin} \u00b7 ") + CYAN(entry.item.url))
    # Merged duplicates keep their links: the other outlet's angle is often
    # worth reading even though it doesn't deserve its own slot.
    for origin, _title, url in entry.item.related:
        lines.append(DIM(f"   also {origin} \u00b7 ") + CYAN(url))
    if flags:
        # Marked inline, next to the claim, not only in a footer — an unverified
        # figure must be visible at the moment it is read.
        lines.append(YELLOW(f"   \u26a0 unverified figure(s): {', '.join(flags)} "
                            "\u2014 check the source before quoting"))
    return "\n".join(lines) + ("" if terse else "\n")


def _degraded(warnings: list[str]) -> list[str]:
    """Sources that were lost this run, as opposed to merely quiet."""
    return [w for w in warnings if "UNREACHABLE" in w]


def to_terminal(brief: Brief, *, considered: int, warnings: list[str],
                flagged: dict[str, list[str]] | None = None) -> str:
    flagged = flagged or {}
    lost = _degraded(warnings)
    out = [
        "",
        BOLD(f"  TECH BRIEF  ") + DIM(datetime.now().strftime("%A %d %B %Y, %H:%M")),
        DIM("═" * _width()),
    ]

    # A lost source means the brief was built from a smaller pool than intended.
    # That belongs at the top, where it can be read before the content, not in a
    # footer the reader reaches after already trusting the ranking.
    if lost:
        out.append(YELLOW(f"  ⚠ DEGRADED — {len(lost)} source(s) unreachable; "
                          "ranking drew on a reduced pool"))
        for warning in lost:
            out.append(YELLOW(f"      {warning}"))
        out.append("")

    if brief.top:
        out.append(_rule("🔥 TOP SIGNAL"))
        out += [_entry(e, flags=flagged.get(e.item.id)) for e in brief.top]

    if brief.also:
        out.append(_rule("📎 ALSO WORTH KNOWING"))
        if brief.groups:
            # Heading sits left of its entries so the eye can drop down the
            # labels and stop where it wants — the whole point of grouping.
            for name, entries in brief.groups:
                out.append(DIM(f"  ── {name.upper()}"))
                out += [_entry(e, terse=True, flags=flagged.get(e.item.id))
                        for e in entries]
                out.append("")
        else:
            out += [_entry(e, terse=True, flags=flagged.get(e.item.id)) for e in brief.also]
            out.append("")

    if brief.video:
        out.append(_rule("🎥 VIDEO"))
        out.append(_entry(brief.video, flags=flagged.get(brief.video.item.id)))

    if brief.meta:
        out.append(_rule("💭 META"))
        out.append(_wrap(brief.meta, indent=" "))
        out.append("")

    out.append(DIM("─" * _width()))
    out.append(DIM(f"  {considered} items considered · "
                   f"{len(brief.top) + len(brief.also) + (1 if brief.video else 0)} kept"))
    if brief.shortfall:
        out.append(YELLOW(f"  ⚠ under target: {brief.shortfall}"))
    for warning in warnings:
        if warning not in lost:
            out.append(YELLOW(f"  ⚠ {warning}"))
    out.append("")
    return "\n".join(out)


def raw_listing(items, warnings: list[str]) -> str:
    """--dry output: what ingestion found, before any LLM call."""
    out = ["", BOLD(f"  {len(items)} items ingested"), DIM("═" * _width())]
    current = None
    for item in items:
        if item.source != current:
            current = item.source
            out.append("\n" + BOLD(current))
        stamp = item.published.strftime("%m-%d %H:%M") if item.published else "  --  "
        out.append(f"  {DIM(item.id)} {DIM(stamp)}  {item.title}")
        out.append(f"        {CYAN(item.url)}")
    for warning in warnings:
        out.append(YELLOW(f"\n  ⚠ {warning}"))
    out.append("")
    return "\n".join(out)