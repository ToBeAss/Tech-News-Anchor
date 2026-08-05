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


def _entry(entry: Entry, *, terse: bool = False) -> str:
    lines = [f" {BOLD(entry.headline)}"]
    if entry.comment:
        lines.append(_wrap(entry.comment))
    lines.append(DIM(f"   {entry.item.source} · ") + CYAN(entry.item.url))
    return "\n".join(lines) + ("" if terse else "\n")


def to_terminal(brief: Brief, *, considered: int, warnings: list[str]) -> str:
    out = [
        "",
        BOLD(f"  TECH BRIEF  ") + DIM(datetime.now().strftime("%A %d %B %Y, %H:%M")),
        DIM("═" * _width()),
    ]

    if brief.top:
        out.append(_rule("🔥 TOP SIGNAL"))
        out += [_entry(e) for e in brief.top]

    if brief.also:
        out.append(_rule("📎 ALSO WORTH KNOWING"))
        out += [_entry(e, terse=True) for e in brief.also]
        out.append("")

    if brief.video:
        out.append(_rule("🎥 VIDEO"))
        out.append(_entry(brief.video))

    if brief.meta:
        out.append(_rule("💭 META"))
        out.append(_wrap(brief.meta, indent=" "))
        out.append("")

    out.append(DIM("─" * _width()))
    out.append(DIM(f"  {considered} items considered · "
                   f"{len(brief.top) + len(brief.also) + (1 if brief.video else 0)} kept"))
    for warning in warnings:
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
