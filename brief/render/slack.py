"""Slack Block Kit rendering — pure, no network.

Mirrors discord.py's contract (a Brief in, ready-to-POST payloads out), but not
its shape: Block Kit is a flat `blocks` array with no per-block colour, mrkdwn
syntax differs from Discord markdown, and there is no colour channel at all —
state Discord expresses as embed colour (degraded, horizon, further afield)
has to become text or a dedicated block type here instead. That's why this is
a second renderer rather than a shared abstraction over both: the structural
gap costs more to paper over than two separate, readable modules do.

Reuses budget.py's clip/date_title/origin_line/degraded/footer_text/
MAX_RELATED_SHOWN — those are target-agnostic and discord.py shares them too.

Section names (Today's Signal / On the Horizon / Further Afield) are the whole
lighthouse-metaphor budget for this brand — do not extend it into block names,
the header, the context block, or variable/log names.
"""

from __future__ import annotations

from ..synth import Brief, Entry
from ..warnings import Warning
from .budget import (ALSO_MAX_CHARS, MAX_RELATED_SHOWN, clip, date_title, degraded,
                     footer_text, origin_line)

# Slack Block Kit limits (enforced below, not hoped for).
MAX_BLOCKS_PER_MESSAGE = 50
MAX_SECTION_TEXT = 3000          # per text object, not per message
MAX_HEADER_TEXT = 150            # Slack's own limit on a header block's plain_text


def _escape(text: str) -> str:
    # '&' first, or the '&' introduced by escaping '<'/'>' would itself get
    # escaped on a second pass.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_link(text: str, url: str) -> str:
    # _escape() already neutralises a raw '>' (-> '&gt;'), so it can't close the
    # link early; '|' has no entity form and would still end the link text
    # there, so it gets substituted — same spirit as discord._md_link swapping
    # out '[' and ']' rather than trying to escape markdown syntax characters.
    return f"<{url}|{_escape(text).replace('|', '/')}>"


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:MAX_SECTION_TEXT]}}


def _context(text: str) -> dict:
    return {"type": "context",
           "elements": [{"type": "mrkdwn", "text": text[:MAX_SECTION_TEXT]}]}


def _header() -> dict:
    return {"type": "header",
           "text": {"type": "plain_text",
                    "text": clip(f"Fyrtårn — {date_title()}", MAX_HEADER_TEXT)}}


def _degraded_block(lost: list[Warning]) -> dict:
    text = (f"⚠️ *Degraded run:* {len(lost)} source(s) unreachable — "
            "ranking drew on a reduced pool\n"
            + "\n".join(f"• {_escape(w.text)}" for w in lost))
    return _section(text)


def _top_block(entry: Entry, flagged: dict[str, list[str]]) -> str:
    lines = [f"*{_md_link(entry.headline, entry.item.url)}*"]
    # Fact and comment are separate fields so the model must produce both, but
    # they read as one paragraph — same join as terminal.py's _entry().
    body = " ".join(p for p in (entry.fact, entry.comment) if p)
    if body:
        lines.append(_escape(body))
    lines.append(f"_{_escape(origin_line(entry.item))}_")
    # Capped: a 6-outlet merge produced six link lines under one headline,
    # drowning it. See budget.MAX_RELATED_SHOWN.
    shown = entry.item.related[:MAX_RELATED_SHOWN]
    lines += [f"also: {_md_link(origin_line(r), r.url)}" for r in shown]
    remaining = len(entry.item.related) - len(shown)
    if remaining:
        lines.append(f"+{remaining} more")
    bad = flagged.get(entry.item.id)
    if bad:
        # Inline, next to the claim it belongs to — not collected somewhere the
        # reader reaches after already trusting the fact.
        lines.append(f"⚠️ unverified: {_escape(', '.join(bad))}")
    return "\n".join(lines)


def _horizon_line(entry: Entry, flagged: dict[str, list[str]]) -> str:
    head = f"*{_md_link(entry.headline, entry.item.url)}*"
    text = (f"{head} — {_escape(clip(entry.comment, ALSO_MAX_CHARS))}"
            if entry.comment else head)
    # A trailing count, not a link per sibling — in Slack's tighter spacing
    # an "also:" line on its own floats ambiguously between entries. Tier two
    # is a scan surface; the count is enough.
    if entry.item.related:
        text += f" (+{len(entry.item.related)})"
    bad = flagged.get(entry.item.id)
    if bad:
        text += f" ⚠️ {_escape(', '.join(bad))}"
    return text


def _today_blocks(brief: Brief, flagged: dict[str, list[str]]) -> list[dict]:
    blocks = [_section("*Today's Signal*")]
    blocks += [_section(_top_block(e, flagged)) for e in brief.top]
    return blocks


def _horizon_chunks(brief: Brief, flagged: dict[str, list[str]]) -> list[str]:
    if brief.groups:
        return [
            "*{}*\n{}".format(_escape(name),
                              "\n".join(_horizon_line(e, flagged) for e in entries))
            for name, entries in brief.groups
        ]
    return [_horizon_line(e, flagged) for e in brief.also]


def _horizon_blocks(brief: Brief, flagged: dict[str, list[str]]) -> list[dict]:
    """Normally one section block after the heading. Splits on a category (or
    entry, flat) boundary if packing them together would overflow a single
    text object's 3000 chars — never mid-category, never mid-entry, matching
    the section-boundary rule discord.py already follows for its own limit."""
    chunks = _horizon_chunks(brief, flagged)
    if not chunks:
        return []

    sections: list[str] = []
    current: list[str] = []
    current_len = 0
    for chunk in chunks:
        chunk_len = len(chunk) + 2  # + the blank-line separator
        if current and current_len + chunk_len > MAX_SECTION_TEXT:
            sections.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(chunk)
        current_len += chunk_len
    sections.append("\n\n".join(current))

    return [_section("*On the Horizon*")] + [_section(s) for s in sections]


def _further_blocks(brief: Brief, flagged: dict[str, list[str]]) -> list[dict]:
    if brief.video is None:
        return []
    return [_section("*Further Afield*"), _section(_top_block(brief.video, flagged))]


def _meta_block(brief: Brief) -> dict | None:
    if not brief.meta:
        return None
    return _context(_escape(brief.meta))


def _paginate(blocks: list[dict]) -> list[dict]:
    """Pack blocks into messages of at most MAX_BLOCKS_PER_MESSAGE, splitting
    only between blocks, never inside one — a section (already cut on a
    category/entry boundary above) is never divided across messages."""
    messages: list[list[dict]] = []
    current: list[dict] = []
    for block in blocks:
        if len(current) >= MAX_BLOCKS_PER_MESSAGE:
            messages.append(current)
            current = []
        current.append(block)
    if current:
        messages.append(current)
    return [{"blocks": m} for m in messages if m]


def to_slack(brief: Brief, *, considered: int, warnings: list[Warning],
            flagged: dict[str, list[str]] | None = None) -> list[dict]:
    """Render a Brief to one or more ready-to-POST Slack webhook payloads.

    Usually one message. Divider blocks do the separation embeds gave Discord
    for free; splits into more messages only when the block count would exceed
    Slack's 50-block ceiling, always on a section boundary.
    """
    flagged = flagged or {}
    lost = degraded(warnings)

    blocks = [_header()]
    if lost:
        blocks.append(_degraded_block(lost))
    blocks.append({"type": "divider"})

    blocks += _today_blocks(brief, flagged)

    horizon = _horizon_blocks(brief, flagged)
    if horizon:
        blocks.append({"type": "divider"})
        blocks += horizon

    further = _further_blocks(brief, flagged)
    if further:
        blocks.append({"type": "divider"})
        blocks += further

    blocks.append({"type": "divider"})
    meta = _meta_block(brief)
    if meta:
        blocks.append(meta)
    blocks.append(_context(_escape(footer_text(
        considered, len(brief.entries()), brief.shortfall, warnings))))

    return _paginate(blocks)
