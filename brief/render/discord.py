"""Discord embed rendering — pure, no network.

Consumes the same Brief the terminal renderer does, per the seam render/'s
docstring anticipated. brief/deliver.py does the posting; kept separate so this
module is testable without a webhook (mirrors the qotd project's split between
rendering a payload and dispatch.py::post_to_discord actually sending it).

Section names are the whole lighthouse-metaphor budget for this brand — do not
extend it into headers, footers, or variable names elsewhere.
"""

from __future__ import annotations

from ..synth import Brief, Entry
from ..warnings import Warning
from .budget import (ALSO_MAX_CHARS, MAX_RELATED_SHOWN, clip, date_title, degraded,
                     footer_text, origin_line)

ACCENT = 0xD92B2B       # brand red — top signal
SECONDARY = 0x4C72B0    # blue — horizon section
DEGRADED = 0xE8A33D     # amber — a source was lost this run or further afield
META_COLOR = 0x4A4A50   # subdued grey; meta is context, not content

# Discord webhook limits (enforced below, not hoped for).
MAX_MESSAGE_CHARS = 6000
MAX_DESCRIPTION = 4096
MAX_TITLE = 256
MAX_FOOTER = 2048
MAX_EMBEDS_PER_MESSAGE = 10


def _md_link(text: str, url: str) -> str:
    # An unescaped ']' or '[' in link text breaks Discord's markdown parser.
    return f"[{text.replace(']', ')').replace('[', '(')}]({url})"


def _top_block(entry: Entry, flagged: dict[str, list[str]]) -> str:
    lines = [f"**{_md_link(entry.headline, entry.item.url)}**"]
    # Fact and comment are separate fields so the model must produce both, but
    # they read as one paragraph — same join as terminal.py's _entry().
    body = " ".join(p for p in (entry.fact, entry.comment) if p)
    if body:
        lines.append(body)
    lines.append(f"*{origin_line(entry.item)}*")
    # Each merged sibling carries its own publish time: a cluster can span
    # reports written hours apart as an incident's understanding evolved
    # (initial "an attack", official "not malicious" hours later, "resolved"
    # after that), and without it a reader clicking the earliest link has no
    # way to know it's superseded rather than contradicting.
    #
    # Capped: a 6-outlet merge produced six link lines under one headline,
    # drowning it. See budget.MAX_RELATED_SHOWN.
    shown = entry.item.related[:MAX_RELATED_SHOWN]
    lines += [f"also: {_md_link(origin_line(r), r.url)}" for r in shown]
    remaining = len(entry.item.related) - len(shown)
    if remaining:
        lines.append(f"+{remaining} more")
    bad = flagged.get(entry.item.id)
    if bad:
        # Inline, next to the claim it belongs to — not collected in a footer
        # the reader reaches after already trusting the fact.
        lines.append(f"⚠️ unverified: {', '.join(bad)}")
    return "\n".join(lines)


def _horizon_line(entry: Entry, flagged: dict[str, list[str]]) -> str:
    head = f"**{_md_link(entry.headline, entry.item.url)}**"
    text = f"{head} — {clip(entry.comment, ALSO_MAX_CHARS)}" if entry.comment else head
    # A trailing count, not a link per sibling — tier two is a scan surface,
    # and "also:" lines here read as floating between entries rather than
    # attached to the one they belong to.
    if entry.item.related:
        text += f" (+{len(entry.item.related)})"
    bad = flagged.get(entry.item.id)
    if bad:
        text += f" ⚠️ {', '.join(bad)}"
    return text


def _header_embed(lost: list[Warning]) -> dict:
    embed = {
        "author": {"name": "Fyrtårn"},
        "title": date_title(),
        "color": DEGRADED if lost else ACCENT,
    }
    # Degraded state read before the content, not in a footer — same reasoning
    # as terminal.py putting the DEGRADED banner above TOP SIGNAL.
    if lost:
        embed["description"] = (
            f"⚠ {len(lost)} source(s) unreachable — ranking drew on a reduced pool\n"
            + "\n".join(f"• {w.text}" for w in lost)
        )
    return embed


def _today_embed(brief: Brief, flagged: dict[str, list[str]]) -> dict:
    body = "\n\n".join(_top_block(e, flagged) for e in brief.top)
    return {"title": "Today's Signal", "description": body, "color": ACCENT}  # red


def _horizon_embeds(brief: Brief, flagged: dict[str, list[str]]) -> list[dict]:
    """One embed, normally. Splits on a category (or entry, flat) boundary if
    the assembled description would overflow a single embed's 4096 chars —
    never mid-category, matching the section-boundary rule for messages."""
    if brief.groups:
        blocks = [
            "**{}**\n{}".format(name.upper(),
                                "\n".join(_horizon_line(e, flagged) for e in entries))
            for name, entries in brief.groups
        ]
    else:
        blocks = [_horizon_line(e, flagged) for e in brief.also]
    if not blocks:
        return []

    embeds: list[dict] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2  # + the blank-line separator
        if current and current_len + block_len > MAX_DESCRIPTION:
            embeds.append(_horizon_embed(current, first=not embeds))
            current, current_len = [], 0
        current.append(block)
        current_len += block_len
    embeds.append(_horizon_embed(current, first=not embeds))
    return embeds


def _horizon_embed(blocks: list[str], *, first: bool) -> dict:
    embed = {"description": "\n\n".join(blocks), "color": SECONDARY}  # blue
    if first:
        embed["title"] = "On the Horizon"
    return embed


def _further_embed(brief: Brief, flagged: dict[str, list[str]]) -> dict | None:
    if brief.video is None:
        return None
    return {"title": "Further Afield", "description": _top_block(brief.video, flagged),
            "color": DEGRADED}  # amber — calls out as separate from main signal


def _meta_embed(brief: Brief) -> dict | None:
    if not brief.meta:
        return None
    return {"title": "Context", "description": f"*{brief.meta}*", "color": META_COLOR}


def _embed_chars(embed: dict) -> int:
    total = len(embed.get("title", "")) + len(embed.get("description", ""))
    total += len((embed.get("author") or {}).get("name", ""))
    total += len((embed.get("footer") or {}).get("text", ""))
    return total


def _paginate(embeds: list[dict]) -> list[dict]:
    """Pack embeds into webhook payloads. Splits only between embeds, never
    inside one, so a section (or a horizon sub-embed, itself already cut on a
    category boundary) is never divided across messages."""
    messages: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for embed in embeds:
        chars = _embed_chars(embed)
        if current and (len(current) >= MAX_EMBEDS_PER_MESSAGE
                        or current_chars + chars > MAX_MESSAGE_CHARS):
            messages.append(current)
            current, current_chars = [], 0
        current.append(embed)
        current_chars += chars
    if current:
        messages.append(current)
    # Filter out empty message lists to prevent posting blank messages.
    return [{"embeds": m} for m in messages if m]


def to_discord(brief: Brief, *, considered: int, warnings: list[Warning],
               flagged: dict[str, list[str]] | None = None) -> list[dict]:
    """Render a Brief to one or more ready-to-POST webhook payloads.

    Usually one message. Splits into more only when the assembled embeds would
    exceed Discord's 6000-char message budget, and always on a section
    boundary — Today's Signal first, On the Horizon onward after — never
    mid-section. The header embed is therefore always in the first message and
    the footer always lands on the true last embed, wherever it ends up.
    """
    flagged = flagged or {}
    lost = degraded(warnings)

    embeds = [_header_embed(lost), _today_embed(brief, flagged)]
    embeds += _horizon_embeds(brief, flagged)
    further = _further_embed(brief, flagged)
    if further:
        embeds.append(further)
    meta = _meta_embed(brief)
    if meta:
        embeds.append(meta)

    embeds[-1]["footer"] = {
        "text": footer_text(considered, len(brief.entries()), brief.shortfall,
                            warnings)[:MAX_FOOTER]
    }

    for embed in embeds:
        if "title" in embed:
            embed["title"] = embed["title"][:MAX_TITLE]
        if "description" in embed:
            embed["description"] = embed["description"][:MAX_DESCRIPTION]

    # Filter out embeds with no title and no description
    embeds = [e for e in embeds
              if e.get("title") or e.get("description")
              or e.get("author") or e.get("footer")]

    return _paginate(embeds)
