"""Fact verification — catch figures the model invented.

The `fact` field asks for specifics ("the number, the version, the claim") but
the model only ever sees a title and a truncated summary. When the specific is
not in there, it will supply a plausible one: a NOK 55 billion contract was
reported as NOK 38 billion, in the most authoritative-looking line of the brief.

Numbers are the highest-risk hallucination and the only one that is cheap to
check mechanically, so that is what this does: every figure in a headline or
fact must appear somewhere in the source item's own text. This is the same
philosophy as the URL handling — rather than asking the model to be careful,
make the error structurally detectable.

Non-numeric claims ("the policy covers disclosure, review, and acceptable use")
are NOT checked here. That needs either the full article text or a separate
verification pass; see the notes in the project discussion.
"""

from __future__ import annotations

import re

# A run of digits, optionally with internal separators: 55, 0.32, 1,200, 1 000
_NUM_RE = re.compile(r"\d[\d.,\u00a0 ]*\d|\d")

# Years are routinely correct-by-context and rarely present in a short summary,
# so checking them produces mostly false alarms.
_YEAR_RANGE = range(1900, 2101)


def _normalise(raw: str) -> str:
    """'1 200' -> '1200', '55,0' -> '550'. Separator-insensitive comparison."""
    return re.sub(r"[.,\u00a0 ]", "", raw)


def _is_embedded(text: str, start: int, end: int) -> bool:
    """True when the digits are part of a name rather than a quantity.

    "kode24", "GPT-4", "Web3", "COVID-19" all contain digit runs that are not
    figures. Flagging them produced a false alarm on a brand name, which is
    worse than a miss: a checker that cries wolf gets ignored.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return (before.isalnum() or before == "-") or after.isalpha()


def _numbers(text: str) -> list[tuple[str, str]]:
    """Return (raw, normalised) for each figure worth checking."""
    text = text or ""
    out = []
    for match in _NUM_RE.finditer(text):
        if _is_embedded(text, match.start(), match.end()):
            continue
        raw = match.group().strip()
        norm = _normalise(raw)
        if not norm:
            continue
        if len(norm) == 4 and norm.isdigit() and int(norm) in _YEAR_RANGE:
            continue                     # a year: skip
        out.append((raw, norm))
    return out


def _source_text(item, *, lead_only: bool = False) -> str:
    parts = [item.title, item.summary]
    if not lead_only:
        parts += [t for _o, t, _u in item.related]
    return " ".join(p for p in parts if p)


def check_entry(entry) -> list[str]:
    """Figures in the entry's own words that do not appear in its source."""
    supported = {n for _raw, n in _numbers(_source_text(entry.item))}
    written = _numbers(f"{entry.headline} {entry.fact} {entry.comment}")

    unsupported: list[str] = []
    seen: set[str] = set()
    for raw, norm in written:
        if norm in seen:
            continue
        # Accept a rounded or truncated form: "55" backs "55.4", and a figure
        # written as "38" is only accepted if 38 actually starts a source number.
        if any(s == norm or s.startswith(norm) or norm.startswith(s) for s in supported):
            continue
        seen.add(norm)
        unsupported.append(raw)
    return unsupported


def check_lead(entry) -> list[str]:
    """Figures supported only by a merged sibling, not by the item being linked.

    A reader clicks the primary link. If the headline's number lives in a
    different outlet's article about a different contract in the same deal
    chain, the click lands somewhere that does not say what the brief says.
    """
    if not entry.item.related:
        return []
    lead = {n for _raw, n in _numbers(_source_text(entry.item, lead_only=True))}
    off_lead, seen = [], set()
    for raw, norm in _numbers(f"{entry.headline} {entry.fact}"):
        if norm in seen:
            continue
        if any(s == norm or s.startswith(norm) or norm.startswith(s) for s in lead):
            continue
        seen.add(norm)
        off_lead.append(raw)
    return off_lead


def check(brief) -> tuple[dict[str, list[str]], list[str]]:
    """Verify every entry. Returns (item_id -> bad figures, warnings)."""
    flagged: dict[str, list[str]] = {}
    warnings: list[str] = []
    entries = [*brief.top, *brief.also] + ([brief.video] if brief.video else [])

    for entry in entries:
        bad = check_entry(entry)
        if bad:
            flagged[entry.item.id] = bad
            warnings.append(
                f"UNVERIFIED FIGURE in {entry.item.id}: {', '.join(bad)} "
                "\u2014 not present in the source text"
            )
            continue

        # Supported, but by a merged sibling rather than the article being linked.
        off_lead = check_lead(entry)
        if off_lead:
            warnings.append(
                f"OFF-LEAD FIGURE in {entry.item.id}: {', '.join(off_lead)} "
                "\u2014 supported only by a merged source, not the primary link"
            )
    return flagged, warnings