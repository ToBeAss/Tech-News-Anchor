"""Fact verification — catch figures the model invented, and point the link at
the article that supports them.

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
from dataclasses import replace

from .dedupe import lead_rank
from .warnings import Warning

# A run of digits, optionally with internal separators: 55, 0.32, 1,200, 1 000.
# The escape is deliberate — the set includes a non-breaking space, which a
# plain literal loses to any editor that normalises whitespace.
_SEPARATORS = ".,  "
_NUM_RE = re.compile(rf"\d[\d{_SEPARATORS}]*\d|\d")

# Years are routinely correct-by-context and rarely present in a short summary,
# so checking them produces mostly false alarms.
_YEAR_RANGE = range(1900, 2101)

# Letters that follow a figure as a magnitude or unit rather than making it a
# name: $10B, 38bn, 133MW, 4.7bn. Without this, "$10B" in a headline reads as an
# identifier and the figure it states goes unrecognised.
_UNIT_SUFFIXES = {
    "b", "bn", "m", "mn", "k", "t", "g", "x", "%",
    "mw", "gw", "kw", "tw", "kb", "mb", "gb", "tb", "pb",
}
_TRAILING_ALPHA = re.compile(r"[A-Za-z%]+")


def _normalise(raw: str) -> str:
    """'1 200' -> '1200', '55,0' -> '550'. Separator-insensitive comparison."""
    return re.sub(f"[{_SEPARATORS}]", "", raw)


def _is_embedded(text: str, start: int, end: int) -> bool:
    """True when the digits are part of a name rather than a quantity.

    "kode24", "GPT-4", "Web3", "COVID-19" all contain digit runs that are not
    figures. Flagging them produced a false alarm on a brand name, which is
    worse than a miss: a checker that cries wolf gets ignored.
    """
    before = text[start - 1] if start > 0 else ""
    if before.isalnum() or before == "-":
        return True                     # kode24, GPT-4, Web3, COVID-19

    match = _TRAILING_ALPHA.match(text, end)
    if match and match.group().lower() not in _UNIT_SUFFIXES:
        return True                     # 3rd, 2026Q1 and similar word-forms
    return False


def _numbers(text: str) -> list[tuple[str, str]]:
    """Return (raw, normalised) for each figure worth checking."""
    out = []
    for match in _NUM_RE.finditer(text or ""):
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
    """The text a figure may be drawn from.

    Merged siblings contribute their TITLE only, matching Item.for_prompt()
    exactly. Widening this to their summaries would let a figure the model was
    never shown pass as supported, which is the failure this module exists for.
    """
    parts = [item.title, item.summary]
    if not lead_only:
        parts += [r.title for r in item.related]
    return " ".join(p for p in parts if p)


def _unsupported(written: str, source: str) -> list[str]:
    """Figures in `written` that nothing in `source` backs.

    A rounded or truncated form counts: "55" backs "55.4", and a figure written
    as "38" is accepted only if 38 actually starts a source number.
    """
    supported = {norm for _raw, norm in _numbers(source)}
    out, reported = [], set()
    for raw, norm in _numbers(written):
        if norm in reported:
            continue
        if any(s == norm or s.startswith(norm) or norm.startswith(s) for s in supported):
            continue
        reported.add(norm)
        out.append(raw)
    return out


def check_entry(entry) -> list[str]:
    """Figures in the entry's own words that do not appear in its source."""
    return _unsupported(f"{entry.headline} {entry.fact} {entry.comment}",
                        _source_text(entry.item))


def check_lead(entry) -> list[str]:
    """Figures supported only by a merged sibling, not by the item being linked.

    A reader clicks the primary link. If the headline's number lives in a
    different outlet's article about a different contract in the same deal
    chain, the click lands somewhere that does not say what the brief says.
    """
    if not entry.item.related:
        return []
    return _unsupported(f"{entry.headline} {entry.fact}",
                        _source_text(entry.item, lead_only=True))


def _promote_by_figure(entry):
    """Swap in the merged sibling that actually supports the headline's figures.

    Dedupe leads with whichever item the model listed first, which is often not
    the one carrying the number the brief quotes — The Register leads, the
    figure comes from digi.no. Rewriting the link is better than warning about
    it every run: a warning that fires on every brief stops being read.

    The whole sibling Item moves into the lead position, so the summary recorded
    in memory still belongs to the URL the reader was given. The id is kept so
    flags and grouping still resolve.
    """
    if not check_lead(entry):
        return entry

    written = f"{entry.headline} {entry.fact}"
    lead = entry.item
    for n, sibling in enumerate(lead.related):
        if _unsupported(written, _source_text(sibling, lead_only=True)):
            continue
        demoted = replace(lead, related=())
        others = lead.related[:n] + lead.related[n + 1:]
        return replace(entry, item=replace(sibling, id=lead.id,
                                           related=(demoted, *others)))
    return entry


# Capitalised tokens, sentence-initial included. Excluding position 0 was the
# first draft and it broke the exact case this exists to catch: this
# project's headlines lead with the actor ("Datatilsynet is investigating
# the Ryde data breach", "Norway funds AI centres it cannot staff") per
# synth.py's own instruction to "name who did what", so the word most likely
# to reveal a subject mismatch is routinely the first one. A stray common
# word ("The", "New") capitalised only by sentence position costs nothing:
# it is exactly as likely to appear in the lead's text as in a sibling's, so
# it cannot tip a STRICT comparison either way.
_PROPER_NOUN_RE = re.compile(r"[A-ZÆØÅ]\w*")


def _proper_nouns(headline: str) -> set[str]:
    """What's most likely to survive translation intact. synthesis writes
    headlines in English from non-English sources (see synth.py's system
    prompt); an entity name is usually copied verbatim where a verb or
    adjective would be translated, so it is the part of the headline a
    Norwegian-language sibling can still be scored against."""
    return {m.group() for m in _PROPER_NOUN_RE.finditer(headline)}


def _coverage(nouns: set[str], text: str) -> int:
    return sum(1 for n in nouns if n in text)


def _best_subject_sibling(entry) -> int | None:
    """Index into entry.item.related of the sibling whose text covers more of
    the headline's proper nouns than the lead's own text does, or None.

    A STRICT improvement, not a subset test. Subset fails on a true positive:
    an on-topic merged sibling almost always shares some of the lead's nouns
    without being a better match for what the headline actually says, and
    the earlier subset draft demoted a correct PREFER_LEAD source (digi.no)
    to a coincidentally-overlapping one (simonwillison.net) over an AISI
    headline. Ties keep the lead for the same reason — a sibling has to win
    the comparison, not draw it.

    title + summary on BOTH sides — deliberately NOT title-only despite
    _source_text's rule to the contrary. That rule exists so a figure
    verification never treats text the model was never shown as support for a
    claim; it governs what may be trusted as evidence FOR a specific written
    figure. This function verifies nothing — it only compares candidates
    against each other to decide which one the reader should land on — so
    there is no unseen text being smuggled in as support for anything. Scoring
    the sibling on title alone while scoring the lead on title+summary would
    just handicap every sibling by however much summary the lead happens to
    carry, for no correctness reason. Do not narrow this back to title-only.

    Ranked, not first-match: among every sibling that strictly beats the
    lead's score, the highest-scoring one wins; a tie between siblings falls
    back to PREFER_LEAD order, then to tuple position (which is itself
    PREFER_LEAD-ordered by construction — see dedupe._order_group — so this
    third level only ever matters for two siblings PREFER_LEAD doesn't rank).
    """
    nouns = _proper_nouns(entry.headline)
    if not nouns:
        return None
    lead = entry.item
    lead_score = _coverage(nouns, f"{lead.title} {lead.summary}")
    improvers = [
        (n, _coverage(nouns, f"{sibling.title} {sibling.summary}"), sibling)
        for n, sibling in enumerate(lead.related)
    ]
    improvers = [(n, score, sibling) for n, score, sibling in improvers if score > lead_score]
    if not improvers:
        return None
    best_n, _score, _sibling = max(
        improvers, key=lambda c: (c[1], -lead_rank(c[2]), -c[0])
    )
    return best_n


def _promote_by_subject(entry):
    """The second gate. check_lead (and so _promote_by_figure) no-ops on any
    figure-free headline — "Datatilsynet is investigating the Ryde data
    breach" states no number — so a headline written from a merged sibling's
    angle rather than the lead's own could pass through _promote_by_figure
    untouched while still linking the wrong article. This catches that class
    by subject rather than by figure."""
    n = _best_subject_sibling(entry)
    if n is None:
        return entry
    lead = entry.item
    sibling = lead.related[n]
    demoted = replace(lead, related=())
    others = lead.related[:n] + lead.related[n + 1:]
    return replace(entry, item=replace(sibling, id=lead.id,
                                       related=(demoted, *others)))


def _promote_lead(entry):
    """Figure support first, subject coverage second. The figure gate is
    tried first because it is the stronger claim — an unsupported number is a
    factual defect, a subject mismatch is a framing one — and a promotion it
    makes is left alone rather than being second-guessed by the subject gate.
    """
    fixed = _promote_by_figure(entry)
    if fixed is not entry:
        return fixed
    return _promote_by_subject(entry)


def check_subject(entry) -> bool:
    """True when a merged sibling's title covers the headline's subject better
    than the lead being linked does.

    Called again here, after promote_leads has already run, for the same
    reason check_lead is: _promote_by_subject only ever runs when the figure
    gate didn't already claim the swap, so an entry the figure gate promoted
    for an unrelated reason can still leave a subject mismatch unresolved.
    Firing here means no better sibling was available to fix it automatically.
    """
    return _best_subject_sibling(entry) is not None


def promote_leads(brief):
    """Re-point any entry whose figures — or, failing that, whose headline
    subject — are better supported by a sibling than by its own lead. Must run
    before memory records the brief, so the snapshot is taken against the link
    that ships."""
    swapped = {}

    def fix(entry):
        if entry is None:
            return None
        promoted = _promote_lead(entry)
        if promoted is not entry:
            swapped[entry.item.id] = promoted
        return promoted

    top = [fix(e) for e in brief.top]
    also = [fix(e) for e in brief.also]
    video = fix(brief.video)
    # `groups` holds the same Entry objects as `also`, so it is rebuilt from the
    # swap map rather than re-promoted — otherwise the two views disagree.
    groups = tuple((name, tuple(swapped.get(e.item.id, e) for e in entries))
                   for name, entries in brief.groups)
    return replace(brief, top=top, also=also, video=video, groups=groups)


def check(brief) -> tuple[dict[str, list[str]], list[Warning]]:
    """Verify every entry. Returns (item_id -> bad figures, warnings).

    Every warning here is operator-only: a figure or subject mismatch is a
    defect to fix or investigate, not something the brief's reader has any
    use for seeing in a Slack/Discord footer.
    """
    flagged: dict[str, list[str]] = {}
    warnings: list[Warning] = []

    for entry in brief.entries():
        bad = check_entry(entry)
        if bad:
            flagged[entry.item.id] = bad
            warnings.append(Warning(
                f"UNVERIFIED FIGURE in {entry.item.id}: {', '.join(bad)} "
                "— not present in the source text"
            ))
            continue

        # Supported, but by a merged sibling rather than the article being
        # linked — and no sibling could be promoted to fix it.
        off_lead = check_lead(entry)
        if off_lead:
            warnings.append(Warning(
                f"OFF-LEAD FIGURE in {entry.item.id}: {', '.join(off_lead)} "
                "— supported only by a merged source, not the primary link"
            ))

        # Same idea, no figure involved: the headline's subject reads closer
        # to a merged sibling than to the article being linked, and promotion
        # found no sibling that covers it strictly better than the lead does.
        if check_subject(entry):
            warnings.append(Warning(
                f"OFF-LEAD SUBJECT in {entry.item.id}: headline reads closer "
                "to a merged sibling than to the primary link"
            ))
    return flagged, warnings
