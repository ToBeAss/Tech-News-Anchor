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
a second line of defence. That fallback is lossy, though — synthesis dedups by
DROPPING the weaker version, so the other outlet's link is gone, whereas a merge
here keeps it in `related`. A miss at this stage is not recoverable later, which
is why it gets summaries and a co-reference rule rather than titles alone.
"""

from __future__ import annotations

from dataclasses import replace

from . import llm
from .sources import Item

# How much of each summary the grouping call sees. Titles alone could not link
# "Volta claims $10B AI lab deal for Norway bit barn" to "Anthropic flytter
# trolig inn i Trøndelag": the bridging facts — Volta operates the Tydal site,
# Tydal is in Trøndelag, Anthropic is the customer — appear in no title. They do
# appear in the summaries. Truncated because the opening sentences carry the
# who/what and this is meant to stay the cheap call.
#
# This does NOT widen what synthesis or verify see. The
# `verify._source_text` / `Item.for_prompt` invariant governs the text a written
# figure is checked against; dedupe feeds neither and produces no prose.
SUMMARY_CHARS = 240

# Sources that win the lead position inside a merged cluster, most-preferred
# first. The lead owns the primary link and the origin line, so this is a
# ranking rule rather than cosmetics: `priorities` says the Norwegian instance of
# a Norwegian story outranks the international wire version, and enforcing that
# here beats hoping synthesis remembers it mid-ranking. Same reasoning as the
# caps in synth._validate.
#
# verify.promote_leads can still override this when a written figure is only
# supported by a sibling — a broken link promise outranks outlet preference.
PREFER_LEAD: tuple[str, ...] = ("digi.no", "kode24")

_SYSTEM = """You group news items that cover the SAME underlying event.

You are given numbered items, each with an id, source, headline, and the \
opening of its summary. Some describe the same event from different outlets, in \
different languages, or from different angles (e.g. one reports a contract being \
signed, another names the likely customer in that same contract).

The same event is often named differently by each outlet — by the operator, the \
site, the customer, the parent company, or the region. Read the summaries for \
these bridges, not just the headlines: an item whose headline names only the \
customer and one whose headline names only the site are the same story if a \
summary connects them.

Differing figures are NOT evidence of different events. Outlets report different \
contract scopes, different currencies, and revised numbers for one deal, and a \
feed may revise its own figure during the day.

Group ONLY items that a reader would consider the same story. Do NOT group \
items that merely share a topic, a company, a sector, or a theme. Two separate \
attacks on the same organisation are one story if they are the same reported \
incident, and two stories if they are distinct events. A local instance of a \
global trend is NOT the same story as the global one.

EVERY item in a group must be about the same specific event as every OTHER \
item in that group — check each pair, not just each item against the first. \
Do not chain: if A and B are the same story and B mentions C, that does not \
put C in the group. Two datacentre contracts in different countries are two \
stories even when one article discusses both.

An actor advancing its own project and an authority acting on that class of \
project are two events, even in the same place and the same week. "Company X \
progresses its datacentre" and "the state pauses datacentre approvals" are \
separate stories however closely related — do not merge a vendor's own news \
with a regulator's, a union's, or a court's response to the general situation \
that vendor operates in.

For each group, first write "event": one short sentence naming the specific \
thing that happened, with its actor and its place. If you cannot write one \
sentence that is true of every item in the group, it is not a group.

Output STRICT JSON, nothing else — no prose, no markdown fences:
{"groups": [{"event": "Acme signs compute contract with Example Corp in Oslo", \
"ids": ["i03", "i17"]}]}

Only include groups of 2 or more ids. If nothing should be grouped, output \
{"groups": []}. Never invent ids."""


def _listing_line(item: Item) -> str:
    line = f"{item.id} ({item.origin}) {item.title}"
    if item.summary:
        line += f"\n    {item.summary[:SUMMARY_CHARS]}"
    return line


def _lead_rank(item: Item) -> int:
    try:
        return PREFER_LEAD.index(item.source)
    except ValueError:
        return len(PREFER_LEAD)


def _order_group(ids: list[str], index: dict[str, Item]) -> list[str]:
    """Put the preferred outlet first. Stable, so a group with no preferred
    source keeps the model's own ordering."""
    return sorted(ids, key=lambda i: _lead_rank(index[i]))


def _valid_groups(data: dict, index: dict[str, Item]) -> list[tuple[str, list[str]]]:
    """Keep only groups of 2+ known, not-yet-claimed ids, paired with the event
    the model named for them.

    The bare-list shape is still accepted. A model that ignores the schema
    should degrade to unnamed-but-usable groups rather than to nothing — dedupe
    is fail-open, and that applies to its own output format too.
    """
    raw_groups = data.get("groups") or []
    if not isinstance(raw_groups, list):
        return []
    claimed: set[str] = set()
    groups: list[tuple[str, list[str]]] = []
    for group in raw_groups:
        if isinstance(group, list):
            event, raw_ids = "", group
        elif isinstance(group, dict):
            event = str(group.get("event") or "").strip()
            raw_ids = group.get("ids") or []
        else:
            continue
        if not isinstance(raw_ids, list):
            continue
        ids = [str(i).strip() for i in raw_ids]
        ids = [i for i in ids if i in index and i not in claimed]
        if len(ids) >= 2:
            claimed.update(ids)
            groups.append((event, _order_group(ids, index)))
    return groups


def merge(items: list[Item], *, model=None, max_output_tokens: int = 500):
    """Return (items, note). Items in a group collapse into the first, with the
    rest attached as `related` so their links survive into the brief."""
    if len(items) < 2:
        return items, None

    listing = "\n".join(_listing_line(i) for i in items)
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
    followers: dict[str, list[Item]] = {
        ids[0]: [index[i] for i in ids[1:]] for _event, ids in groups
    }
    absorbed = {i for _event, ids in groups for i in ids[1:]}

    merged: list[Item] = []
    for item in items:
        if item.id in absorbed:
            continue
        extra = followers.get(item.id)
        merged.append(replace(item, related=tuple(extra)) if extra else item)

    collapsed = len(items) - len(merged)
    note = (f"merged {collapsed} duplicate item(s) into "
            f"{len(groups)} stor{'y' if len(groups) == 1 else 'ies'}")
    # The named events are the audit trail for this stage: a chained over-merge
    # is obvious in one line here, whereas a bad merge is otherwise only visible
    # as a wrong `also` link three sections into the brief.
    named = "; ".join(event for event, _ids in groups if event)
    return merged, (f"{note} — {named}" if named else note)
