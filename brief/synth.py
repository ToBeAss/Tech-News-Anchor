"""Synthesis — raw items in, ranked+annotated brief out.

The integrity contract: the model receives items labelled [i01], [i02]… with no
URLs, and returns only those ids. Anything referencing an id we didn't send is
dropped at validation. The model therefore cannot fabricate, mangle, or
mis-attribute a source link — it never handles one.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import llm
from .sources import EPOCH, Item


class SynthError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    item: Item
    headline: str
    comment: str
    fact: str = ""      # what happened; empty for tier two, which carries it in the headline


# Fixed shape: three ranked items on top, then 6-9 grouped ones. Locked counts
# make the output predictable to skim — you learn where to look. The cost is
# that on a thin day the last slots are the best of a weak field rather than
# genuinely strong; the commentary is expected to say so rather than oversell.
TARGET_TOP = 3

# Tier two is grouped, because a list this long stops being scannable flat.
# Categories only pay off with enough items to group: a one-item category is a
# headline with extra formatting.
CAT_MIN_ITEMS = 2
CAT_MIN_GROUPS = 2
CAT_MAX_GROUPS = 4
CAT_TOTAL_MAX = 9

# Backstop, not budget: set above the prompt's own limits (fact 20-30, tier-one
# comment 40-55, tier-two comment 25) so a well-behaved entry never trips them
# and only genuine overruns are reported. Same relationship as ALSO_MAX_CHARS in
# the terminal renderer, which only ever backstopped tier two — tier one had no
# equivalent, and a ~50% overrun there is invisible in a terminal scroll but
# expensive inside a 6000-char Discord message.
TOP_FACT_WORDS = 35
TOP_COMMENT_WORDS = 65
ALSO_COMMENT_WORDS = 28


@dataclass(frozen=True)
class Brief:
    top: list[Entry]
    also: list[Entry]                       # flat view of `groups`, in render order
    video: Entry | None
    meta: str | None
    shortfall: str | None = None            # set when a section came back under target
    groups: tuple[tuple[str, tuple[Entry, ...]], ...] = ()   # (name, entries)

    def entries(self) -> list[Entry]:
        """Everything actually published, in reading order. What memory records
        and verify checks — both must cover exactly what the reader sees."""
        return [*self.top, *self.also, *([self.video] if self.video else [])]


_SYSTEM = """You are a technical editor writing a daily tech brief for one \
engineer. You are given today's candidate stories, each tagged with an id.

Your job is NOT to summarise everything. It is to filter hard, then say \
something worth reading about what survives.

Who this is for:
{audience}

The reader's interests:
{interests}

Priority overrides:
{priorities}

## Selection
- Rank ALL candidates by relevance to those interests, then take the top 9-12. \
This is a ranking task with a fixed output size, not a threshold test.
- The 3 strongest go in top_signal, the next 6-9 into also_worth_knowing. The \
split is by strength, so anything in top_signal must be more important than \
everything in also_worth_knowing.
- You must fill both sections. If the day is thin, the weakest entries are \
the best of a weak field — say so plainly in the commentary rather than \
inflating them. "Minor, but the only movement on X this week" is an honest \
line; pretending a slow news day is significant is not.
- Prefer spread across sources. If one source dominates the candidates by \
volume, that is an artefact of its publishing rate, not importance.
- Video candidates are tracked separately and get their own reserved slot. Do \
not skip a video because the written stories are stronger — it is not \
competing with them. If a video is strong enough to lead the whole brief, it \
may go in top_signal instead, but then it must not also appear in video.
- ONE ENTRY PER UNDERLYING STORY. Several candidates often cover the same \
event from different outlets or angles. Pick the single best version and drop \
the rest — do not spend a second slot on the same story because another \
source framed it differently. If two angles are both worth having, merge them \
into one entry's commentary rather than listing both.
- Apply the priority overrides above as real ranking weight, not a tiebreak. \
Do not discount an item because it is short, in another language, or from a \
smaller outlet — a local instance of a global story usually matters more \
to this reader than the global one. If you include the global version, \
prefer the local version instead where both are present.
- Use the audience context to judge SECOND-ORDER relevance — what this \
reader will have to reason about, argue for, or build in the next year. A \
story about another country's public-sector procurement, a standards change, \
or a regulatory precedent can be highly relevant without naming their \
organisation or sector.
- Do NOT turn the audience context into a topic filter. Their field is a lens, \
not a subject requirement. Never force a connection to their organisation into \
the commentary, and never pick a weak item because it is superficially \
on-sector over a strong one that is not.
- Do NOT phrase relevance as advice to a category of reader — "public-sector \
teams should watch this", "important for organisations like this", "geospatial \
teams should note" are all the same tell: the audience block leaking into the \
prose as a label instead of being absorbed as judgement. State what the item \
means or implies instead of who should pay attention to it. If a sentence \
would still make sense with "readers of this brief" swapped in for the \
specific claim, rewrite it.
- RECENCY: each item carries its age. The window deliberately reaches back \
further than a day so nothing is lost to a failed run or a crowded one, but \
the brief is published daily and should read as today's. Prefer the fresher \
item when two are comparable; an older one needs to be clearly stronger to \
earn a slot. If you do surface something several days old, say so in the fact \
sentence ("reported Monday", "last week") rather than implying it just broke.
- Never invent stories. Only use ids from the supplied list.

## Writing
- Every top_signal entry has TWO parts, and both are required:
  * "fact" — what actually happened, in one sentence. Name the actor and \
carry the specific detail: the number, the version, the claim, the decision. \
A reader who only reads this sentence should know the news. Do not editorialise \
here.
    MERGED ITEMS: when an item is marked "SAME STORY, also covered by", the \
other coverage may describe a DIFFERENT PART of the same chain — a \
different contract, party, or layer of the same deal. Do not merge their \
figures or attach one source's party to another source's number. Attribute \
every figure to the specific parties it belongs to, and if the layers matter, \
name them ("A's $10bn deal with B, which in turn contracted C for $4.7bn"). \
Two numbers that look contradictory are usually two different agreements.
    HEDGING: preserve the source's certainty exactly. If it says "trolig", \
"reportedly", "says it has", "is believed to", or attributes a claim to \
someone, your sentence must carry the same hedge and the same attribution. \
"Anthropic signed" is wrong where the source said Anthropic is probably the \
customer; "reporting points to Anthropic" is right. Stripping a hedge turns a \
claim into a fabrication even when every word came from the source.
    GROUNDING (absolute): state ONLY details that appear in the supplied title \
or summary for that item. You are working from a headline and a short extract, \
not the article. If a figure, version, date, name or list of provisions is not \
in the text you were given, you do NOT know it — do not supply it, do not \
infer it, do not reconstruct it from memory. Write around the gap instead: \
"a multi-billion-kroner contract" is correct where the amount is absent, and a \
confident wrong number is a serious failure. The same applies to headlines and \
to any figure in a comment.
  * "comment" — why it matters, what it conflicts with, the second-order \
effect. Do not restate the fact. Do not write filler like "this is significant \
as AI advances rapidly".
- If you cannot state a concrete fact, you do not understand the item well \
enough to rank it highly — move it down.
- Have opinions. Skepticism is welcome. Flag vendor announcements as vendor \
announcements — the origin domain tells you when a story is a company \
talking about itself.
- Write your own headline for each entry — terse, concrete, no clickbait. \
Headlines must be FACTUAL, not thematic: name who did what. "Norway funds AI \
centres it cannot staff, says digi.no op-ed" is right; "Norway's AI talent \
bottleneck resurfaces" is wrong — it names a theme and tells the reader \
nothing. This matters most in also_worth_knowing, where the headline carries \
the news because the one-line comment has no room for it.
- Always write in English, whatever language the source is in. Where a source \
is non-English, do not let a short or untranslated summary count against the \
item's importance.

## Budget (hard limits — the brief must stay under a 3 minute read)
- top_signal: EXACTLY 3 entries. Not 2, not 4. fact 20-30 words (one \
sentence). comment 40-55 words. This tier gets the depth. Word counts are \
hard limits, not targets — an entry over them is a defect, and the third \
sentence of a comment is almost always the one that adds nothing.
- also_worth_knowing: 2-4 CATEGORIES holding 2-3 entries each, 6-9 entries in \
total — see the categories section below. 6 is the preferred count and 9 the \
ceiling, not the target: a day that honestly yields 6 should return 6. Filling \
to 9 by including a weak or misread item costs the reader more than the extra \
entries are worth. comment MAX 25 WORDS — count them. One sentence, and it \
must END, not trail off. This tier is scannable, not readable. If an item \
needs more than one line to justify, it belongs in top_signal or nowhere.
- video: REQUIRED whenever any candidate is marked (video). Pick the single \
best one and put it here. comment max 35 words. This slot is reserved — it \
does not compete with the other two sections, and a video appearing here does \
NOT consume a top_signal or also_worth_knowing slot. Only output null when \
there is genuinely no video candidate in the list.
- meta_note: max 50 words, or null. Use it ONLY for a genuine cross-story \
pattern (several sources circling the same underlying shift). Not a summary \
of the brief.

## Categories (also_worth_knowing only)
Group tier-two entries so a longer list stays scannable. Top signal is NOT \
categorised — it is three items ranked by importance and slicing it \
destroys that ranking.

- Categories emerge FROM the items. Never pick a weak item to fill out a \
category, and never invent a category to justify one story.
- MINIMUM 2 entries per category. A category with one entry is a headline with \
extra formatting — if only one item fits a theme, put it in a broader \
category or drop it.
- 2-4 categories. Above that you are back to a flat list.
- Order categories by the importance of their contents, so the first block is \
the one worth reading first. Order entries within a category the same way.
- Names: two or three plain words. No colons, no cleverness, no puns. \
"Infrastructure & compute", not "The physical substrate of AI".
- REUSE these names wherever they fit, so the brief has a familiar shape day \
to day: Infrastructure & compute / Governance & regulation / Security & \
resilience / Developer tooling / Models & research / Norway & public sector.
  Invent a new name only when the day's items genuinely do not fit any of \
them — novelty here costs the reader more than it gains.

Output STRICT JSON, nothing else — no prose, no markdown fences:
{{
  "top_signal": [{{"id": "iNN", "headline": "...", "fact": "...", "comment": "..."}}],
  "also_worth_knowing": [{{"category": "...", "entries": [{{"id": "iNN", \
"headline": "...", "comment": "..."}}]}}],
  "video": {{"id": "iNN", "headline": "...", "fact": "...", "comment": "..."}} or null,
  "meta_note": "..." or null
}}"""


def build(items: list[Item], interests: str, priorities: str = "",
          audience: str = "", *, model=None, temperature=None,
          max_output_tokens=None) -> Brief:
    if not items:
        raise SynthError("no items to synthesise")

    catalogue = "\n\n".join(i.for_prompt() for i in items)
    try:
        raw = llm.generate(
            [{"role": "user", "content": f"Today's candidates:\n\n{catalogue}"}],
            instructions=_SYSTEM.format(
                audience=(audience or "(not specified)").strip(),
                interests=interests.strip(),
                priorities=(priorities or "(none)").strip(),
            ),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        data = llm.parse_json(raw)
    except llm.LLMError as exc:
        # Synthesis is the one stage where failing open makes no sense: there is
        # no brief without it. Re-raised as SynthError so the CLI has one thing
        # to catch.
        raise SynthError(str(exc)) from exc
    return _validate(data, {i.id: i for i in items})


def _entry(node, index: dict[str, Item]) -> Entry | None:
    """Resolve one node to an Entry, or None if it references an unknown id.

    This is the enforcement half of the no-URLs contract: an id we did not send
    cannot resolve to a link, so it is dropped rather than guessed at.
    """
    if not isinstance(node, dict):
        return None
    item = index.get(str(node.get("id", "")).strip())
    if item is None:
        return None
    return Entry(
        item=item,
        headline=str(node.get("headline") or item.title).strip(),
        comment=str(node.get("comment") or "").strip(),
        fact=str(node.get("fact") or "").strip(),
    )


def _section(data: dict, key: str, index: dict[str, Item]) -> list[Entry]:
    nodes = data.get(key) or []
    if not isinstance(nodes, list):
        return []
    seen: set[str] = set()
    out: list[Entry] = []
    for node in nodes:
        entry = _entry(node, index)
        if entry and entry.item.id not in seen:
            seen.add(entry.item.id)
            out.append(entry)
    return out


def _categories(data: dict, index: dict[str, Item], claimed: set[str]):
    """Parse tier two into (name, entries) groups.

    Categories under CAT_MIN_ITEMS are dropped rather than shown: the whole
    point of grouping is scannability, and a one-item group is noise.
    """
    groups: list[tuple[str, tuple[Entry, ...]]] = []
    seen = set(claimed)
    total = 0

    for node in data.get("also_worth_knowing") or []:
        if not isinstance(node, dict):
            continue
        name = str(node.get("category") or "").strip()
        entries = []
        for child in node.get("entries") or []:
            entry = _entry(child, index)
            if entry and entry.item.id not in seen:
                seen.add(entry.item.id)
                entries.append(entry)
        if not name or len(entries) < CAT_MIN_ITEMS:
            continue
        if total + len(entries) > CAT_TOTAL_MAX:
            entries = entries[:CAT_TOTAL_MAX - total]
            if len(entries) < CAT_MIN_ITEMS:
                break
        groups.append((name, tuple(entries)))
        total += len(entries)
        if len(groups) >= CAT_MAX_GROUPS or total >= CAT_TOTAL_MAX:
            break

    return groups


def _pick_video(data: dict, index: dict[str, Item], placed: set[str]) -> Entry | None:
    """The reserved video slot. If the model ignored it despite a video candidate
    existing, surface the newest one as a bare pointer rather than dropping it.

    The model's pick is kind-checked. Told the slot is reserved and required, it
    will fill it on a day with no video candidates by reaching for something that
    merely *mentions* video — a blog post about generating a clip locally is not
    a video. Same philosophy as the unknown-id drop: make the error structurally
    impossible instead of asking the prompt to be careful.
    """
    video = _entry(data.get("video"), index)
    if video and video.item.kind == "video" and video.item.id not in placed:
        return video

    spare = [i for i in index.values() if i.kind == "video" and i.id not in placed]
    if not spare:
        return None
    newest = max(spare, key=lambda i: i.published or EPOCH)
    return Entry(item=newest, headline=newest.title, comment="")


def _validate(data: dict, index: dict[str, Item]) -> Brief:
    # Caps enforced here, not only in the prompt: prompts drift under pressure,
    # and an over-long section is a silent quality regression.
    top = _section(data, "top_signal", index)[:TARGET_TOP]
    groups = _categories(data, index, claimed={t.item.id for t in top})

    # Backfill: if the model under-filled top_signal, promote from tier two
    # rather than shipping a short tier. Only shifts placement, never invents.
    while len(top) < TARGET_TOP:
        ranked = [(n, e) for n, (_name, entries) in enumerate(groups) for e in entries]
        if not ranked:
            break
        # Prefer an entry its category can spare. Promoting out of a
        # minimum-sized group collapses the group, so one promotion costs the
        # reader two items; strength order is the fallback, not the first choice.
        spare = next((e for n, e in ranked if len(groups[n][1]) > CAT_MIN_ITEMS),
                     ranked[0][1])
        top.append(spare)
        groups = [(n, tuple(e for e in es if e is not spare)) for n, es in groups]
        groups = [(n, es) for n, es in groups if len(es) >= CAT_MIN_ITEMS]

    # `groups` is what the renderer walks, so the flat view is derived from it
    # rather than tracked alongside. Tracking both let an entry survive in
    # `also` after its group was dropped as undersized — invisible to the
    # reader, but still recorded as briefed and so never shown again.
    also = [e for _name, entries in groups for e in entries]

    video = _pick_video(data, index, {e.item.id for e in top + also})

    meta = data.get("meta_note")
    meta = meta.strip() if isinstance(meta, str) and meta.strip() else None

    if not (top or also or video):
        raise SynthError("model returned no resolvable entries")

    shortfall = []
    if len(top) < TARGET_TOP:
        shortfall.append(f"top_signal {len(top)}/{TARGET_TOP}")
    floor = CAT_MIN_GROUPS * CAT_MIN_ITEMS
    if len(also) < floor:
        shortfall.append(f"also_worth_knowing {len(also)}/{floor}")

    return Brief(top=top, also=also, video=video, meta=meta,
                 shortfall=", ".join(shortfall) or None, groups=tuple(groups))


def budget_warnings(brief: Brief) -> list[str]:
    """Entries that overshot the prompt's length budget.

    Reported rather than truncated: a clipped tier-one comment loses the take,
    and the fix is the prompt, not the renderer. Returned as warnings so the
    caller can fold them in with verify's — Brief's shape stays unchanged.
    """
    over: list[str] = []
    for entry in brief.top:
        fact_words = len(entry.fact.split())
        comment_words = len(entry.comment.split())
        if fact_words > TOP_FACT_WORDS:
            over.append(f"{entry.item.id} fact {fact_words}w > {TOP_FACT_WORDS}")
        if comment_words > TOP_COMMENT_WORDS:
            over.append(f"{entry.item.id} comment {comment_words}w > {TOP_COMMENT_WORDS}")
    for entry in brief.also:
        comment_words = len(entry.comment.split())
        if comment_words > ALSO_COMMENT_WORDS:
            over.append(f"{entry.item.id} comment {comment_words}w > {ALSO_COMMENT_WORDS}")
    return [f"OVER BUDGET: {', '.join(over)}"] if over else []
