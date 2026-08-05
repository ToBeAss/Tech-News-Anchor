# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal daily tech brief: RSS/Atom feeds in, a ranked and annotated brief out, printed to the terminal. Iteration 1 is manual runs only; the code is structured so iteration 2 can add Discord/Teams delivery without touching the pipeline.

## Commands

```bash
source .venv/bin/activate           # .venv/Scripts/activate on Windows
pip install -r requirements.txt

python main.py                 # full run: fetch, gate, dedupe, synthesise, print
python main.py --dry           # ingestion + gate only, no synthesis call (nearly free)
python main.py --dry --no-gate # ingestion only — completely free
python main.py --json          # machine-readable output
python main.py --hours 72      # widen the lookback window for a catch-up run
python main.py --model gpt-5.4-mini   # override the model for every stage
python main.py --no-dedupe --no-gate --no-memory   # skip individual stages
python main.py --forget        # clear state/seen.json and exit
python main.py --why <url>     # what the source said when that item was briefed
```

There is no test suite, linter, or build step. `--dry --no-gate` is the free feedback loop when changing ingestion, and `--json` when changing synthesis.

`OPENAI_API_KEY` comes from `.env` (see `.env.example`). Everything else lives in [config.yaml](config.yaml).

## Layout

```
main.py              CLI and composition root — the only place stages are wired
config.yaml          sources, window, model-per-stage, and the four prompt blocks
brief/               the pipeline; ROOT (project root) is defined in __init__.py
brief/render/        output targets; new delivery channels go here
state/seen.json      briefed-items history (gitignored)
```

## Pipeline

[main.py](main.py) wires the stages, each in its own module under `brief/`. Every stage after ingestion is optional via a flag.

1. **[brief/sources.py](brief/sources.py)** — fetch every feed, filter by window and per-source keywords, canonicalise URLs, dedupe by URL, assign ids.
2. **[brief/gate.py](brief/gate.py)** — one small LLM call per source that has a `gate:` rule, dropping off-topic items. Runs *before* `--dry` so the dry listing reflects the real candidate pool.
3. **[brief/memory.py](brief/memory.py)** — drop items already briefed in a previous run. Also before `--dry`.
4. **[brief/dedupe.py](brief/dedupe.py)** — one LLM call that groups ids covering the same underlying event; followers collapse into the leader's `related` tuple.
5. **[brief/synth.py](brief/synth.py)** — the main call: rank, select 3 + 6-9 grouped + 1 video, write headline/fact/comment.
6. **[brief/verify.py](brief/verify.py)** — promote the merged sibling that supports the headline, then check every figure against the source text. Runs *before* memory is recorded.
7. **[brief/render/](brief/render/)** — terminal and JSON output.

[brief/llm.py](brief/llm.py) is the only network path to the model: raw `requests` against the OpenAI Responses API, no SDK (this is meant to run on a Raspberry Pi). Everything it raises is an `LLMError`, including transport failures, so callers never have to know `requests` exceptions exist.

## Design invariants

These are load-bearing; several exist because the obvious alternative failed in practice.

**The model never sees URLs.** Items are labelled `[i01]`, `[i02]`… and the model returns only ids. Anything referencing an unknown id is dropped in `synth._entry`. A fabricated or mis-attributed link is therefore structurally impossible, not merely discouraged. Preserve this when adding stages.

**Numbers are verified mechanically, not by prompting.** `verify.check` requires every figure in a headline/fact/comment to appear in the item's own title or summary (separator-insensitive, years and name-embedded digits like `kode24`/`GPT-4` excluded). Same philosophy as the URL rule: make the error detectable rather than asking the model to be careful.

**`verify._source_text` and `Item.for_prompt` must agree.** Merged siblings contribute their *title only* to both. Widening the check to sibling summaries would let a figure the model was never shown pass as supported, which is the exact failure the module exists to catch.

**A merged sibling that supports the headline is promoted to lead.** The reader clicks the primary link, so a figure that only lives in the other outlet's article is a broken promise. `verify.promote_leads` swaps the whole sibling `Item` into the lead position — not a projection of it — so the snapshot memory records still belongs to the URL that shipped. When no sibling supports the figure, the `OFF-LEAD FIGURE` warning still fires.

**Failure modes are chosen per stage, deliberately:**
- Ingestion — a dead feed degrades the brief but never aborts the run; `UNREACHABLE` warnings surface at the *top* of the output, not in a footer, because a reduced pool changes how much to trust the ranking.
- Gate and dedupe — **fail-open**. A failed call passes items through with a warning. A silently discarded item is gone for good; an off-topic one that slips past will very likely be dropped by ranking anyway.
- Synthesis — fatal, re-raised as `SynthError` so the CLI has one thing to catch.

**Caps are enforced in code, not only in the prompt.** `synth._validate` truncates `top_signal` to `TARGET_TOP`, drops categories under `CAT_MIN_ITEMS`, backfills an under-filled `top_signal` by promoting from tier two (placement only — it never invents entries), reserves the video slot by falling back to the newest unused video candidate, and reports a `shortfall`. The backfill prefers an entry whose category can spare it: promoting out of a minimum-sized group collapses the group, so one promotion would otherwise cost the reader two items.

**`Brief.groups` is the source of truth for tier two; `Brief.also` is derived from it.** Tracking both independently let an entry survive in `also` after its category was dropped as undersized — invisible to the reader, but still recorded as briefed and therefore never shown again.

**Memory records only what was actually briefed**, keyed by canonical URL, with a 400-char snapshot of the source text *as it read at the time*. Feeds mutate — digi.no revised a contract figure between two runs on the same day, and the URL slug still carries the old number today — and without the snapshot there is no way to answer "what did the model actually see?". `--why <url>` reads it back. A merged item marks all its sibling URLs as seen, so tomorrow's run cannot resurface the story through the other outlet's link.

**Model choice is per stage** (`llm.stages` in config, overridable with `--model`). `gate` is binary classification and a mini model is ample; `dedupe` must spot that two headlines in different languages sharing no words are one story; `synthesis` is where the capable model earns its cost.

**stdout is reconfigured to UTF-8 in `main.py`.** Windows consoles default to a legacy code page and the renderer's box drawing and emoji raise `UnicodeEncodeError` there — after every LLM call has already been paid for.

## Editing config.yaml

Per-source keys are documented at the top of the file. The distinctions that matter:

- `match` is a keyword allowlist applied **before** the per-feed cap — needed for firehose feeds like arXiv cs.AI, where "newest 3 of 150/day" is close to random sampling.
- `gate` is a free-text rule judged by an LLM call — for sources where the topic is not reliably in the title (PewDiePie is the motivating case).
- `preserve_order: true` skips the date re-sort. Hacker News and lobste.rs return point/hotness-ranked feeds; the arrival order *is* the quality signal.
- `aggregator: true` displays the origin domain (`mistral.ai via Hacker News`), which the model sees and uses to call out vendor announcements.
- `audience`, `interests`, and `priorities` are three separate blocks on purpose. `priorities` is a ranking weight, not a topic; folding it into `interests` got Norwegian stories consistently dropped. `audience` is context for second-order judgement, and the prompt explicitly forbids treating it as a topic filter.

## Conventions

Comments in this codebase explain *why*, usually citing the concrete failure that motivated the code. Match that — a comment restating what the line does is noise here. Renderers stay isolated from synthesis so new delivery targets build against the `Brief` dataclass.
