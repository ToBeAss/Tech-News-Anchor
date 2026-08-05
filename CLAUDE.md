# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal daily tech brief: RSS/Atom feeds in, a ranked and annotated brief out, printed to the terminal. Iteration 1 is manual runs only; the code is structured so iteration 2 can add Discord/Teams delivery without touching the pipeline.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt

python main.py                 # full run: fetch, gate, dedupe, synthesise, print
python main.py --dry           # ingestion + gate only, no synthesis call (nearly free)
python main.py --json          # machine-readable output
python main.py --hours 72      # widen the lookback window for a catch-up run
python main.py --model gpt-5.4-mini   # override the model for every stage
python main.py --no-dedupe --no-gate --no-memory   # skip individual stages
python main.py --forget        # clear state/seen.json and exit
```

There is no test suite, linter, or build step. `--dry` is the cheap feedback loop when changing ingestion, and `--json` when changing synthesis.

`OPENAI_API_KEY` comes from `.env` (see `.env.example`). Everything else lives in [config.yaml](config.yaml).

## Pipeline

[main.py](main.py) wires six stages, each in its own module. Every stage after ingestion is optional via a flag.

1. **[sources.py](sources.py)** — fetch every feed, filter by window and per-source keywords, canonicalise URLs, dedupe by URL, assign ids.
2. **[gate.py](gate.py)** — one small LLM call per source that has a `gate:` rule, dropping off-topic items. Runs *before* `--dry` so the dry listing reflects the real candidate pool.
3. **[memory.py](memory.py)** — drop items already briefed in a previous run. Also before `--dry`.
4. **[dedupe.py](dedupe.py)** — one LLM call that groups ids covering the same underlying event; followers collapse into the leader's `related` tuple.
5. **[synth.py](synth.py)** — the main call: rank, select 3 + 5 + 1 video, write headline/fact/comment.
6. **[verify.py](verify.py)** — mechanical check that every figure in the written text appears in the source text. Runs *before* memory is recorded.
7. **[render.py](render.py)** — terminal output.

[llm.py](llm.py) is the only network path to the model: raw `requests` against the OpenAI Responses API, no SDK (this is meant to run on a Raspberry Pi).

## Design invariants

These are load-bearing; several exist because the obvious alternative failed in practice.

**The model never sees URLs.** Items are labelled `[i01]`, `[i02]`… and the model returns only ids. Anything referencing an unknown id is dropped in `synth._validate`. A fabricated or mis-attributed link is therefore structurally impossible, not merely discouraged. Preserve this when adding stages.

**Numbers are verified mechanically, not by prompting.** `verify.check` requires every figure in a headline/fact/comment to appear in the item's own title or summary (separator-insensitive, years and name-embedded digits like `kode24`/`GPT-4` excluded). Same philosophy as the URL rule: make the error detectable rather than asking the model to be careful. `check_lead` additionally flags figures supported only by a *merged sibling* — the reader clicks the primary link, so a number that lives only in the other outlet's article is a broken promise.

**Failure modes are chosen per stage, deliberately:**
- Ingestion — a dead feed degrades the brief but never aborts the run; `UNREACHABLE` warnings surface at the *top* of the output, not in a footer, because a reduced pool changes how much to trust the ranking.
- Gate and dedupe — **fail-open**. A failed call passes items through with a warning. A silently discarded item is gone for good; an off-topic one that slips past will very likely be dropped by ranking anyway.
- Synthesis — fatal.

**Caps are enforced in code, not only in the prompt.** `synth._validate` truncates sections to `TARGET_TOP`/`TARGET_ALSO`, backfills an under-filled `top_signal` by promoting from `also` (placement only — it never invents entries), reserves the video slot by falling back to the newest unused video candidate, and reports a `shortfall`.

**Memory records only what was actually briefed**, keyed by canonical URL, with a 400-char snapshot of the source text *as it read at the time*. Feeds mutate — digi.no revised a contract figure between two runs on the same day — and without the snapshot there is no way to answer "what did the model actually see?" A merged item marks all its sibling URLs as seen, so tomorrow's run cannot resurface the story through the other outlet's link.

**Model choice is per stage** (`llm.stages` in config, overridable with `--model`). `gate` is binary classification and a mini model is ample; `dedupe` must spot that two headlines in different languages sharing no words are one story; `synthesis` is where the capable model earns its cost.

## Editing config.yaml

Per-source keys are documented at the top of the file. The distinctions that matter:

- `match` is a keyword allowlist applied **before** the per-feed cap — needed for firehose feeds like arXiv cs.AI, where "newest 3 of 150/day" is close to random sampling.
- `gate` is a free-text rule judged by an LLM call — for sources where the topic is not reliably in the title (PewDiePie is the motivating case).
- `preserve_order: true` skips the date re-sort. Hacker News and lobste.rs return point/hotness-ranked feeds; the arrival order *is* the quality signal.
- `aggregator: true` displays the origin domain (`mistral.ai via Hacker News`), which the model sees and uses to call out vendor announcements.
- `audience`, `interests`, and `priorities` are three separate blocks on purpose. `priorities` is a ranking weight, not a topic; folding it into `interests` got Norwegian stories consistently dropped. `audience` is context for second-order judgement, and the prompt explicitly forbids treating it as a topic filter.

## Conventions

Comments in this codebase explain *why*, usually citing the concrete failure that motivated the code. Match that — a comment restating what the line does is noise here. Renderers stay isolated from synthesis so new delivery targets build against the `Brief` dataclass.
