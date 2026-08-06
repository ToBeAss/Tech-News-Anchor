# Fyrtårn

A personal daily tech brief: RSS/Atom feeds in, a ranked and annotated brief
out — printed to the terminal, or posted to Discord and/or Slack.

RSS/Atom feeds (including YouTube channels, which expose a per-channel Atom
feed) are fetched, filtered, and deduplicated, then a single LLM call ranks
and writes up the day's top stories. The model never sees a URL — items are
labelled `[i01]`, `[i02]`… and it returns only ids — so a fabricated or
mis-attributed link is structurally impossible, not merely discouraged. Every
figure it writes is checked mechanically against the source text it was
actually shown.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env                # fill in OPENAI_API_KEY, and webhooks if you post
```

`.env` holds secrets and is gitignored. Everything else — sources, window,
model per stage, the audience/interests/priorities prompt blocks — lives in
[config.yaml](config.yaml).

## Usage

```bash
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

There is no linter or build step. `--dry --no-gate` is the free feedback loop
when changing ingestion, `--json` when changing synthesis.

### Posting to Discord / Slack

```bash
python main.py --post                # send to Discord (bare --post; the original behaviour)
python main.py --post slack          # send to Slack instead
python main.py --post both           # send to both — one target failing doesn't stop the other
python main.py --discord-dry         # print the Discord payloads; no network, no posting
python main.py --slack-dry           # print the Slack payloads; no network, no posting
python main.py --replay out.json --slack-dry   # iterate on formatting only, no pipeline run
```

Posting needs `DISCORD_WEBHOOK` and/or `SLACK_WEBHOOK` set in `.env`. Each
target tracks its own "already posted today" marker under `state/`, so
posting to one does not suppress the other; `--force` reposts regardless.
`--replay` reconstructs a brief from a `--json` dump, skipping ingestion
through verification entirely — the formatting-iteration loop for a renderer,
not a real run.

## Layout

```
main.py              CLI and composition root — the only place stages are wired
config.yaml           sources, window, model-per-stage, and the four prompt blocks
brief/                the pipeline; ROOT (project root) is defined in __init__.py
brief/render/          output targets — terminal, JSON, Discord, Slack
brief/deliver.py       webhook delivery (network) for the render/ payloads
state/seen.json        briefed-items history (gitignored)
tests/                 unittest suite for the renderers and delivery
```

## Pipeline

1. **`brief/sources.py`** — fetch every feed, filter by window and per-source
   keywords, canonicalise URLs, dedupe by URL, assign ids.
2. **`brief/gate.py`** — one small LLM call per source that has a `gate:`
   rule, dropping off-topic items.
3. **`brief/memory.py`** — drop items already briefed in a previous run.
4. **`brief/dedupe.py`** — one LLM call that groups ids covering the same
   underlying event; followers collapse into the leader's `related` tuple.
5. **`brief/synth.py`** — the main call: rank, select 3 + 6-9 grouped + 1
   video, write headline/fact/comment.
6. **`brief/verify.py`** — promote the merged sibling that supports the
   headline, then check every figure against the source text.
7. **`brief/render/`** — terminal, JSON, Discord, and Slack output.
8. **`brief/deliver.py`** — posts a renderer's payloads to a webhook, if asked.

Gate and dedupe fail open (a failed call passes items through with a
warning); synthesis is fatal — there is no brief without it. A dead feed
degrades the brief but never aborts the run.

## Tests

```bash
python -m unittest discover
```

Covers the Discord/Slack renderers (block/embed limits, escaping, section
splitting, degraded-run notices) and delivery (per-target success codes,
retry handling, independent posted-today markers). No network calls are made.

## Editing config.yaml

Per-source keys are documented at the top of the file. `match` is a keyword
allowlist applied before the per-feed cap (needed for firehose feeds like
arXiv cs.AI); `gate` is a free-text rule judged by an LLM call (for sources
where the topic isn't reliably in the title); `preserve_order: true` skips
the date re-sort for feeds that arrive pre-ranked (Hacker News, lobste.rs);
`aggregator: true` shows the origin domain (`mistral.ai via Hacker News`).

`audience`, `interests`, and `priorities` are three separate prompt blocks on
purpose — `priorities` is a ranking weight, not a topic, and `audience` is
context for second-order judgement, not a topic filter.
