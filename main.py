"""Tech Brief — manual run, terminal output.

  python main.py                 # fetch, gate, dedupe, synthesise, print
  python main.py --dry           # ingestion + gate only, no synthesis call
  python main.py --hours 72      # widen the window for a catch-up run
  python main.py --json          # machine-readable, for piping/inspection
  python main.py --why <url>     # what a briefed item said when it was briefed
  python main.py --post          # render and send to Discord
  python main.py --discord-dry   # print the Discord payloads; no network
  python main.py --replay out.json --discord-dry   # iterate on formatting only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from brief import ROOT, deliver, dedupe, gate, memory, render, sources, synth, verify

load_dotenv(override=True)  # override: dotenv cache can serve stale values

# Windows consoles still default to a legacy code page, and the renderer's box
# drawing, emoji and en-dashes raise UnicodeEncodeError there — the brief dies
# at the print, after every LLM call has already been paid for.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def model_for(llm_cfg: dict, stage: str, override: str | None) -> str | None:
    """CLI override wins, then the per-stage setting, then the global default."""
    if override:
        return override
    return (llm_cfg.get("stages") or {}).get(stage) or llm_cfg.get("model")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--dry", action="store_true",
                        help="ingest and list only; no synthesis call")
    parser.add_argument("--hours", type=int, help="override the lookback window")
    parser.add_argument("--per-feed", type=int, help="override per-feed item cap")
    parser.add_argument("--model", help="override the model for every stage")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="skip the duplicate-clustering pre-pass")
    parser.add_argument("--no-gate", action="store_true",
                        help="skip per-source LLM relevance gating")
    parser.add_argument("--no-memory", action="store_true",
                        help="ignore and do not update the briefed-items history")
    parser.add_argument("--forget", action="store_true",
                        help="clear the briefed-items history and exit")
    parser.add_argument("--why", metavar="URL",
                        help="print the source snapshot recorded when URL was briefed")
    parser.add_argument("--post", action="store_true",
                        help="render and send today's brief to Discord")
    parser.add_argument("--discord-dry", action="store_true",
                        help="print the Discord payloads that would be sent; no network")
    parser.add_argument("--replay", metavar="PATH",
                        help="render a brief saved with --json; skips the whole pipeline")
    parser.add_argument("--force", action="store_true",
                        help="post even if today's brief was already posted")
    return parser.parse_args(argv)


def _run_pipeline(args, config):
    """Ingest through verify.

    Returns (brief, considered, warnings, flagged) on success, or an int when
    the caller should stop immediately and return it as-is (0 after --dry
    printed its listing, 1 on a hard failure) — never raises, so main() keeps
    a single always-returns-an-int contract.
    """
    window = config.get("window", {})
    llm_cfg = config.get("llm", {})

    items, warnings = sources.collect(
        config["sources"],
        hours=args.hours or window.get("hours", 48),
        per_feed=args.per_feed or window.get("per_feed", 12),
    )

    # Gate and memory both run before --dry so the dry listing reflects the real
    # candidate pool, not everything ingestion happened to find.
    if not args.no_gate:
        items, gate_warnings = gate.apply(
            items, sources.gate_rules(config["sources"]),
            model=model_for(llm_cfg, "gate", args.model),
        )
        warnings.extend(gate_warnings)

    seen = {} if args.no_memory else memory.load()
    if seen:
        items, already = memory.filter_unseen(items, seen)
        if already:
            warnings.append(f"memory: skipped {already} item(s) briefed previously")

    if args.dry:
        print(render.raw_listing(items, warnings))
        return 0

    if not args.no_dedupe:
        items, note = dedupe.merge(items, model=model_for(llm_cfg, "dedupe", args.model))
        if note:
            warnings.append(note)

    if not items:
        print("Nothing in the window. Try --hours 72.", file=sys.stderr)
        for warning in warnings:
            print(f"  ⚠ {warning}", file=sys.stderr)
        return 1

    try:
        brief = synth.build(
            items,
            config["interests"],
            config.get("priorities", ""),
            config.get("audience", ""),
            model=model_for(llm_cfg, "synthesis", args.model),
            temperature=llm_cfg.get("temperature"),
            max_output_tokens=llm_cfg.get("max_output_tokens"),
        )
    except synth.SynthError as exc:
        print(f"Synthesis failed: {exc}", file=sys.stderr)
        return 1

    # Promote before verifying and before recording: promotion changes which URL
    # ships, and both the flags and the stored snapshot must describe that URL.
    brief = verify.promote_leads(brief)
    warnings.extend(synth.budget_warnings(brief))
    flagged, fact_warnings = verify.check(brief)
    warnings.extend(fact_warnings)

    if not args.no_memory:
        memory.save(memory.prune(memory.record(seen, brief)))

    return brief, len(items), warnings, flagged


def _post_to_discord(brief, considered, warnings, flagged, *, force: bool) -> int:
    if deliver.already_posted_today() and not force:
        print("Already posted today's brief; use --force to repost.", file=sys.stderr)
        return 0

    payloads = render.to_discord(brief, considered=considered, warnings=warnings,
                                 flagged=flagged)
    ok = deliver.post(payloads)
    if ok:
        deliver.mark_posted()
        print(f"Posted to Discord ({len(payloads)} message(s)).")
    else:
        print("Discord delivery failed; the brief was not marked as posted.",
              file=sys.stderr)
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.forget:
        memory.save({})
        print("Cleared briefed-items history.")
        return 0

    # The snapshot exists to be looked up rather than inferred — a feed that
    # revises a figure mid-day is how a correct fact gets called a hallucination.
    if args.why:
        record = memory.snapshot(memory.load(), args.why)
        if record is None:
            print(f"Not in the briefed-items history: {args.why}", file=sys.stderr)
            return 1
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0

    if args.replay:
        # Skips ingestion, gating, dedupe, synthesis and verification entirely —
        # the formatting-iteration loop for a renderer, not a real run, so it
        # must not touch memory or the posted-today marker's underlying brief.
        text = Path(args.replay).read_text(encoding="utf-8")
        brief, considered, warnings, flagged = render.replay(text)
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        result = _run_pipeline(args, config)
        if isinstance(result, int):
            return result
        brief, considered, warnings, flagged = result

    if args.discord_dry:
        payloads = render.to_discord(brief, considered=considered, warnings=warnings,
                                     flagged=flagged)
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        return 0

    if args.post:
        return _post_to_discord(brief, considered, warnings, flagged, force=args.force)

    print(render.as_json(brief, considered=considered, warnings=warnings, flagged=flagged)
          if args.json
          else render.to_terminal(brief, considered=considered, warnings=warnings,
                                  flagged=flagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
