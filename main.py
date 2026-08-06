"""Tech Brief — manual run, terminal output.

  python main.py                 # fetch, gate, dedupe, synthesise, print
  python main.py --dry           # ingestion + gate only, no synthesis call
  python main.py --hours 72      # widen the window for a catch-up run
  python main.py --json          # machine-readable, for piping/inspection
  python main.py --why <url>     # what a briefed item said when it was briefed
  python main.py --post          # render and send to Discord (bare --post: Discord only,
                                  # preserving every existing cron/systemd invocation)
  python main.py --post slack    # render and send to Slack instead
  python main.py --post both     # render and send to both; a failure on one
                                  # target does not stop the attempt on the other
  python main.py --discord-dry   # print the Discord payloads; no network
  python main.py --slack-dry     # print the Slack payloads; no network
  python main.py --replay out.json --slack-dry   # iterate on formatting only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from brief import ROOT, deliver, dedupe, gate, memory, render, runlog, sources, synth, verify
from brief.warnings import Warning

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
    parser.add_argument("--post", nargs="?", const="discord", default=None,
                        choices=("discord", "slack", "both"),
                        help="render and send today's brief. Bare --post posts to Discord "
                             "only (preserves every existing invocation); pass 'slack' or "
                             "'both' explicitly for the rest")
    parser.add_argument("--discord-dry", action="store_true",
                        help="print the Discord payloads that would be sent; no network")
    parser.add_argument("--slack-dry", action="store_true",
                        help="print the Slack payloads that would be sent; no network")
    parser.add_argument("--replay", metavar="PATH",
                        help="render a brief saved with --json; skips the whole pipeline")
    parser.add_argument("--force", action="store_true",
                        help="post even if today's brief was already posted")
    return parser.parse_args(argv)


def _run_pipeline(args, config, *, write_memory: bool = True):
    """Ingest through verify.

    Returns (brief, considered, warnings, flagged, clusters) on success, or an
    int when the caller should stop immediately and return it as-is (0 after
    --dry printed its listing, 1 on a hard failure) — never raises, so main()
    keeps a single always-returns-an-int contract.

    write_memory is distinct from args.no_memory: no_memory also skips
    READING seen.json (an explicit "ignore history" request), while
    write_memory=False still reads it — a dry render should filter exactly
    like the real run it's previewing would — and only suppresses the write
    at the end, so --discord-dry/--slack-dry can never mark anything as
    briefed for a message that was never sent.
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
            warnings.append(Warning(f"memory: skipped {already} item(s) briefed previously"))

    if args.dry:
        print(render.raw_listing(items, warnings))
        return 0

    clusters: list[dict] = []
    if not args.no_dedupe:
        items, dedupe_warnings, clusters = dedupe.merge(
            items, model=model_for(llm_cfg, "dedupe", args.model))
        warnings.extend(dedupe_warnings)

    if not items:
        print("Nothing in the window. Try --hours 72.", file=sys.stderr)
        for warning in warnings:
            print(f"  ⚠ {warning.text}", file=sys.stderr)
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

    if not args.no_memory and write_memory:
        memory.save(memory.prune(memory.record(seen, brief)))

    # Unconditional: the audit trail's whole point is surviving past whatever
    # the run was for, and unlike seen.json a missed write here has no
    # correctness consequence for tomorrow's run — no reason to gate it on
    # --no-memory or a dry render.
    runlog.record(clusters, brief)
    runlog.prune()

    return brief, len(items), warnings, flagged, clusters


_RENDER_FOR_TARGET = {"discord": render.to_discord, "slack": render.to_slack}
_TARGET_FOR_NAME = {"discord": deliver.DISCORD, "slack": deliver.SLACK}


def _post_to(name: str, brief, considered, warnings, flagged, *, force: bool) -> int:
    target = _TARGET_FOR_NAME[name]
    if deliver.already_posted_today(target) and not force:
        print(f"Already posted today's brief to {name}; use --force to repost.",
              file=sys.stderr)
        return 0

    payloads = _RENDER_FOR_TARGET[name](brief, considered=considered, warnings=warnings,
                                        flagged=flagged)
    ok = deliver.post(payloads, target)
    if ok:
        deliver.mark_posted(target)
        print(f"Posted to {name.capitalize()} ({len(payloads)} message(s)).")
    else:
        print(f"{name.capitalize()} delivery failed; the brief was not marked as posted.",
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
        # dedupe_clusters is audit data for a live pipeline run and isn't part
        # of the replay schema; a replayed brief just has none to show.
        text = Path(args.replay).read_text(encoding="utf-8")
        brief, considered, warnings, flagged = render.replay(text)
        clusters: list[dict] = []
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        # A dry render previews what --post would send; it must never mark
        # anything as briefed for a message that was never actually posted.
        write_memory = not (args.discord_dry or args.slack_dry)
        result = _run_pipeline(args, config, write_memory=write_memory)
        if isinstance(result, int):
            return result
        brief, considered, warnings, flagged, clusters = result

    if args.discord_dry:
        payloads = render.to_discord(brief, considered=considered, warnings=warnings,
                                     flagged=flagged)
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        return 0

    if args.slack_dry:
        payloads = render.to_slack(brief, considered=considered, warnings=warnings,
                                   flagged=flagged)
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        return 0

    if args.post:
        names = ("discord", "slack") if args.post == "both" else (args.post,)
        failed = False
        for name in names:
            rc = _post_to(name, brief, considered, warnings, flagged, force=args.force)
            failed = failed or rc != 0
        return 1 if failed else 0

    print(render.as_json(brief, considered=considered, warnings=warnings, flagged=flagged,
                         clusters=clusters)
          if args.json
          else render.to_terminal(brief, considered=considered, warnings=warnings,
                                  flagged=flagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
