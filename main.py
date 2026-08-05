"""Tech Brief — iteration 1. Manual run, terminal output.

  python main.py                 # fetch, synthesise, print
  python main.py --dry           # ingestion only, no LLM call (free)
  python main.py --hours 72      # widen the window for a catch-up run
  python main.py --json          # machine-readable, for piping/inspection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

import render
import sources
import synth

load_dotenv(override=True)  # override: dotenv cache can serve stale values

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def as_json(brief: synth.Brief) -> str:
    def node(entry):
        return {
            "id": entry.item.id,
            "headline": entry.headline,
            "comment": entry.comment,
            "source": {"name": entry.item.source, "url": entry.item.url},
            "title": entry.item.title,
        }
    return json.dumps({
        "top_signal": [node(e) for e in brief.top],
        "also_worth_knowing": [node(e) for e in brief.also],
        "video": node(brief.video) if brief.video else None,
        "meta_note": brief.meta,
    }, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--dry", action="store_true",
                        help="ingest and list only; no LLM call")
    parser.add_argument("--hours", type=int, help="override the lookback window")
    parser.add_argument("--per-feed", type=int, help="override per-feed item cap")
    parser.add_argument("--model", help="override the model")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of prose")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    window = config.get("window", {})
    llm_cfg = config.get("llm", {})

    items, warnings = sources.collect(
        config["sources"],
        hours=args.hours or window.get("hours", 36),
        per_feed=args.per_feed or window.get("per_feed", 8),
    )

    if args.dry:
        print(render.raw_listing(items, warnings))
        return 0

    if not items:
        print("Nothing in the window. Try --hours 72.", file=sys.stderr)
        for warning in warnings:
            print(f"  ⚠ {warning}", file=sys.stderr)
        return 1

    try:
        brief = synth.build(
            items,
            config["interests"],
            model=args.model or llm_cfg.get("model"),
            temperature=llm_cfg.get("temperature"),
            max_output_tokens=llm_cfg.get("max_output_tokens"),
        )
    except (synth.SynthError, Exception) as exc:
        print(f"Synthesis failed: {exc}", file=sys.stderr)
        return 1

    print(as_json(brief) if args.json
          else render.to_terminal(brief, considered=len(items), warnings=warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
