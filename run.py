#!/usr/bin/env python3
"""CLI:  python run.py "impact of solar energy in East Africa" --depth standard"""
from __future__ import annotations

import argparse
import sys

from research_agent.config import Config
from research_agent.orchestrator import ResearchOrchestrator


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent deep research -> PDF report")
    ap.add_argument("topic", nargs="?", help="idea or topic to research")
    ap.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard")
    ap.add_argument("--provider", choices=["gemini", "mock"], help="override LLM_PROVIDER")
    ap.add_argument("--model", help="override GEMINI_MODEL")
    ap.add_argument("--max-sources", type=int)
    ap.add_argument("--max-datasets", type=int)
    ap.add_argument("--rpm", type=int, help="LLM requests per minute")
    ap.add_argument("--out", help="output folder (default: runs/)")
    ap.add_argument("--demo", action="store_true", help="offline demo: fake search/downloads + mock LLM (no keys, no internet)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if not args.topic:
        args.topic = input("Research topic / idea: ").strip()

    cfg = Config().apply_depth(args.depth)
    if args.demo:
        cfg.offline_demo, cfg.llm_provider = True, "mock"
    if args.provider:
        cfg.llm_provider = args.provider
    if args.model:
        cfg.gemini_model = args.model
    if args.max_sources:
        cfg.max_sources = args.max_sources
    if args.max_datasets is not None:
        cfg.max_datasets = args.max_datasets
    if args.rpm:
        cfg.requests_per_minute = args.rpm
    if args.out:
        from pathlib import Path

        cfg.runs_dir = Path(args.out)
    cfg.verbose = not args.quiet

    try:
        pdf = ResearchOrchestrator(args.topic, cfg).run()
    except Exception as e:  # noqa: BLE001
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    print(f"\nDone. Report: {pdf}\nAll artifacts (sources, datasets, charts, state.json) are in: {pdf.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
