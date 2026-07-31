from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings
from .data import load_events
from .demo import generate_demo
from .quality import validate_events
from .reporting import build_match_report
from .service import clear_cache, enriched_events
from .xg import train_xg, save_artifact


def main():
    parser = argparse.ArgumentParser(prog="football-analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("generate-demo", help="Generate synthetic event data")
    demo.add_argument("--matches", type=int, default=28)
    demo.add_argument("--seed", type=int, default=42)

    sub.add_parser("validate-data", help="Run data-quality controls")
    sub.add_parser("train-xg", help="Train, evaluate and save the xG model")

    report = sub.add_parser("build-report", help="Build a self-contained match report")
    report.add_argument("--match-id", required=True)
    report.add_argument("--output")

    args = parser.parse_args()
    settings = load_settings()

    if args.command == "generate-demo":
        matches, events, _ = generate_demo(settings.data_dir, args.matches, args.seed)
        clear_cache()
        print(f"Generated {len(matches)} matches and {len(events):,} events in {settings.data_dir}")

    elif args.command == "validate-data":
        events = load_events(settings.data_dir / "events.csv")
        report = validate_events(events)
        print(json.dumps(report.to_dict(), indent=2))
        raise SystemExit(0 if report.passed else 1)

    elif args.command == "train-xg":
        events = load_events(settings.data_dir / "events.csv")
        artifact = train_xg(events, settings.random_state, settings.test_size)
        model_path, metadata_path = save_artifact(artifact, settings.model_dir)
        print(json.dumps(artifact.metrics, indent=2))
        print(f"Saved model to {model_path} and metadata to {metadata_path}")

    elif args.command == "build-report":
        events, shots = enriched_events()
        output = Path(args.output) if args.output else settings.report_dir / f"{args.match_id}_analysis.html"
        path = build_match_report(events, shots, args.match_id, output)
        print(f"Report written to {path}")


if __name__ == "__main__":
    main()
