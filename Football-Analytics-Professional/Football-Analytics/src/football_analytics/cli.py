from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_settings
from .data import load_events
from .datasets import build_dataset_manifest
from .demo import generate_demo
from .forecasting import season_projection
from .match_forecasting import load_cached_match_history, write_cached_forecasts
from .training import retrain_and_recalculate
from .providers.footystats import fetch_premier_league_once
from .quality import validate_events
from .reporting import build_match_report
from .service import clear_cache, datasets, enriched_events
from .xg import save_artifact, train_xg


def main():
    parser = argparse.ArgumentParser(prog="football-analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("generate-demo", help="Generate clearly labelled synthetic event data")
    demo.add_argument("--matches", type=int, default=38); demo.add_argument("--seed", type=int, default=42)
    sub.add_parser("validate-data", help="Run data-quality controls")
    sub.add_parser("train-xg", help="Train, temporally evaluate and save the xG model")
    import_fs = sub.add_parser("import-footystats", help="Fetch and cache Premier League match history once")
    import_fs.add_argument("--league-id", type=int, action="append", required=True, dest="league_ids")
    import_fs.add_argument("--force", action="store_true", help="Deliberately replace an existing provider cache")
    train_matches = sub.add_parser("train-match-models", help="Train match forecast models from the local FootyStats cache")
    train_matches.add_argument("--test-fraction", type=float, default=0.25)
    projection = sub.add_parser("project-season", help="Create a probabilistic next-season projection")
    projection.add_argument("--output", default="reports/season_projection.csv")
    report = sub.add_parser("build-report", help="Build a self-contained match report")
    report.add_argument("--match-id", required=True); report.add_argument("--output")
    args = parser.parse_args(); settings = load_settings()

    if args.command == "generate-demo":
        if not settings.is_demo:
            raise SystemExit("Refusing to overwrite historical mode. Set data.mode='demo'.")
        matches, events, _ = generate_demo(settings.demo_data_dir, args.matches, args.seed)
        clear_cache(); print(f"Generated {len(matches)} synthetic matches and {len(events):,} synthetic events in {settings.demo_data_dir}")
    elif args.command == "import-footystats":
        api_key = os.getenv("FOOTYSTATS_API_KEY", "")
        result = fetch_premier_league_once(
            api_key=api_key,
            league_ids=args.league_ids,
            raw_path=settings.footystats_raw_cache,
            matches_path=settings.footystats_matches_cache,
            force=args.force,
        )
        history = load_cached_match_history(result.matches_path)
        season_path, fixtures_path = write_cached_forecasts(history, settings.report_dir)
        bundle, ml_paths = retrain_and_recalculate(
            history,
            model_dir=settings.model_dir,
            report_dir=settings.report_dir,
            random_state=settings.random_state,
            test_fraction=settings.test_size,
        )
        print(
            f"Cached {result.match_count} Premier League matches from seasons "
            f"{', '.join(result.seasons)} using {result.request_count} paginated requests. "
            "The provider cache was then used locally to train the match models. "
            f"Statistical forecasts: {season_path}, {fixtures_path}. "
            f"ML bundle: {ml_paths['bundle']}."
        )
        print(json.dumps(bundle.metrics.__dict__, indent=2))
    elif args.command == "train-match-models":
        history = load_cached_match_history(settings.footystats_matches_cache)
        bundle, paths = retrain_and_recalculate(
            history,
            model_dir=settings.model_dir,
            report_dir=settings.report_dir,
            random_state=settings.random_state,
            test_fraction=args.test_fraction,
        )
        print(json.dumps(bundle.metrics.__dict__, indent=2))
        print("Saved local ML artifacts:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
    elif args.command == "validate-data":
        events = load_events(settings.data_dir / settings.events_file)
        quality = validate_events(events); print(json.dumps(quality.to_dict(), indent=2)); raise SystemExit(0 if quality.passed else 1)
    elif args.command == "train-xg":
        events = load_events(settings.data_dir / settings.events_file)
        artifact = train_xg(events, settings.random_state, settings.test_size, settings.evaluation_season if not settings.is_demo else None)
        model_path, metadata_path = save_artifact(artifact, settings.model_dir)
        match_frame, _, _ = datasets(); build_dataset_manifest(events, match_frame, settings.data_mode, settings.model_dir / "dataset_manifest.json")
        print(json.dumps(artifact.metrics, indent=2)); print(f"Saved model to {model_path} and metadata to {metadata_path}")
    elif args.command == "project-season":
        matches, _, _ = datasets(); _, shots = enriched_events(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        season_projection(matches, shots).to_csv(output, index=False); print(f"Projection written to {output}")
    elif args.command == "build-report":
        events, shots = enriched_events(); output = Path(args.output) if args.output else settings.report_dir / f"{args.match_id}_analysis.html"
        print(f"Report written to {build_match_report(events, shots, args.match_id, output)}")


if __name__ == "__main__":
    main()
