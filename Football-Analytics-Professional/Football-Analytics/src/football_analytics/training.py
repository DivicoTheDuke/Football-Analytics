from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from .match_models import (
    MatchModelBundle,
    forecast_with_match_models,
    ml_season_projection,
    save_match_model_bundle,
    train_match_models,
)


def retrain_and_recalculate(
    matches: pd.DataFrame,
    *,
    model_dir: str | Path,
    report_dir: str | Path,
    random_state: int = 42,
    test_fraction: float = 0.25,
) -> tuple[MatchModelBundle, dict[str, Path]]:
    """Train locally from cached matches and regenerate all match-level reports."""
    bundle = train_match_models(
        matches,
        random_state=random_state,
        test_fraction=test_fraction,
    )
    bundle_path, metadata_path = save_match_model_bundle(bundle, model_dir)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    season_path = report_dir / "ml_season_projection.csv"
    fixture_path = report_dir / "ml_fixture_forecasts.csv"
    metrics_path = report_dir / "ml_match_model_metrics.json"

    projection = ml_season_projection(bundle, matches)
    projection.to_csv(season_path, index=False)

    latest_season = matches.sort_values("match_date")["season"].dropna().astype(str).iloc[-1]
    latest = matches[matches["season"].astype(str).eq(latest_season)]
    teams = sorted(set(latest["home_team"]) | set(latest["away_team"]))
    fixture_rows: list[dict[str, object]] = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            forecast = forecast_with_match_models(bundle, matches, home, away)
            fixture_rows.append(asdict(forecast))
    pd.DataFrame(fixture_rows).to_csv(fixture_path, index=False)
    metrics_path.write_text(json.dumps(asdict(bundle.metrics), indent=2), encoding="utf-8")

    return bundle, {
        "bundle": bundle_path,
        "metadata": metadata_path,
        "season_projection": season_path,
        "fixture_forecasts": fixture_path,
        "metrics": metrics_path,
    }
