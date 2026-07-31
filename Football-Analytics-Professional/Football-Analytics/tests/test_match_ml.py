from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analytics.match_features import FEATURE_COLUMNS, build_prematch_features, latest_fixture_features
from football_analytics.match_models import forecast_with_match_models, train_match_models
from football_analytics.training import retrain_and_recalculate


def _history(match_count: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    teams = [f"Team {index}" for index in range(10)]
    rows = []
    date = pd.Timestamp("2023-08-01")
    for index in range(match_count):
        home = teams[index % len(teams)]
        away = teams[(index * 3 + 1) % len(teams)]
        if away == home:
            away = teams[(teams.index(home) + 1) % len(teams)]
        home_rate = 1.1 + 0.08 * teams.index(home)
        away_rate = 0.9 + 0.05 * teams.index(away)
        rows.append(
            {
                "match_id": f"M{index:04d}",
                "season": "2023/2024",
                "match_date": date + pd.Timedelta(days=index * 2),
                "home_team": home,
                "away_team": away,
                "home_goals": int(rng.poisson(home_rate)),
                "away_goals": int(rng.poisson(away_rate)),
                "game_week": index // 5 + 1,
            }
        )
    return pd.DataFrame(rows)


def test_features_are_prematch_and_complete():
    result = build_prematch_features(_history())
    assert len(result.frame) == 120
    assert set(FEATURE_COLUMNS).issubset(result.frame.columns)
    assert result.frame.iloc[0]["home_matches_played"] == 0
    assert result.frame.iloc[-1]["home_matches_played"] > 0


def test_latest_fixture_features_returns_one_row():
    history = _history()
    result = latest_fixture_features(history, "Team 0", "Team 1")
    assert result.shape == (1, len(FEATURE_COLUMNS))


def test_train_and_forecast_probabilities_sum_to_one():
    history = _history()
    bundle = train_match_models(history, test_fraction=0.2)
    forecast = forecast_with_match_models(bundle, history, "Team 0", "Team 1")
    assert forecast.home_expected_goals > 0
    assert forecast.away_expected_goals > 0
    assert forecast.home_win + forecast.draw + forecast.away_win == pytest.approx(1.0)
    assert 0 <= forecast.both_teams_to_score <= 1
    assert 0 <= forecast.over_2_5 <= 1


def test_training_pipeline_writes_artifacts(tmp_path):
    bundle, paths = retrain_and_recalculate(
        _history(),
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        test_fraction=0.2,
    )
    assert bundle.metrics.test_matches > 0
    assert all(path.exists() for path in paths.values())
