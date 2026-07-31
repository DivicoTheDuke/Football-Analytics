from __future__ import annotations

import json

import pandas as pd
import pytest

from football_analytics.match_forecasting import (
    forecast_from_match_history,
    season_projection_from_match_history,
)
from football_analytics.providers import footystats


def _rows() -> list[dict]:
    rows = []
    teams = ["Arsenal", "Chelsea", "Liverpool"]
    match_id = 1
    for season, year in [("2024/2025", 2024), ("2025/2026", 2025)]:
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append(
                    {
                        "id": match_id,
                        "home_name": home,
                        "away_name": away,
                        "season": season,
                        "status": "complete",
                        "date": f"{year}-09-{min(match_id, 28):02d}",
                        "homeGoalCount": 2 if home == "Arsenal" else 1,
                        "awayGoalCount": 0 if away == "Chelsea" else 1,
                    }
                )
                match_id += 1
    return rows


def test_normalise_and_forecast_cached_history():
    matches = footystats.normalise_footystats_matches(_rows())
    result = forecast_from_match_history(matches, "Arsenal", "Chelsea")
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0)
    assert 0 <= result.both_teams_to_score <= 1
    table = season_projection_from_match_history(matches)
    assert set(table["team"]) == {"Arsenal", "Chelsea", "Liverpool"}
    assert table.iloc[0]["projected_position"] == 1


def test_one_time_import_refuses_existing_cache(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="No provider request was made"):
        footystats.fetch_premier_league_once(
            api_key="example",
            league_ids=[1625],
            raw_path=raw_path,
            matches_path=tmp_path / "matches.csv",
        )


def test_one_time_import_caches_paginated_responses(tmp_path, monkeypatch):
    responses = [
        {"data": _rows()[:5], "pagination": {"page": 1, "total_pages": 2}},
        {"data": _rows()[5:], "pagination": {"page": 2, "total_pages": 2}},
    ]

    def fake_request(url: str, timeout_seconds: int = 30):
        return responses.pop(0)

    monkeypatch.setattr(footystats, "_request_json", fake_request)
    raw_path = tmp_path / "raw.json"
    matches_path = tmp_path / "matches.csv"
    result = footystats.fetch_premier_league_once(
        api_key="test-key",
        league_ids=[999],
        raw_path=raw_path,
        matches_path=matches_path,
    )
    assert result.request_count == 2
    assert result.match_count == len(_rows())
    assert matches_path.exists()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["request_count"] == 2
    cached = pd.read_csv(matches_path)
    assert cached["synthetic_data"].eq(False).all()
