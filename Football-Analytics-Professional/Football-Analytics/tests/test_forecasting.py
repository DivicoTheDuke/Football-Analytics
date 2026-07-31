import pytest

from football_analytics.forecasting import attacking_side_profile, forecast_fixture, scorer_probabilities, season_projection, team_style_profiles
from football_analytics.xg import train_xg, predict_xg


def _shots(events):
    artifact = train_xg(events, random_state=7, test_size=0.25)
    return predict_xg(artifact.model, events)


def test_fixture_probabilities_sum_close_to_one(demo_data):
    matches, events, _ = demo_data; shots = _shots(events)
    home, away = matches.iloc[0][["home_team", "away_team"]]
    result = forecast_fixture(matches, shots, home, away)
    assert result.home_xg > 0 and result.away_xg > 0
    assert result.home_win + result.draw + result.away_win == pytest.approx(1.0, abs=0.02)


def test_team_identity_outputs(demo_data):
    _, events, _ = demo_data
    sides = attacking_side_profile(events); styles = team_style_profiles(events)
    assert {"left_share", "centre_share", "right_share", "preferred_side"}.issubset(sides.columns)
    assert "style_label" in styles.columns


def test_season_and_scorer_projection(demo_data):
    matches, events, _ = demo_data; shots = _shots(events)
    table = season_projection(matches, shots)
    assert table["projected_position"].iloc[0] == 1
    team = table.iloc[0]["team"]
    scorers = scorer_probabilities(events, shots, team, 1.5)
    assert scorers["score_probability"].between(0, 1).all()
