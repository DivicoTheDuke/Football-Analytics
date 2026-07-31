from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchHistoryForecast:
    home_team: str
    away_team: str
    home_rate: float
    away_rate: float
    home_win: float
    draw: float
    away_win: float
    most_likely_score: str
    both_teams_to_score: float
    over_2_5: float


def load_cached_match_history(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached FootyStats match file not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cached match history is missing columns: {sorted(missing)}")
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    for column in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
    return frame.sort_values("match_date").reset_index(drop=True)


def _metric_columns(matches: pd.DataFrame) -> tuple[str, str, str]:
    if {
        "home_xg",
        "away_xg",
    }.issubset(matches.columns) and matches[["home_xg", "away_xg"]].notna().mean().min() >= 0.8:
        return "home_xg", "away_xg", "provider xG"
    return "home_goals", "away_goals", "goals proxy"


def _long_history(matches: pd.DataFrame, half_life_days: float = 730.5) -> tuple[pd.DataFrame, str]:
    home_metric, away_metric, metric_label = _metric_columns(matches)
    base = matches.copy()
    latest = base["match_date"].max()
    age_days = (latest - base["match_date"]).dt.days.fillna(0).clip(lower=0)
    base["weight"] = np.exp(-np.log(2.0) * age_days / half_life_days)

    home = pd.DataFrame(
        {
            "match_id": base["match_id"],
            "match_date": base["match_date"],
            "season": base["season"],
            "team": base["home_team"],
            "opponent": base["away_team"],
            "venue": "home",
            "for_rate": base[home_metric],
            "against_rate": base[away_metric],
            "weight": base["weight"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": base["match_id"],
            "match_date": base["match_date"],
            "season": base["season"],
            "team": base["away_team"],
            "opponent": base["home_team"],
            "venue": "away",
            "for_rate": base[away_metric],
            "against_rate": base[home_metric],
            "weight": base["weight"],
        }
    )
    return pd.concat([home, away], ignore_index=True), metric_label


def _shrunk_mean(group: pd.DataFrame, column: str, prior: float, prior_matches: float = 6.0) -> float:
    if group.empty:
        return prior
    weights = group["weight"].to_numpy(dtype=float)
    values = group[column].to_numpy(dtype=float)
    sample_weight = float(weights.sum())
    observed = float(np.average(values, weights=weights)) if sample_weight > 0 else prior
    return (observed * sample_weight + prior * prior_matches) / (sample_weight + prior_matches)


def _poisson(goals: int, rate: float) -> float:
    return exp(-rate) * rate**goals / factorial(goals)


def _score_matrix(home_rate: float, away_rate: float, max_goals: int = 12) -> np.ndarray:
    matrix = np.array(
        [
            [_poisson(i, home_rate) * _poisson(j, away_rate) for j in range(max_goals + 1)]
            for i in range(max_goals + 1)
        ],
        dtype=float,
    )
    return matrix / matrix.sum()


def forecast_from_match_history(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> MatchHistoryForecast:
    if home_team == away_team:
        raise ValueError("Home and away team must be different")
    long, _ = _long_history(matches)
    known = set(long["team"].astype(str))
    if home_team not in known or away_team not in known:
        raise KeyError("Both teams must exist in the cached FootyStats history")

    home_rows = long[long["venue"].eq("home")]
    away_rows = long[long["venue"].eq("away")]
    league_home = float(np.average(home_rows["for_rate"], weights=home_rows["weight"]))
    league_away = float(np.average(away_rows["for_rate"], weights=away_rows["weight"]))

    home_history = home_rows[home_rows["team"].eq(home_team)]
    away_history = away_rows[away_rows["team"].eq(away_team)]

    home_attack = _shrunk_mean(home_history, "for_rate", league_home) / max(league_home, 0.1)
    away_defence = _shrunk_mean(away_history, "against_rate", league_home) / max(league_home, 0.1)
    away_attack = _shrunk_mean(away_history, "for_rate", league_away) / max(league_away, 0.1)
    home_defence = _shrunk_mean(home_history, "against_rate", league_away) / max(league_away, 0.1)

    home_rate = float(np.clip(league_home * home_attack * away_defence, 0.2, 4.0))
    away_rate = float(np.clip(league_away * away_attack * home_defence, 0.2, 4.0))
    matrix = _score_matrix(home_rate, away_rate)

    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return MatchHistoryForecast(
        home_team=home_team,
        away_team=away_team,
        home_rate=home_rate,
        away_rate=away_rate,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        most_likely_score=f"{i}-{j}",
        both_teams_to_score=float(matrix[1:, 1:].sum()),
        over_2_5=float(
            sum(
                matrix[i, j]
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if i + j >= 3
            )
        ),
    )


def season_projection_from_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Project a double round-robin among teams in the latest cached season."""
    latest_season = (
        matches.sort_values("match_date")["season"].dropna().astype(str).iloc[-1]
    )
    latest = matches[matches["season"].astype(str).eq(latest_season)]
    teams = sorted(set(latest["home_team"]) | set(latest["away_team"]))
    rows = {
        team: {
            "team": team,
            "points": 0.0,
            "expected_goals": 0.0,
            "expected_goals_against": 0.0,
        }
        for team in teams
    }
    for home in teams:
        for away in teams:
            if home == away:
                continue
            forecast = forecast_from_match_history(matches, home, away)
            rows[home]["points"] += 3 * forecast.home_win + forecast.draw
            rows[away]["points"] += 3 * forecast.away_win + forecast.draw
            rows[home]["expected_goals"] += forecast.home_rate
            rows[home]["expected_goals_against"] += forecast.away_rate
            rows[away]["expected_goals"] += forecast.away_rate
            rows[away]["expected_goals_against"] += forecast.home_rate

    frame = pd.DataFrame(rows.values())
    frame["goal_difference"] = frame["expected_goals"] - frame["expected_goals_against"]
    frame = frame.sort_values(
        ["points", "goal_difference", "expected_goals"], ascending=False
    ).reset_index(drop=True)
    frame.insert(0, "projected_position", np.arange(1, len(frame) + 1))
    frame.attrs["latest_season"] = latest_season
    frame.attrs["metric_label"] = _metric_columns(matches)[2]
    return frame


def write_cached_forecasts(
    matches: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    season = season_projection_from_match_history(matches)
    season_path = output_dir / "footystats_season_projection.csv"
    season.to_csv(season_path, index=False)

    teams = season["team"].tolist()
    fixture_rows = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            forecast = forecast_from_match_history(matches, home, away)
            fixture_rows.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_rate": forecast.home_rate,
                    "away_rate": forecast.away_rate,
                    "home_win": forecast.home_win,
                    "draw": forecast.draw,
                    "away_win": forecast.away_win,
                    "most_likely_score": forecast.most_likely_score,
                    "both_teams_to_score": forecast.both_teams_to_score,
                    "over_2_5": forecast.over_2_5,
                }
            )
    fixtures_path = output_dir / "footystats_fixture_forecasts.csv"
    pd.DataFrame(fixture_rows).to_csv(fixtures_path, index=False)
    return season_path, fixtures_path
