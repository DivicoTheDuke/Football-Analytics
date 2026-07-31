from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "game_week",
    "home_matches_played",
    "away_matches_played",
    "home_points_per_game",
    "away_points_per_game",
    "home_goals_for_per_game",
    "away_goals_for_per_game",
    "home_goals_against_per_game",
    "away_goals_against_per_game",
    "home_home_goals_for_per_game",
    "home_home_goals_against_per_game",
    "away_away_goals_for_per_game",
    "away_away_goals_against_per_game",
    "home_recent_points_per_game",
    "away_recent_points_per_game",
    "home_recent_goals_for",
    "away_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_against",
    "points_per_game_difference",
    "recent_points_difference",
    "attack_difference",
    "defence_difference",
]


@dataclass(frozen=True)
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metric_label: str


def _points_for(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _mean(values: list[float], fallback: float) -> float:
    return float(np.mean(values)) if values else float(fallback)


def build_prematch_features(
    matches: pd.DataFrame,
    *,
    recent_window: int = 5,
) -> FeatureBuildResult:
    """Build leakage-safe features using only matches completed before each fixture.

    The implementation intentionally processes fixtures chronologically and updates
    team state only after a row has been created. Therefore the target match result
    can never leak into its own input features.
    """
    required = {
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Matches are missing required columns: {sorted(missing)}")

    frame = matches.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.dropna(
        subset=["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    ).sort_values(["match_date", "match_id"]).reset_index(drop=True)

    league_home_goal_prior = max(float(frame["home_goals"].mean()), 0.1)
    league_away_goal_prior = max(float(frame["away_goals"].mean()), 0.1)
    league_ppg_prior = 1.35

    state: dict[str, dict[str, list[float]]] = {}

    def team_state(team: str) -> dict[str, list[float]]:
        return state.setdefault(
            team,
            {
                "points": [],
                "goals_for": [],
                "goals_against": [],
                "home_goals_for": [],
                "home_goals_against": [],
                "away_goals_for": [],
                "away_goals_against": [],
            },
        )

    rows: list[dict[str, object]] = []

    for match in frame.itertuples(index=False):
        home = team_state(str(match.home_team))
        away = team_state(str(match.away_team))

        home_recent_points = home["points"][-recent_window:]
        away_recent_points = away["points"][-recent_window:]
        home_recent_gf = home["goals_for"][-recent_window:]
        away_recent_gf = away["goals_for"][-recent_window:]
        home_recent_ga = home["goals_against"][-recent_window:]
        away_recent_ga = away["goals_against"][-recent_window:]

        home_ppg = _mean(home["points"], league_ppg_prior)
        away_ppg = _mean(away["points"], league_ppg_prior)
        home_gf = _mean(home["goals_for"], league_home_goal_prior)
        away_gf = _mean(away["goals_for"], league_away_goal_prior)
        home_ga = _mean(home["goals_against"], league_away_goal_prior)
        away_ga = _mean(away["goals_against"], league_home_goal_prior)

        row = {
            "match_id": match.match_id,
            "season": str(match.season),
            "match_date": match.match_date,
            "home_team": str(match.home_team),
            "away_team": str(match.away_team),
            "game_week": float(getattr(match, "game_week", np.nan)),
            "home_matches_played": float(len(home["points"])),
            "away_matches_played": float(len(away["points"])),
            "home_points_per_game": home_ppg,
            "away_points_per_game": away_ppg,
            "home_goals_for_per_game": home_gf,
            "away_goals_for_per_game": away_gf,
            "home_goals_against_per_game": home_ga,
            "away_goals_against_per_game": away_ga,
            "home_home_goals_for_per_game": _mean(
                home["home_goals_for"], league_home_goal_prior
            ),
            "home_home_goals_against_per_game": _mean(
                home["home_goals_against"], league_away_goal_prior
            ),
            "away_away_goals_for_per_game": _mean(
                away["away_goals_for"], league_away_goal_prior
            ),
            "away_away_goals_against_per_game": _mean(
                away["away_goals_against"], league_home_goal_prior
            ),
            "home_recent_points_per_game": _mean(home_recent_points, league_ppg_prior),
            "away_recent_points_per_game": _mean(away_recent_points, league_ppg_prior),
            "home_recent_goals_for": _mean(home_recent_gf, league_home_goal_prior),
            "away_recent_goals_for": _mean(away_recent_gf, league_away_goal_prior),
            "home_recent_goals_against": _mean(home_recent_ga, league_away_goal_prior),
            "away_recent_goals_against": _mean(away_recent_ga, league_home_goal_prior),
            "points_per_game_difference": home_ppg - away_ppg,
            "recent_points_difference": _mean(home_recent_points, league_ppg_prior)
            - _mean(away_recent_points, league_ppg_prior),
            "attack_difference": home_gf - away_ga,
            "defence_difference": away_gf - home_ga,
            "home_goals": int(match.home_goals),
            "away_goals": int(match.away_goals),
        }
        if match.home_goals > match.away_goals:
            row["outcome"] = "H"
        elif match.home_goals < match.away_goals:
            row["outcome"] = "A"
        else:
            row["outcome"] = "D"
        rows.append(row)

        home_points = _points_for(match.home_goals, match.away_goals)
        away_points = _points_for(match.away_goals, match.home_goals)
        home["points"].append(home_points)
        away["points"].append(away_points)
        home["goals_for"].append(float(match.home_goals))
        home["goals_against"].append(float(match.away_goals))
        away["goals_for"].append(float(match.away_goals))
        away["goals_against"].append(float(match.home_goals))
        home["home_goals_for"].append(float(match.home_goals))
        home["home_goals_against"].append(float(match.away_goals))
        away["away_goals_for"].append(float(match.away_goals))
        away["away_goals_against"].append(float(match.home_goals))

    result = pd.DataFrame(rows)
    result["game_week"] = result["game_week"].fillna(
        result.groupby("season").cumcount().add(1).astype(float)
    )
    return FeatureBuildResult(
        frame=result,
        feature_columns=tuple(FEATURE_COLUMNS),
        metric_label="historical goals and pre-match rolling form",
    )


def latest_fixture_features(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    *,
    recent_window: int = 5,
) -> pd.DataFrame:
    """Build one future-fixture row from all locally cached completed matches."""
    if home_team == away_team:
        raise ValueError("Home and away team must be different")
    base = matches.copy()
    known = set(base["home_team"].astype(str)) | set(base["away_team"].astype(str))
    if home_team not in known or away_team not in known:
        raise KeyError("Both teams must exist in the cached match history")

    synthetic = {
        column: np.nan for column in base.columns
    }
    synthetic.update(
        {
            "match_id": "FUTURE-FIXTURE",
            "season": "forecast",
            "match_date": pd.to_datetime(base["match_date"]).max() + pd.Timedelta(days=7),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": 0,
            "away_goals": 0,
            "game_week": float(pd.to_numeric(base.get("game_week"), errors="coerce").max() + 1),
        }
    )
    augmented = pd.concat([base, pd.DataFrame([synthetic])], ignore_index=True)
    built = build_prematch_features(augmented, recent_window=recent_window).frame
    return built.loc[built["match_id"].eq("FUTURE-FIXTURE"), FEATURE_COLUMNS].reset_index(drop=True)
