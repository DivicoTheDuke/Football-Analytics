from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np
import pandas as pd


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def _season_order(values: pd.Series) -> dict[str, int]:
    seasons = sorted(values.dropna().astype(str).unique())
    return {season: index for index, season in enumerate(seasons, start=1)}


def team_form_table(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Build recency-weighted attack and defence strengths from match-level xG."""
    shot_totals = shots.groupby(["match_id", "team"], as_index=False)["xg"].sum()
    home = matches[["match_id", "season", "match_date", "home_team", "away_team"]].merge(
        shot_totals.rename(columns={"team": "home_team", "xg": "xg_for"}),
        on=["match_id", "home_team"], how="left",
    )
    away_xg = shot_totals.rename(columns={"team": "away_team", "xg": "xg_against"})
    home = home.merge(away_xg[["match_id", "away_team", "xg_against"]], on=["match_id", "away_team"], how="left")
    home_rows = home.rename(columns={"home_team": "team", "away_team": "opponent"})

    away = matches[["match_id", "season", "match_date", "home_team", "away_team"]].merge(
        shot_totals.rename(columns={"team": "away_team", "xg": "xg_for"}),
        on=["match_id", "away_team"], how="left",
    )
    home_xg = shot_totals.rename(columns={"team": "home_team", "xg": "xg_against"})
    away = away.merge(home_xg[["match_id", "home_team", "xg_against"]], on=["match_id", "home_team"], how="left")
    away_rows = away.rename(columns={"away_team": "team", "home_team": "opponent"})

    long = pd.concat([
        home_rows[["match_id", "season", "match_date", "team", "opponent", "xg_for", "xg_against"]],
        away_rows[["match_id", "season", "match_date", "team", "opponent", "xg_for", "xg_against"]],
    ], ignore_index=True)
    long[["xg_for", "xg_against"]] = long[["xg_for", "xg_against"]].fillna(0.0)
    order = _season_order(long["season"])
    long["weight"] = long["season"].map(order).astype(float)
    long["weight"] = np.exp((long["weight"] - long["weight"].max()) * 0.35)

    rows = []
    league_xg = np.average(long["xg_for"], weights=long["weight"]) if len(long) else 1.3
    for team, group in long.groupby("team"):
        weight = group["weight"].to_numpy()
        xgf = float(np.average(group["xg_for"], weights=weight))
        xga = float(np.average(group["xg_against"], weights=weight))
        rows.append({
            "team": team,
            "matches": int(len(group)),
            "weighted_xg_for": xgf,
            "weighted_xg_against": xga,
            "attack_strength": xgf / max(league_xg, 0.1),
            "defence_strength": xga / max(league_xg, 0.1),
        })
    return pd.DataFrame(rows).sort_values("attack_strength", ascending=False).reset_index(drop=True)


def attacking_side_profile(events: pd.DataFrame) -> pd.DataFrame:
    """Estimate which flank a team uses in the attacking two-thirds."""
    actions = events.loc[
        events["event_type"].isin(["Pass", "Carry", "Shot"]) & events["x"].ge(52.5)
    ].copy()
    actions["channel"] = pd.cut(
        actions["y"], bins=[-0.01, 22.67, 45.33, 68.01], labels=["Right", "Centre", "Left"]
    ).astype(str)
    counts = actions.groupby(["team", "channel"], observed=False).size().unstack(fill_value=0)
    for column in ["Left", "Centre", "Right"]:
        if column not in counts:
            counts[column] = 0
    shares = counts[["Left", "Centre", "Right"]].div(counts.sum(axis=1), axis=0).fillna(0.0)
    shares.columns = [f"{column.lower()}_share" for column in shares.columns]
    result = shares.reset_index()
    result["preferred_side"] = result[["left_share", "centre_share", "right_share"]].idxmax(axis=1).str.replace("_share", "", regex=False).str.title()
    result["confidence"] = result[["left_share", "centre_share", "right_share"]].max(axis=1)
    return result.sort_values("confidence", ascending=False).reset_index(drop=True)


def team_style_profiles(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, group in events.groupby("team"):
        passes = group.loc[group["event_type"] == "Pass"]
        completed = passes.loc[passes["outcome"].eq("Complete")]
        progressive = group.loc[group.get("end_x", pd.Series(index=group.index, dtype=float)).sub(group["x"]).ge(10)]
        directness = float((passes["end_x"] - passes["x"]).clip(lower=0).mean()) if len(passes) else 0.0
        pressure_rate = float(group["under_pressure"].mean()) if "under_pressure" in group else 0.0
        final_third = float(group["x"].ge(70).mean())
        pass_completion = float(len(completed) / len(passes)) if len(passes) else 0.0
        progressive_rate = float(len(progressive) / len(group)) if len(group) else 0.0
        if directness > 9 and progressive_rate > 0.25:
            label = "Direct progression"
        elif pass_completion > 0.82 and directness < 7:
            label = "Possession circulation"
        elif pressure_rate > 0.35:
            label = "Pressure-resistant transition"
        elif final_third > 0.32:
            label = "Territorial attack"
        else:
            label = "Balanced build-up"
        rows.append({
            "team": team,
            "style_label": label,
            "pass_completion": pass_completion,
            "directness": directness,
            "progressive_action_rate": progressive_rate,
            "final_third_action_rate": final_third,
            "under_pressure_rate": pressure_rate,
        })
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def scorer_probabilities(events: pd.DataFrame, shots: pd.DataFrame, team: str, expected_team_goals: float) -> pd.DataFrame:
    team_shots = shots.loc[shots["team"] == team].copy()
    if team_shots.empty:
        return pd.DataFrame(columns=["player", "shot_share", "xg_share", "expected_goals", "score_probability"])
    player = team_shots.groupby("player", as_index=False).agg(shots=("event_id", "size"), xg=("xg", "sum"))
    total_xg = max(float(player["xg"].sum()), 1e-9)
    total_shots = max(int(player["shots"].sum()), 1)
    player["shot_share"] = player["shots"] / total_shots
    player["xg_share"] = player["xg"] / total_xg
    player["expected_goals"] = expected_team_goals * (0.7 * player["xg_share"] + 0.3 * player["shot_share"])
    player["score_probability"] = 1 - np.exp(-player["expected_goals"])
    return player.sort_values("score_probability", ascending=False).reset_index(drop=True)


def _poisson_probability(goals: int, rate: float) -> float:
    return exp(-rate) * rate**goals / factorial(goals)


@dataclass(frozen=True)
class FixtureForecast:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float
    most_likely_score: str


def forecast_fixture(matches: pd.DataFrame, shots: pd.DataFrame, home_team: str, away_team: str) -> FixtureForecast:
    strengths = team_form_table(matches, shots).set_index("team")
    league_rate = float(shots.groupby(["match_id", "team"])["xg"].sum().mean())
    home = strengths.loc[home_team]
    away = strengths.loc[away_team]
    home_xg = float(np.clip(league_rate * home["attack_strength"] * away["defence_strength"] * 1.08, 0.15, 4.5))
    away_xg = float(np.clip(league_rate * away["attack_strength"] * home["defence_strength"] * 0.94, 0.15, 4.5))
    matrix = np.array([[_poisson_probability(i, home_xg) * _poisson_probability(j, away_xg) for j in range(8)] for i in range(8)])
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return FixtureForecast(home_team, away_team, home_xg, away_xg, home_win, draw, away_win, f"{i}-{j}")


def season_projection(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    rows = {team: {"team": team, "points": 0.0, "expected_goals": 0.0, "expected_goals_against": 0.0} for team in teams}
    for home in teams:
        for away in teams:
            if home == away:
                continue
            forecast = forecast_fixture(matches, shots, home, away)
            rows[home]["points"] += 3 * forecast.home_win + forecast.draw
            rows[away]["points"] += 3 * forecast.away_win + forecast.draw
            rows[home]["expected_goals"] += forecast.home_xg
            rows[home]["expected_goals_against"] += forecast.away_xg
            rows[away]["expected_goals"] += forecast.away_xg
            rows[away]["expected_goals_against"] += forecast.home_xg
    frame = pd.DataFrame(rows.values()).sort_values(["points", "expected_goals"], ascending=False).reset_index(drop=True)
    frame.insert(0, "projected_position", np.arange(1, len(frame) + 1))
    frame["goal_difference"] = frame["expected_goals"] - frame["expected_goals_against"]
    return frame
