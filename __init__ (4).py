from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

EVENT_COLUMNS = [
    "event_id", "match_id", "competition", "season", "match_date", "home_team",
    "away_team", "team", "opponent", "period", "minute", "second",
    "timestamp_seconds", "possession_id", "player", "position", "recipient",
    "event_type", "outcome", "x", "y", "end_x", "end_y", "body_part",
    "play_pattern", "under_pressure", "first_time", "assisted", "key_pass",
    "shot_goal"
]


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        events = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        events = pd.json_normalize(raw)
    else:
        raise ValueError(f"Unsupported event file type: {path.suffix}")

    for column in ["match_date"]:
        if column in events:
            events[column] = pd.to_datetime(events[column], errors="coerce")

    bool_columns = ["under_pressure", "first_time", "assisted", "key_pass", "shot_goal"]
    for column in bool_columns:
        if column in events:
            events[column] = events[column].fillna(False).astype(bool)

    return events


def load_matches(path: str | Path) -> pd.DataFrame:
    matches = pd.read_csv(path)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    return matches


def load_lineups(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def match_events(events: pd.DataFrame, match_id: str) -> pd.DataFrame:
    result = events.loc[events["match_id"] == match_id].copy()
    if result.empty:
        raise KeyError(f"Unknown match_id: {match_id}")
    return result.sort_values(["period", "timestamp_seconds", "event_id"]).reset_index(drop=True)


def team_events(events: pd.DataFrame, team: str) -> pd.DataFrame:
    return events.loc[events["team"] == team].copy()
